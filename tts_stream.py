"""
tts_stream.py — sentence-pipelined streaming TTS playback.

Turns a stream of incoming text tokens (as they arrive from an LLM) into
speech that starts playing on the FIRST complete sentence, while later
sentences are still being generated and/or synthesized — instead of waiting
for the entire response before saying anything.

Public interface:
    split_into_sentences(text, keep_remainder=False)
    StreamingSpeaker(voice, on_first_audio=None, on_finished=None, gen_start_ts=None)
        .feed(token)                        — call repeatedly as text streams in
        .finish(default_text_if_empty=None) -> bool
        .abort()                            — drop anything not yet queued, used
                                               when the underlying generation fails

Uses edge_tts's Communicate.stream() — the current streaming API (verified
against the installed edge-tts 7.2.8: Communicate.save() is just a thin
wrapper that writes stream() chunks straight to disk) — so each sentence's
audio is fetched via the same modern path.

Pipeline (two background threads per StreamingSpeaker instance):
    feed() (caller's thread — e.g. the loop reading Ollama's HTTP stream)
        -> sentence queue
    _synth_worker (background thread)
        -> synthesizes sentence N+1 while N plays
        -> audio queue
    _play_worker (background thread)
        -> plays queued audio strictly in arrival order, back-to-back
           (single synth thread processes the sentence queue in strict FIFO
           order, so audio is always enqueued for playback in the same
           order it was spoken — no re-ordering buffer needed)

Error handling:
    - A failure synthesizing the FIRST chunk (index 0) aborts the rest of
      the stream (no further chunks are synthesized) and finish() returns
      False so the caller can fall back to a plain, whole-response speak().
    - A failure on any later chunk is logged and skipped; playback carries
      on with the next chunk that does succeed.
"""
from __future__ import annotations

import asyncio
import queue
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import edge_tts

from jarvis_log import log

# ─────────────────────────────────────────
# Sentence splitting
# ─────────────────────────────────────────

# Find a sentence-ending punctuation run followed by whitespace (or end of
# string). Python's `re` can't do variable-length lookbehind, so false
# splits after abbreviations are filtered afterwards with a plain suffix
# check instead of trying to bake them into the regex itself.
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?]+(?=\s|$)")

# Common abbreviations that end in a period but aren't sentence ends.
# Not exhaustive by design — a full NLP sentence tokenizer is out of scope.
_ABBREVIATIONS = (
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.",
    "etc.", "e.g.", "i.e.", "approx.", "no.", "vol.", "fig.",
    "a.m.", "p.m.", "u.s.", "u.k.", "inc.", "ltd.", "co.", "ph.d.",
)
_MAX_ABBR_LEN = max(len(a) for a in _ABBREVIATIONS)


def _ends_with_abbreviation(text_up_to_boundary: str) -> bool:
    tail = text_up_to_boundary[-_MAX_ABBR_LEN:].lower()
    return any(tail.endswith(a) for a in _ABBREVIATIONS)


def split_into_sentences(text: str, keep_remainder: bool = False):
    """
    Split `text` into complete sentences on '.', '!' or '?' boundaries,
    skipping likely-false splits right after a common abbreviation.

    If keep_remainder is False (default): returns a list of sentences,
    treating the whole text as complete (trailing text with no terminal
    punctuation is still returned as a final "sentence").

    If keep_remainder is True: returns (sentences, remainder), where
    `remainder` is the trailing text after the last real sentence boundary
    — i.e. an incomplete sentence still being generated. The caller is
    expected to hold onto `remainder` and prepend it to the next chunk fed
    in (this is exactly what StreamingSpeaker.feed() does).
    """
    sentences = []
    last_end = 0
    for m in _SENTENCE_BOUNDARY_RE.finditer(text):
        end = m.end()
        if _ends_with_abbreviation(text[:end]):
            continue
        sentence = text[last_end:end].strip()
        if sentence:
            sentences.append(sentence)
        last_end = end

    remainder = text[last_end:]
    if keep_remainder:
        return sentences, remainder

    tail = remainder.strip()
    if tail:
        sentences.append(tail)
    return sentences


# ─────────────────────────────────────────
# Streaming speaker
# ─────────────────────────────────────────

class StreamingSpeaker:
    """
    Feed it text as it streams from an LLM; it plays back complete sentences
    via edge-tts as soon as each one is available, synthesizing the next
    sentence while the current one plays.
    """

    # If fed text never hits a sentence boundary, force a flush at this many
    # buffered characters so one abnormally long, unpunctuated response
    # can't block all speech indefinitely.
    _FORCE_FLUSH_CHARS = 400

    def __init__(self, voice: str, on_first_audio=None, on_finished=None,
                 gen_start_ts: Optional[float] = None, tmp_dir: Path = Path("/tmp")):
        self.voice = voice
        self._on_first_audio = on_first_audio
        self._on_finished = on_finished
        self._gen_start_ts = gen_start_ts if gen_start_ts is not None else time.time()
        self._tmp_dir = tmp_dir

        self._text_buffer = ""
        self._sentence_q: "queue.Queue" = queue.Queue()
        self._audio_q: "queue.Queue" = queue.Queue()
        self._next_index = 0
        self._first_chunk_ok: Optional[bool] = None   # None=undecided, True/False once chunk 0 resolves
        self._any_sentence_fed = False
        self._finished = False

        self._synth_thread = threading.Thread(target=self._synth_worker, daemon=True, name="tts-synth")
        self._play_thread = threading.Thread(target=self._play_worker, daemon=True, name="tts-play")
        self._synth_thread.start()
        self._play_thread.start()

    # ── feeding ──
    def feed(self, token: str):
        """Call repeatedly as text streams in from the LLM."""
        if not token:
            return
        self._text_buffer += token
        sentences, self._text_buffer = split_into_sentences(self._text_buffer, keep_remainder=True)
        for s in sentences:
            self._queue_sentence(s)

        if len(self._text_buffer) >= self._FORCE_FLUSH_CHARS:
            cut = self._text_buffer.rfind(" ")
            if cut == -1:
                cut = len(self._text_buffer)
            self._queue_sentence(self._text_buffer[:cut])
            self._text_buffer = self._text_buffer[cut:].lstrip()

    def _queue_sentence(self, sentence: str):
        sentence = sentence.strip()
        if not sentence:
            return
        self._any_sentence_fed = True
        self._sentence_q.put((self._next_index, sentence))
        self._next_index += 1

    def finish(self, default_text_if_empty: Optional[str] = None) -> bool:
        """
        Call once no more tokens are coming. Flushes any trailing partial
        sentence, waits for all queued audio to finish playing, and returns
        True if streaming succeeded (or there was nothing to say), False if
        the FIRST chunk failed to synthesize — meaning the caller should
        fall back to a plain, whole-text speak() call instead.
        """
        remainder = self._text_buffer.strip()
        self._text_buffer = ""
        if remainder:
            self._queue_sentence(remainder)
        elif not self._any_sentence_fed and default_text_if_empty:
            self._queue_sentence(default_text_if_empty)

        self._sentence_q.put(None)  # sentinel: no more sentences coming
        self._synth_thread.join()
        self._play_thread.join()
        self._finished = True

        if self._on_finished:
            self._on_finished()

        # _first_chunk_ok stays None if nothing was ever queued (empty
        # response, no default given) — treat that as "nothing to fall
        # back for", i.e. success.
        return self._first_chunk_ok is not False

    def abort(self):
        """
        Stop immediately without speaking anything not already queued —
        used when the underlying LLM call fails mid-stream. Sentences
        already fully formed and queued before the failure still finish
        playing (so speech doesn't cut off mid-word); the incomplete
        trailing fragment is dropped rather than spoken half-formed.
        """
        if self._finished:
            return
        self._text_buffer = ""
        self._sentence_q.put(None)
        self._synth_thread.join()
        self._play_thread.join()
        self._finished = True
        if self._on_finished:
            self._on_finished()

    # ── synth thread: one sentence at a time, strictly in order ──
    def _synth_worker(self):
        while True:
            item = self._sentence_q.get()
            if item is None:
                self._audio_q.put(None)
                return
            index, sentence = item

            if self._first_chunk_ok is False:
                # First chunk already failed — the caller is falling back to
                # a single whole-text speak() call; drain without wasting
                # time synthesizing anything further.
                continue

            try:
                audio_bytes = asyncio.run(self._synthesize(sentence))
                if not audio_bytes:
                    raise RuntimeError("edge-tts returned no audio data")
                tmp_path = self._tmp_dir / f"_jarvis_stream_{index}.mp3"
                tmp_path.write_bytes(audio_bytes)
                self._audio_q.put((index, tmp_path))
                if index == 0:
                    self._first_chunk_ok = True
            except Exception as e:
                log.warning("Streaming TTS: chunk %d synthesis failed (%r): %s", index, sentence[:60], e)
                if index == 0:
                    self._first_chunk_ok = False
                    log.error("Streaming TTS: first chunk failed — caller will fall back to whole-response speak()")

    async def _synthesize(self, sentence: str) -> bytes:
        communicate = edge_tts.Communicate(sentence, self.voice)
        buf = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.extend(chunk["data"])
        return bytes(buf)

    # ── playback thread: strictly in order, back-to-back ──
    def _play_worker(self):
        while True:
            item = self._audio_q.get()
            if item is None:
                return
            index, path = item
            try:
                if index == 0:
                    elapsed = time.time() - self._gen_start_ts
                    log.info("Streaming TTS: first audio playback started %.2fs after generation began", elapsed)
                    if self._on_first_audio:
                        self._on_first_audio()
                subprocess.run(
                    ["mpv", "--no-terminal", "--quiet", str(path)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except Exception as e:
                log.warning("Streaming TTS: playback of chunk %d failed: %s", index, e)
            finally:
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass
