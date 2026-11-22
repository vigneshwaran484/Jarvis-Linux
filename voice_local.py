"""
voice_local.py — on-device speech for Jarvis: Whisper for speech-to-text and
Piper for text-to-speech. Both run on the CPU so the GPU stays free for Ollama.

With STT_ENGINE=whisper and TTS_ENGINE=piper nothing spoken leaves the machine.
Each engine loads lazily and reports failure instead of raising, so the agent
can fall back to the online engines (Google STT / edge-tts) when a model is
missing.

Public API:
    transcribe(audio)  -> (text, confidence)   audio = speech_recognition.AudioData
    synthesize(text)   -> WAV bytes
    stt_ready() / tts_ready() -> bool          (loads the model on first call)
    warm_up()                                  load both in a background thread
"""
import io
import math
import threading
import wave
from pathlib import Path

from jarvis_log import log
from config import (
    WHISPER_MODEL, WHISPER_LANGUAGE, PIPER_VOICE, VOICE_DIR, TTS_ENGINE,
    KOKORO_VOICE, KOKORO_SPEED, KOKORO_MODEL,
)

_stt_lock = threading.Lock()
_tts_lock = threading.Lock()
_whisper = None          # faster_whisper.WhisperModel, or False once loading failed
_piper = None            # piper.PiperVoice, or False once loading failed
_kokoro = None           # kokoro_onnx.Kokoro, or False once loading failed


class LocalVoiceError(RuntimeError):
    pass


# ── Speech-to-text ──────────────────────────────────────────────────────────

def _load_whisper():
    global _whisper
    with _stt_lock:
        if _whisper is None:
            try:
                from faster_whisper import WhisperModel
                # int8 on CPU: a few hundred MB of RAM, no VRAM taken from the LLM
                _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
                log.info("Whisper STT loaded (model=%s, cpu/int8)", WHISPER_MODEL)
            except Exception as e:
                log.error("Whisper STT unavailable: %s", e)
                _whisper = False
        return _whisper


def stt_ready():
    return bool(_load_whisper())


def transcribe(audio):
    """Transcribe one utterance. Returns (text, confidence 0-1).
    Text is '' when Whisper judges the clip to be silence or noise."""
    model = _load_whisper()
    if not model:
        raise LocalVoiceError("whisper model not loaded")

    import numpy as np
    pcm = audio.get_raw_data(convert_rate=16000, convert_width=2)
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

    segments, _info = model.transcribe(
        samples,
        language=WHISPER_LANGUAGE or None,
        beam_size=5,
        vad_filter=True,                  # drops leading/trailing silence
        condition_on_previous_text=False,
    )
    segments = list(segments)
    if not segments:
        return "", 0.0

    text = " ".join(s.text.strip() for s in segments).strip()
    # avg_logprob is the mean token log-probability: exp() of it is a usable
    # 0-1 confidence. Weight by duration so a long clear segment outvotes a blip.
    total = sum(max(s.end - s.start, 0.01) for s in segments)
    confidence = sum(math.exp(s.avg_logprob) * max(s.end - s.start, 0.01) for s in segments) / total
    if max(s.no_speech_prob for s in segments) > 0.8 and confidence < 0.5:
        return "", confidence             # Whisper hallucinating on noise
    return text, confidence


# ── Text-to-speech ──────────────────────────────────────────────────────────

def voice_path():
    return Path(VOICE_DIR) / f"{PIPER_VOICE}.onnx"


def _load_piper():
    global _piper
    with _tts_lock:
        if _piper is None:
            try:
                from piper import PiperVoice
                path = voice_path()
                if not path.exists():
                    raise FileNotFoundError(
                        f"{path} missing; run: python -m piper.download_voices "
                        f"{PIPER_VOICE} --data-dir {VOICE_DIR}")
                _piper = PiperVoice.load(str(path))
                log.info("Piper TTS loaded (voice=%s)", PIPER_VOICE)
            except Exception as e:
                log.error("Piper TTS unavailable: %s", e)
                _piper = False
        return _piper


def _load_kokoro():
    global _kokoro
    with _tts_lock:
        if _kokoro is None:
            try:
                from kokoro_onnx import Kokoro
                model, voices = Path(VOICE_DIR) / KOKORO_MODEL, Path(VOICE_DIR) / "voices-v1.0.bin"
                for f in (model, voices):
                    if not f.exists():
                        raise FileNotFoundError(f"{f} missing; see install.sh for the download")
                _kokoro = Kokoro(str(model), str(voices))
                log.info("Kokoro TTS loaded (model=%s, voice=%s)", KOKORO_MODEL, KOKORO_VOICE)
            except Exception as e:
                log.error("Kokoro TTS unavailable: %s", e)
                _kokoro = False
        return _kokoro


def kokoro_voices():
    k = _load_kokoro()
    return sorted(k.get_voices()) if k else []


def synthesize_kokoro(text, voice=None, speed=None):
    """Text -> WAV bytes using Kokoro."""
    k = _load_kokoro()
    if not k:
        raise LocalVoiceError("kokoro model not loaded")
    import numpy as np
    voice = voice or KOKORO_VOICE
    lang = "en-gb" if voice[:1] == "b" else "en-us"     # bm_/bf_ are British voices
    with _tts_lock:
        samples, rate = k.create(text, voice=voice, speed=speed or KOKORO_SPEED, lang=lang)
    if samples is None or len(samples) == 0:
        raise LocalVoiceError("kokoro produced no audio")
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())
    return buf.getvalue()


def tts_ready():
    return bool(_load_kokoro() if TTS_ENGINE == "kokoro" else _load_piper())


def synthesize(text):
    """Text -> WAV bytes with the configured local engine (TTS_ENGINE)."""
    if TTS_ENGINE == "kokoro":
        return synthesize_kokoro(text)
    return synthesize_piper(text)


def synthesize_piper(text):
    """Text -> WAV bytes using Piper."""
    voice = _load_piper()
    if not voice:
        raise LocalVoiceError("piper voice not loaded")
    buf = io.BytesIO()
    with _tts_lock:                       # one synthesis at a time per voice
        with wave.open(buf, "wb") as wav:
            voice.synthesize_wav(text, wav)
    data = buf.getvalue()
    if len(data) <= 44:                   # header only
        raise LocalVoiceError("piper produced no audio")
    return data


def warm_up(stt=True, tts=True):
    """Load the models off the main thread so the first command isn't slow."""
    def _run():
        if stt:
            _load_whisper()
        if tts:
            tts_ready()
    threading.Thread(target=_run, name="voice-warmup", daemon=True).start()
