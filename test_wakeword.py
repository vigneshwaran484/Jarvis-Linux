"""
test_wakeword.py — the wake word path end to end without a microphone:
a synthesised "Hey Jarvis" is given the same defects as this laptop's mic
(large constant offset, noise), pushed through HighPass, and fed to
openWakeWord in 80 ms frames exactly like the live listener does.

Needs the Piper voice and wake word model on disk; no mic, Ollama or network.
Run:  .venv/bin/python test_wakeword.py
"""
import io, os, sys, tempfile, wave

os.environ["LOG_FILE"] = os.path.join(tempfile.mkdtemp(), "test.log")

import audioop                      # noqa: E402
import numpy as np                  # noqa: E402
import voice_local, wakeword        # noqa: E402

ok = True
def check(cond, msg):
    global ok
    print(("✅ " if cond else "❌ ") + msg)
    ok &= bool(cond)

RATE = 48000                        # the rate the live listener captures at
FRAME = int(RATE * wakeword.WAKEWORD_FRAME_MS / 1000)

def speech_48k(text):
    w = wave.open(io.BytesIO(voice_local.synthesize(text)))
    pcm, _ = audioop.ratecv(w.readframes(w.getnframes()), 2, 1, w.getframerate(), RATE, None)
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float64)

def mic_like(signal, offset=6400, noise=300, seed=0):
    rng = np.random.default_rng(seed)
    pad = np.zeros(RATE)
    x = np.concatenate([pad, signal * 0.5, pad]) + offset
    return np.clip(x + rng.normal(0, noise, x.size), -32768, 32767).astype(np.int16)

def best_score(detector, pcm48, use_filter=True):
    detector.reset()
    hp, state, best = wakeword.HighPass(RATE), None, 0.0
    for i in range(0, len(pcm48) - FRAME, FRAME):
        raw = pcm48[i:i + FRAME].tobytes()
        if use_filter:
            raw = hp.process(raw)
        raw, state = audioop.ratecv(raw, 2, 1, RATE, 16000, state)
        score = detector.predict(np.frombuffer(raw, dtype=np.int16))[detector._pred_key]
        best = max(best, float(score))
    return best

det = wakeword.WakeWordDetector()
check(det.enabled, "wake word model loads")

hey = mic_like(speech_48k("Hey Jarvis."))
other = mic_like(speech_48k("What is the weather like in Chennai today?"))

# 1. the filter removes the offset without flattening speech
hp = wakeword.HighPass(RATE)
out = np.frombuffer(hp.process(hey.tobytes()), dtype=np.int16).astype(np.float64)
check(abs(hey.astype(np.float64).mean()) > 6000 and abs(out.mean()) < 50, "HighPass removes the constant offset")
check(np.abs(out[:FRAME]).max() < 2000, "no click at the start of the stream")
check(out[RATE:-RATE].std() > 5 * out[:RATE // 2].std(), "speech stays well above the noise after filtering")

# 2. filter state carries across chunks (chunked == one-shot)
hp2 = wakeword.HighPass(RATE)
chunked = b"".join(hp2.process(hey[i:i + FRAME].tobytes()) for i in range(0, len(hey), FRAME))
check(chunked == wakeword.HighPass(RATE).process(hey.tobytes()), "chunked filtering equals one-shot filtering")

# 3. detection
s = best_score(det, hey)
check(s >= wakeword.WAKEWORD_THRESHOLD, f"'Hey Jarvis' fires when streamed in 80 ms frames (score {s:.2f})")
s = best_score(det, other)
check(s < wakeword.WAKEWORD_THRESHOLD, f"other speech does not fire (score {s:.2f})")

# 4. the old approach, kept as a regression guard: one whole clip in a single call scores nothing
det.reset()
pcm16, _ = audioop.ratecv(wakeword.HighPass(RATE).process(hey.tobytes()), 2, 1, RATE, 16000, None)
whole = float(det.predict(np.frombuffer(pcm16, dtype=np.int16))[det._pred_key])
check(whole < wakeword.WAKEWORD_THRESHOLD, f"a whole clip in one predict() call does not fire (score {whole:.2f})")

# 5. Whisper confirmation of borderline scores
for text, want in [("Hey Jarvis what time it is", True), ("hey, Jervis!", True), ("Jar vis, wake up", True),
                   ("what's the service status", False), ("Travis is here", False), ("", False), (None, False)]:
    check(wakeword.confirms_wake(text) is want, f"confirms_wake({text!r}) -> {want}")

pcm16, _ = audioop.ratecv(wakeword.HighPass(RATE).process(hey.tobytes()), 2, 1, RATE, 16000, None)
frames = [pcm16[i:i + 2560] for i in range(0, len(pcm16), 2560)]
check(wakeword._whisper_confirms(frames), "Whisper confirms a real 'Hey Jarvis' clip")
pcm16, _ = audioop.ratecv(wakeword.HighPass(RATE).process(other.tobytes()), 2, 1, RATE, 16000, None)
frames = [pcm16[i:i + 2560] for i in range(0, len(pcm16), 2560)]
check(not wakeword._whisper_confirms(frames), "Whisper rejects other speech")

# 6. the live listener loop, driven by a fake microphone, with the wake model
#    forced to score "borderline" (as it does for this user's real voice)
import threading, time

class FakeStream:
    def __init__(self, pcm48):
        self.data, self.pos = pcm48.tobytes(), 0
    def read(self, n, exception_on_overflow=False):
        chunk = self.data[self.pos:self.pos + n * 2]
        self.pos += n * 2
        if len(chunk) < n * 2:                 # clip finished: quiet room from here on
            time.sleep(0.01)
            chunk += bytes(n * 2 - len(chunk))
        return chunk
    def stop_stream(self): pass
    def close(self): pass

class FakePyAudio:
    clip = None
    def open(self, **kw): return FakeStream(FakePyAudio.clip)
    def terminate(self): pass

def run_listener(clip, cap, seconds=8):
    FakePyAudio.clip = clip
    hits, errors = [], []
    t = threading.Thread(target=wakeword.listen_for_wake_word_continuous,
                         args=(lambda: hits.append(1), errors.append), daemon=True)
    real_score = wakeword.WakeWordDetector.score
    wakeword.WakeWordDetector.score = lambda self, a: min(real_score(self, a), cap)
    try:
        t.start(); t.join(seconds)
    finally:
        wakeword.WakeWordDetector.score = real_score
    return bool(hits), errors

real_pa, real_details = wakeword.pyaudio.PyAudio, wakeword.get_microphone_details
wakeword.pyaudio.PyAudio = FakePyAudio
wakeword.get_microphone_details = lambda: (0, RATE)
wakeword.init_detector()
try:
    woke, errs = run_listener(hey, cap=1.0)
    check(woke and not errs, "listener wakes directly on a clear 'Hey Jarvis'")
    woke, errs = run_listener(hey, cap=0.3)
    check(woke and not errs, "listener wakes on a borderline score once Whisper hears 'Jarvis'")
    woke, errs = run_listener(other, cap=0.3, seconds=5)
    check(not woke and not errs, "listener stays asleep on borderline score when Whisper hears other speech")
    woke, errs = run_listener(hey, cap=0.10, seconds=4)
    check(not woke, "scores under the confirm threshold never reach Whisper")
finally:
    wakeword.pyaudio.PyAudio, wakeword.get_microphone_details = real_pa, real_details

print("\nALL PASSED" if ok else "\nFAILURES")
sys.exit(0 if ok else 1)
