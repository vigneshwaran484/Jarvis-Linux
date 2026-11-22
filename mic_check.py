"""
mic_check.py — find out whether Jarvis can hear you, without running Jarvis.

It records a few seconds from the same microphone Jarvis uses, then reports:
  1. how loud the room is against how loud your voice was
  2. the wake word score for "Hey Jarvis"
  3. what Whisper heard

Run:  .venv/bin/python mic_check.py
      then say:  "Hey Jarvis ... what time is it"
"""
import audioop
import os
import sys
import tempfile
import time

os.environ.setdefault("LOG_FILE", os.path.join(tempfile.mkdtemp(), "mic_check.log"))

import numpy as np                    # noqa: E402
import pyaudio                        # noqa: E402
import speech_recognition as sr       # noqa: E402

import voice_local                    # noqa: E402
import wakeword                       # noqa: E402
from config import WAKEWORD_THRESHOLD, WAKEWORD_CONFIRM_THRESHOLD, WHISPER_MIN_CONFIDENCE  # noqa: E402

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 6

idx, rate = wakeword.get_microphone_details()
detector = wakeword.init_detector()
voice_local.stt_ready()

frames = int(rate * wakeword.WAKEWORD_FRAME_MS / 1000)
pa = pyaudio.PyAudio()
stream = pa.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True,
                 input_device_index=idx, frames_per_buffer=frames)
hp = wakeword.HighPass(rate)
for _ in range(5):
    hp.process(stream.read(frames, exception_on_overflow=False))

print("\nStay quiet for 1 second...")
quiet = [audioop.rms(hp.process(stream.read(frames, exception_on_overflow=False)), 2) for _ in range(12)]
noise = sorted(quiet)[len(quiet) // 2]

print(f'Now say:  "Hey Jarvis ... what time is it"   ({SECONDS:.0f} seconds)')
state, chunks, levels, best = None, [], [], 0.0
end = time.time() + SECONDS
while time.time() < end:
    raw = hp.process(stream.read(frames, exception_on_overflow=False))
    levels.append(audioop.rms(raw, 2))
    raw16, state = audioop.ratecv(raw, 2, 1, rate, 16000, state)
    chunks.append(raw16)
    if detector and detector.enabled:
        score = detector.predict(np.frombuffer(raw16, dtype=np.int16)).get(detector._pred_key, 0.0)
        best = max(best, float(score))
stream.close()
pa.terminate()

loud = sorted(levels)[int(len(levels) * 0.95)]
text, conf = voice_local.transcribe(sr.AudioData(b"".join(chunks), 16000, 2))

print("\n── Result ─────────────────────────────")
print(f"Room noise level : {noise}")
print(f"Your voice level : {loud}   ({loud / max(noise, 1):.1f}x the noise; want 3x or more)")
if best >= WAKEWORD_THRESHOLD:
    verdict = "WAKES (direct)"
elif best >= WAKEWORD_CONFIRM_THRESHOLD and wakeword.confirms_wake(text):
    verdict = "WAKES (borderline score, confirmed by Whisper)"
elif best >= WAKEWORD_CONFIRM_THRESHOLD:
    verdict = "NO WAKE (borderline score, but Whisper did not hear 'Jarvis')"
else:
    verdict = "NO WAKE"
print(f"Wake word score  : {best:.2f}   (direct at {WAKEWORD_THRESHOLD:.2f}, "
      f"Whisper check from {WAKEWORD_CONFIRM_THRESHOLD:.2f})  {verdict}")
print(f"Whisper heard    : {text!r}  confidence {conf:.2f} (needs {WHISPER_MIN_CONFIDENCE:.2f})")
if loud < noise * 3:
    print("\nYour voice barely rises above the noise: move closer to the laptop mic,\n"
          "or use a headset mic and make it the default input.")
