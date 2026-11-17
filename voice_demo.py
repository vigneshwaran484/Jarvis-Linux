"""
voice_demo.py — hear the local voices before picking one.

Run:  .venv/bin/python voice_demo.py                 # the shortlist below
      .venv/bin/python voice_demo.py bm_fable am_adam
      .venv/bin/python voice_demo.py --list          # every Kokoro voice name

Names starting bm_/bf_/am_/af_ are Kokoro voices; "piper" plays the Piper voice.
To keep one, put in .env:   TTS_ENGINE=kokoro   KOKORO_VOICE=<name>
"""
import os
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("LOG_FILE", os.path.join(tempfile.mkdtemp(), "voice_demo.log"))

import voice_local  # noqa: E402

SHORTLIST = ["bm_george", "bm_fable", "bm_lewis", "bm_daniel", "am_michael", "am_onyx", "piper"]
LINE = ("Good morning, sir. It is 29 degrees and clear in Chennai. "
        "You have two things today, and the first one starts at ten.")

args = sys.argv[1:]
if args == ["--list"]:
    print("\n".join(voice_local.kokoro_voices()))
    sys.exit(0)

for name in (args or SHORTLIST):
    t = time.time()
    try:
        wav = voice_local.synthesize_piper(LINE) if name == "piper" else voice_local.synthesize_kokoro(LINE, voice=name)
    except Exception as e:
        print(f"{name}: failed ({e})")
        continue
    print(f"▶ {name}   (made in {time.time() - t:.1f}s)")
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        f.write(wav)
        f.flush()
        subprocess.run(["mpv", "--no-terminal", "--quiet", f.name])
    time.sleep(0.4)
