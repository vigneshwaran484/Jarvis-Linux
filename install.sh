#!/usr/bin/env bash
# install.sh — set up the J.A.R.V.I.S virtualenv on Linux.
#
# Steps, in this order:
#   1. CPU-only torch (keeps GPU VRAM free for Ollama)
#   2. everything in requirements.txt
#   3. openwakeword WITHOUT its declared dependencies — its metadata requires
#      tflite-runtime on Linux, which has no wheels for Python 3.12+. Jarvis runs
#      it in ONNX-only mode (see wakeword.py), so tflite is never needed.
#   4. the speech models that make voice work without internet: a Piper voice
#      (text-to-speech) and a Whisper model (speech-to-text)
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
    echo "→ creating .venv"
    python3 -m venv .venv
fi
PIP=.venv/bin/pip

echo "→ CPU torch"
$PIP install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4"

echo "→ requirements.txt"
$PIP install -r requirements.txt

echo "→ openwakeword (ONNX-only, --no-deps)"
$PIP install --no-deps "openwakeword>=0.6.0"

echo "→ downloading wake word model files (openwakeword ships none in its wheel; ~10 MB, needs internet once)"
.venv/bin/python - <<'PY'
import warnings; warnings.filterwarnings("ignore")
from openwakeword.utils import download_models
download_models(["hey_jarvis"])
PY

echo "→ verifying ONNX-only wake word model loads (no tflite-runtime needed)"
.venv/bin/python - <<'PY'
import warnings; warnings.filterwarnings("ignore")
import openwakeword
from openwakeword.model import Model
p = openwakeword.MODELS["hey_jarvis"]["model_path"].replace(".tflite", ".onnx")
m = Model(wakeword_models=[p], inference_framework="onnx")
assert "hey_jarvis_v0.1" in m.models or "hey_jarvis" in m.models, m.models.keys()
print("   ok — loaded:", sorted(m.models))
PY

echo "→ downloading the Piper voice (on-device TTS, ~60 MB, needs internet once)"
VOICE="${PIPER_VOICE:-en_GB-alan-medium}"
mkdir -p voices
if [ ! -f "voices/$VOICE.onnx" ]; then
    .venv/bin/python -m piper.download_voices "$VOICE" --data-dir voices
fi

echo "→ downloading the Kokoro voice model (more natural on-device TTS, ~340 MB, needs internet once)"
KOKORO_URL=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0
for f in kokoro-v1.0.onnx voices-v1.0.bin; do
    [ -f "voices/$f" ] || curl -fL --retry 3 -o "voices/$f" "$KOKORO_URL/$f"
done

echo "→ downloading the Whisper model (on-device STT, ~150 MB for base.en, needs internet once)"
.venv/bin/python - <<'PY'
import os
from faster_whisper import WhisperModel
name = os.getenv("WHISPER_MODEL", "base.en")
WhisperModel(name, device="cpu", compute_type="int8")
print("   ok — loaded:", name)
PY

echo
echo "Done. Run with:  .venv/bin/python agent_v2.py   (or: source .venv/bin/activate && python agent_v2.py)"
echo "Note: 'pip check' will report tflite-runtime missing for openwakeword — expected, harmless."
