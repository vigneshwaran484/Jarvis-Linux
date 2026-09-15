#!/usr/bin/env bash
# install.sh — set up the J.A.R.V.I.S virtualenv on Linux.
#
# Three steps, in this order:
#   1. CPU-only torch (keeps GPU VRAM free for Ollama)
#   2. everything in requirements.txt
#   3. openwakeword WITHOUT its declared dependencies — its metadata requires
#      tflite-runtime on Linux, which has no wheels for Python 3.12+. Jarvis runs
#      it in ONNX-only mode (see wakeword.py), so tflite is never needed.
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

echo
echo "Done. Run with:  .venv/bin/python agent_v2.py   (or: source .venv/bin/activate && python agent_v2.py)"
echo "Note: 'pip check' will report tflite-runtime missing for openwakeword — expected, harmless."
