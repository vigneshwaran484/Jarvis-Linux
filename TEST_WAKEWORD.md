# Wake Word Detection Test Plan

## Overview
This document describes how to verify the openWakeWord-based wake word detection implementation.

## Prerequisites
- Ollama running: `ollama serve`
- Jarvis environment initialized: `.venv/bin/python agent_v2.py`
- openWakeWord installed (ONNX-only, no tflite-runtime): `pip install --no-deps "openwakeword>=0.6.0"`

## Test 1: Wake Word Detection (Core Feature)

### Steps:
1. Start Jarvis: `.venv/bin/python agent_v2.py`
   - You should see: "👂 Listening for wake word..."
   - HUD should show IDLE state (slow blue pulse)
2. Say clearly: **"Hey Jarvis"**
   - Expected: Console shows "✅ Wake word detected!"
   - HUD transitions to LISTENING state
   - Jarvis speaks: "Yes, I'm listening."
   - `jarvis.log` contains: `Wake word detected! Score: X.XXX (threshold: 0.50)`

### Variations to test:
- Try different pronunciations: "Hey Jarvis", "Jarvis", "Hey J"
- Test with background noise (should still work with threshold tuning)
- Test false positives: Say other words like "Hey Google" (should NOT trigger)

### Failure modes:
- If no detection after 30 seconds:
  - Check `jarvis.log` for "Error during wake word inference"
  - Verify microphone input with: `cat /proc/asound/cards`
  - If CUDA error appears: This is normal (ONNX Runtime warning), system uses CPU
- If trigger too often:
  - Lower `WAKEWORD_THRESHOLD` in `.env` (currently 0.5)
  - Range: 0.0 (too sensitive) to 1.0 (too strict)

---

## Test 2: Fallback to Google STT (Error Resilience)

### Steps:
1. Rename/corrupt the ONNX model files:
   ```bash
   # models live inside the venv: .venv/lib/python3.*/site-packages/openwakeword/resources/models/
   # temporarily rename hey_jarvis_v0.1.onnx there (wakeword.py will re-download it on next start)
   ```
2. Start Jarvis again
3. Expected behavior:
   - `jarvis.log` shows: "Wake word model failed to load"
   - HUD flashes ERROR state briefly
   - Falls back to Google STT: "👂 Waiting for wake word (Google STT fallback)..."
   - Say a wake word, should still detect via STT
4. Restore the model files if you want to continue

### Expected log output:
```
ERROR ... Wake word model failed to load
WARNING ... Wake word detection failed (Wake word model failed to load); falling back to Google STT
INFO ... Using Google STT for wake word detection (fallback mode)
```

---

## Test 3: Hotkey Fallback (Manual Override)

### Current Hotkey Setup:
- **Primary**: `SUPER + J` (defined in `~/.config/hypr/hyprland/keybinds.lua` line 65)
- This triggers `caelestia:aiAssistant` (Hyprland action)
- Maps to: `.venv/bin/python agent_v2.py` via jarvis-waybar.sh

### Steps:
1. Start Jarvis with wake word enabled
2. Instead of saying the wake word, press `SUPER+J` on keyboard
3. Expected: HUD transitions to LISTENING immediately (bypassing wake word)
4. Jarvis speaks: "Yes, I'm listening."

### Fallback chain:
```
Wake Word (openWakeWord)
    ↓ (if fails)
Google STT
    ↓ (if disabled/fails)
Hotkey: SUPER+J (manual)
```

---

## Test 4: Configuration Tuning

### Available settings in `.env`:

```bash
# Enable/disable entire wake word system
WAKEWORD_ENABLED=true

# Which model to use (default: hey_jarvis)
# Available: hey_jarvis, hey_mycroft, alexa, timer, weather
WAKEWORD_MODEL=hey_jarvis

# Detection confidence threshold (0.0-1.0, default: 0.5)
# Lower = more sensitive (more false positives)
# Higher = stricter (might miss actual wake words)
WAKEWORD_THRESHOLD=0.5
```

### Test tuning:
1. Start with default: `WAKEWORD_THRESHOLD=0.5`
2. If false positives (triggers on background noise):
   - Increase to 0.6 or 0.7
   - Test by speaking nearby words
3. If misses valid wake words:
   - Decrease to 0.4 or 0.3
   - Retest with natural speech
4. Log detection scores in `jarvis.log` to guide tuning

### Example log lines to check:
```
Wake word detected! Score: 0.68 (threshold: 0.50)   # Good
Wake word detected! Score: 0.52 (threshold: 0.50)   # Barely above threshold
```

---

## Test 5: Disable Wake Word (Debug/Fallback Mode)

### Steps:
1. Edit `.env`:
   ```bash
   WAKEWORD_ENABLED=false
   ```
2. Restart Jarvis
3. Expected:
   - `jarvis.log` shows: "Wake word detection disabled in config"
   - System immediately falls back to Google STT
   - "👂 Waiting for wake word (Google STT fallback)..."

---

## Test 6: Thread Safety & Responsiveness

### Steps:
1. Start Jarvis with wake word enabled
2. While listening, run system operations:
   - Open a file manager
   - Play music
   - Run CPU-intensive task in another terminal
3. Say the wake word
4. Expected:
   - Wake word still detects within 1-2 seconds
   - No audio stuttering
   - No blocking of other system operations

### CPU/Memory check:
```bash
# In another terminal while Jarvis is listening
watch -n 0.5 'ps aux | grep -E "agent_v2|openwakeword" | grep -v grep'
```

---

## Test 7: Logging & Auditing

### Check log file:
```bash
tail -50 jarvis.log
```

### Expected entries for successful wake word detection:
```
INFO | Wake word listener started (model=hey_jarvis, threshold=0.50)
INFO | Wake word detected! Score: 0.XX (threshold: 0.50)
INFO | Wake word detection triggered
```

### Expected entries for fallback:
```
ERROR | Wake word model failed to load
WARNING | Wake word detection failed (reason); falling back to Google STT
INFO | Using Google STT for wake word detection (fallback mode)
INFO | STT heard: 'hey jarvis'
INFO | Wake word detected via STT
```

---

## Quick Manual Test (No Audio Input Required)

```bash
# Test detector initialization only
.venv/bin/python -c "
from wakeword import init_detector, is_detector_ready
init_detector()
print('Detector ready:', is_detector_ready())
"
```

Expected output:
```
Detector ready: True
```

---

## Troubleshooting Checklist

| Problem | Check | Fix |
|---------|-------|-----|
| "openwakeword not installed" | `pip list \| grep openwakeword` | `pip install --no-deps "openwakeword>=0.6.0"` |
| Wake word never triggers | Check threshold in logs | Adjust `WAKEWORD_THRESHOLD` |
| Too many false positives | Background noise triggers detection | Increase threshold to 0.6-0.7 |
| Hotkey doesn't work | `grep "SUPER + J" ~/.config/hypr/hyprland/keybinds.lua` | Verify Hyprland config |
| Microphone errors | `arecord -l` | Check mic permissions, try different `WAKEWORD_MODEL` |
| CUDA warnings | (Harmless) ONNX Runtime checking GPU | Ignore; system falls back to CPU |

---

## Performance Expectations

- **Startup**: 7-10 seconds (model loading)
- **Detection latency**: 0.5-2 seconds (per audio chunk)
- **CPU usage**: ~5-15% on listening (CPU-only inference)
- **Memory**: ~150-200 MB (ONNX Runtime + model)
- **VRAM**: 0 MB (CPU-only, leaves VRAM free for Ollama)

---

## Next Steps (Future Work)

TODO comments in agent_v2.py mark these enhancements:
- Custom wake word training (openWakeWord supports it)
- Alternative models (hey_mycroft, alexa, etc.)
- Streaming audio optimization
- Hyprland IPC integration for direct window control
