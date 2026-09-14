"""
wakeword.py — CPU-only always-listening wake word detection via openWakeWord.

Replaces Google STT polling with a lightweight, offline wake word detector.
Runs on a background thread; on detection, transitions HUD to LISTENING state
and hands off to the STT pipeline (exactly as the hotkey path does).

The hotkey (SUPER+J) remains as a fallback if wake word fails.

Threading: runs on 'wakeword' daemon thread, enqueues state changes through
the same HUD queue as agent_v2's main loop — no race conditions.
"""
import threading
import time
import numpy as np
from pathlib import Path

from config import STT_LANGUAGE, WAKEWORD_ENABLED, WAKEWORD_THRESHOLD
from jarvis_log import log
import speech_recognition as sr


# ─────────────────────────────────────────
# Microphone setup (reused from previous code)
# ─────────────────────────────────────────

def get_microphone_details():
    """Get microphone index and sample rate, prioritizing USB/headset."""
    import sys
    if sys.platform.startswith("linux"):
        names = sr.Microphone.list_microphone_names()
        # Priority 1: Hardware USB / Headset mic devices requiring 48kHz
        for i, name in enumerate(names):
            nlower = name.lower()
            if any(k in nlower for k in ["usb", "headset", "adapter"]):
                return i, 48000
        # Priority 2: Fallback to PulseAudio or Default
        for target in ["pulse", "default"]:
            for i, name in enumerate(names):
                if name.lower() == target:
                    return i, 16000
    return None, 16000


def get_microphone_index():
    idx, _ = get_microphone_details()
    return idx


# ─────────────────────────────────────────
# OpenWakeWord detector
# ─────────────────────────────────────────

class WakeWordDetector:
    """
    CPU-only wake word detection via openWakeWord.

    Pre-trained models available: hey_jarvis, hey_mycroft, alexa, timer, weather
    See: https://github.com/dscripka/openWakeWord
    """

    def __init__(self, model_name: str = "hey_jarvis", threshold: float = WAKEWORD_THRESHOLD):
        self.model_name = model_name
        self.threshold = threshold
        self.model = None
        self.enabled = False
        self._lock = threading.Lock()
        self._load_model()

    def _load_model(self):
        """Load the openWakeWord model on CPU."""
        try:
            # Lazy import so missing openwakeword doesn't crash the whole agent
            from openwakeword.model import Model

            log.info("Loading openWakeWord models on CPU (using model '%s', threshold=%.2f)",
                     self.model_name, self.threshold)
            t0 = time.time()

            # openWakeWord.Model loads all available pre-trained models by default.
            # It defaults to CPU when CUDA is unavailable.
            self.model = Model()
            self.enabled = True

            elapsed = time.time() - t0
            available_models = list(self.model.models.keys())
            if self.model_name not in available_models:
                log.warning(
                    "Wake word model '%s' not available. Available: %s. Using 'hey_jarvis' as fallback.",
                    self.model_name, available_models
                )
                self.model_name = "hey_jarvis"
            log.info("Wake word models loaded successfully in %.1fs (available: %s)", elapsed, available_models)

        except ImportError:
            log.error(
                "openwakeword not installed. Wake word detection disabled. "
                "Install with: pip install openwakeword"
            )
            self.enabled = False
        except Exception as e:
            log.exception("Failed to load wake word models: %s", e)
            self.enabled = False

    def predict(self, audio_data: np.ndarray, sample_rate: int = 16000) -> dict:
        """
        Run inference on a chunk of audio.

        Args:
            audio_data: numpy array of PCM audio (mono or stereo)
            sample_rate: sample rate in Hz (default 16000)

        Returns:
            dict with model predictions (keys are model names, values are scores 0-1)
        """
        if not self.enabled or self.model is None:
            return {}

        try:
            with self._lock:
                predictions = self.model.predict(audio_data, sample_rate=sample_rate)
            return predictions
        except Exception as e:
            log.error("Error during wake word inference: %s", e)
            return {}

    def is_detected(self, audio_data: np.ndarray, sample_rate: int = 16000) -> bool:
        """Check if the wake word was detected above threshold."""
        predictions = self.predict(audio_data, sample_rate)
        if not predictions or self.model_name not in predictions:
            return False
        score = predictions[self.model_name]
        detected = score >= self.threshold
        if detected:
            log.info("Wake word detected! Score: %.3f (threshold: %.2f)", score, self.threshold)
        return detected


# Module-level singleton (loaded at startup)
_detector = None


def init_detector():
    """Initialize the wake word detector (call from main thread at startup)."""
    global _detector
    if not WAKEWORD_ENABLED:
        log.info("Wake word detection disabled in config (WAKEWORD_ENABLED=false)")
        return None
    _detector = WakeWordDetector()
    return _detector


def is_detector_ready():
    """Check if the detector is loaded and ready."""
    return _detector is not None and _detector.enabled


# ─────────────────────────────────────────
# Continuous listening (runs on background thread)
# ─────────────────────────────────────────

def listen_for_wake_word_continuous(on_detected, on_error=None):
    """
    Continuously listen for the wake word on the audio input.

    Args:
        on_detected: callable (no args) to invoke when wake word is detected
        on_error: optional callable (reason_str) if an error occurs

    Blocks until on_detected() is called. Safe to call from any thread.
    """
    if not WAKEWORD_ENABLED:
        log.warning("Wake word listening disabled; will block waiting for hotkey")
        return

    if not is_detector_ready():
        reason = "Wake word model failed to load"
        log.error(reason)
        if on_error:
            on_error(reason)
        return

    print("👂 Listening for wake word...")
    log.info("Wake word listener started (model=%s, threshold=%.2f)", _detector.model_name, _detector.threshold)

    mic_idx, sample_rate = get_microphone_details()

    try:
        _recognizer = sr.Recognizer()
        with sr.Microphone(device_index=mic_idx, sample_rate=sample_rate) as source:
            # One-time ambient noise calibration
            _recognizer.adjust_for_ambient_noise(source, duration=0.5)

            while True:
                try:
                    # Capture audio without blocking forever on silence
                    audio = _recognizer.listen(source, timeout=5, max_seconds=0.5)

                    # Convert audio to numpy array for openWakeWord
                    audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16).astype(np.float32)
                    audio_data /= 32768  # Normalize to [-1, 1]

                    # Run detection
                    if _detector.is_detected(audio_data, sample_rate):
                        print("✅ Wake word detected!")
                        log.info("Wake word detection triggered")
                        on_detected()
                        return  # Exit listener thread

                except sr.WaitTimeoutError:
                    # Timeout is normal during silence
                    continue
                except sr.UnknownValueError:
                    # Speech detected but too quiet; try next chunk
                    continue
                except sr.RequestError:
                    # No internet (shouldn't happen since we're CPU-only)
                    continue
                except Exception as e:
                    log.warning("Audio processing error: %s (will retry)", e)
                    continue

    except Exception as e:
        reason = f"Microphone error: {e}"
        log.exception(reason)
        if on_error:
            on_error(reason)


# ─────────────────────────────────────────
# Fallback: Google STT polling
# ─────────────────────────────────────────

def listen_for_wake_word_google_stt(wake_words=None, timeout_seconds=300):
    """
    Legacy: listen using Google STT (requires internet).

    Falls back to this if openWakeWord fails to load.
    Blocks until a wake word is detected.
    """
    if wake_words is None:
        from config import WAKE_WORDS
        wake_words = WAKE_WORDS

    print("👂 Waiting for wake word (Google STT fallback)...")
    log.info("Using Google STT for wake word detection (fallback mode)")

    _recognizer = sr.Recognizer()
    _recognizer.dynamic_energy_threshold = True
    _recognizer.pause_threshold = 0.6

    mic_idx, sample_rate = get_microphone_details()
    start_time = time.time()

    with sr.Microphone(device_index=mic_idx, sample_rate=sample_rate) as source:
        _recognizer.adjust_for_ambient_noise(source, duration=0.5)

        while time.time() - start_time < timeout_seconds:
            try:
                audio = _recognizer.listen(source, timeout=5, phrase_time_limit=4)
                text = _recognizer.recognize_google(audio, language=STT_LANGUAGE).lower().strip()

                print(f"🔈 Heard: '{text}'")
                log.info("STT heard: %r", text)

                if any(wake in text for wake in wake_words):
                    print("✅ Wake word detected!")
                    log.info("Wake word detected via STT")
                    return True

            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                log.warning("Google STT error: %s — will retry", e)
                time.sleep(2)
            except Exception as e:
                log.warning("Unexpected error in wake word (STT): %s", e)
                continue

    log.warning("Wake word timeout after %d seconds", timeout_seconds)
    return False


if __name__ == "__main__":
    # Quick test
    init_detector()
    print("Wake word detector ready:", is_detector_ready())
    print("Model:", _detector.model_name if _detector else "N/A")
    print("\nNote: to actually test, call listen_for_wake_word_continuous() in the agent loop")
