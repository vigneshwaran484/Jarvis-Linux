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
import pyaudio

from config import STT_LANGUAGE, WAKEWORD_ENABLED, WAKEWORD_THRESHOLD
from jarvis_log import log
import speech_recognition as sr

try:
    import audioop  # stdlib on Python <3.13; the audioop-lts backport on 3.13+
except ImportError:
    audioop = None

WAKEWORD_SAMPLE_RATE = 16000  # openWakeWord's models are trained on 16kHz and hard-assume it


# ─────────────────────────────────────────
# Microphone setup
# ─────────────────────────────────────────

# PipeWire exposes a JACK-compatible server that PortAudio's JACK backend
# can see and enumerate with attractive names (e.g. "AB13X Headset Adapter
# Mono") — but on this system, opening ANY device under that hostAPI
# reliably crashes the process with heap corruption (verified directly: a
# minimal PyAudio open()+read() reproduces it every time, independent of
# SpeechRecognition). Never open a device on this hostAPI.
_UNSAFE_HOST_APIS = {"jack audio connection kit"}

# Within the safe ALSA hostAPI, these ALSA PCM plugin names route through
# PipeWire's real audio graph — following whatever device the user has
# actually set as default input, with transparent rate conversion — rather
# than grabbing a raw hw:X,Y device directly. A raw hw: device can open
# without error yet capture near-silence, because PipeWire normally holds
# the physical card for its own exclusive use (verified: hw:0,0 on this
# machine opens fine but reads a near-zero noise floor, while pw-record
# against the named PipeWire device captures real audio at the same time).
_PREFERRED_ALSA_NAMES = ("pulse", "default")


def _safe_input_devices():
    """Yield (index, name) for every input-capable device whose hostAPI
    isn't the one known to crash PortAudio on this system."""
    pa = pyaudio.PyAudio()
    try:
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) <= 0:
                continue
            host_api = pa.get_host_api_info_by_index(info["hostApi"])["name"]
            if host_api.strip().lower() in _UNSAFE_HOST_APIS:
                continue
            yield i, info["name"]
    finally:
        pa.terminate()


def get_microphone_details():
    """
    Pick a (device_index, sample_rate) to open for microphone capture.

    Deliberately prefers ALSA's 'pulse'/'default' PCM plugin — which
    forwards to PipeWire's real audio graph — over both a raw hw:X,Y
    device and any device on PipeWire's JACK-compatibility hostAPI. See
    the module-level comments above _UNSAFE_HOST_APIS and
    _PREFERRED_ALSA_NAMES for what each of those choices avoids; both were
    found by direct testing, not by assumption.
    """
    import sys
    if not sys.platform.startswith("linux"):
        return None, 16000

    try:
        devices = list(_safe_input_devices())
    except Exception as e:
        log.exception("Failed to enumerate audio input devices: %s", e)
        return None, 16000

    for name_target in _PREFERRED_ALSA_NAMES:
        for i, name in devices:
            if name.strip().lower() == name_target:
                return i, 48000

    # No 'pulse'/'default' plugin found (unusual) — fall back to whatever
    # safe device remains rather than failing outright.
    if devices:
        i, name = devices[0]
        log.warning("No ALSA 'pulse'/'default' device found; falling back to %r (idx %d)", name, i)
        return i, 48000

    log.error("No safe input device found — every input-capable device was on an unsafe hostAPI")
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

    def predict(self, audio_data: np.ndarray) -> dict:
        """
        Run inference on a chunk of audio.

        Args:
            audio_data: numpy int16 array of raw 16kHz mono PCM audio. openWakeWord's
                        preprocessor hard-requires dtype int16 (it raises ValueError on
                        float input) and always assumes a 16kHz sample rate internally —
                        there is no sample_rate parameter on Model.predict().

        Returns:
            dict with model predictions (keys are model names, values are scores 0-1)
        """
        if not self.enabled or self.model is None:
            return {}

        try:
            with self._lock:
                predictions = self.model.predict(audio_data)
            return predictions
        except Exception as e:
            log.error("Error during wake word inference: %s", e)
            return {}

    def is_detected(self, audio_data: np.ndarray) -> bool:
        """Check if the wake word was detected above threshold."""
        predictions = self.predict(audio_data)
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

    # openWakeWord's models are trained on 16kHz audio and hard-require it (no
    # sample_rate parameter exists on Model.predict() — it assumes 16kHz always).
    # Some USB/headset mics on this machine refuse to open at 16kHz at all (PyAudio
    # fails silently, stream ends up None) — get_microphone_details() already
    # discovered the rate each device actually needs (e.g. 48kHz). So we capture
    # at THAT native rate and resample down to 16kHz ourselves before inference,
    # rather than fighting the hardware.
    mic_idx, native_rate = get_microphone_details()
    if native_rate != WAKEWORD_SAMPLE_RATE and audioop is None:
        reason = "audioop unavailable — cannot resample mic audio to 16kHz for openWakeWord"
        log.error(reason)
        if on_error:
            on_error(reason)
        return

    giveup_reason = None  # set inside the mic block; on_error is called only after the mic is released
    try:
        _recognizer = sr.Recognizer()
        with sr.Microphone(device_index=mic_idx, sample_rate=native_rate) as source:
            # One-time ambient noise calibration
            _recognizer.adjust_for_ambient_noise(source, duration=0.5)

            resample_state = None  # audioop.ratecv carries state across calls for a smoother resample
            consecutive_errors = 0
            while True:
                try:
                    # Capture a short phrase (SpeechRecognition's listen() blocks until
                    # speech starts, then returns at silence or phrase_time_limit).
                    # NOTE: this is phrase-triggered, not fixed-size streaming — it's a
                    # reasonable fit for openWakeWord's frame-based predict() but the
                    # built-in energy-threshold VAD can clip the very start of the
                    # wake word. A raw continuous PCM stream (e.g. pyaudio directly,
                    # feeding fixed 80ms frames) would be more responsive; TODO for a
                    # follow-up pass if false negatives turn out to be common.
                    audio = _recognizer.listen(source, timeout=5, phrase_time_limit=1.5)
                    raw_bytes = audio.get_raw_data()  # 16-bit PCM at native_rate

                    if native_rate != WAKEWORD_SAMPLE_RATE:
                        raw_bytes, resample_state = audioop.ratecv(
                            raw_bytes, 2, 1, native_rate, WAKEWORD_SAMPLE_RATE, resample_state
                        )

                    # openWakeWord requires raw int16 PCM — NOT normalized float32.
                    # Its preprocessor raises ValueError on any other dtype.
                    audio_data = np.frombuffer(raw_bytes, dtype=np.int16)

                    # Run detection
                    if _detector.is_detected(audio_data):
                        print("✅ Wake word detected!")
                        log.info("Wake word detection triggered")
                        on_detected()
                        return  # Exit listener thread

                    consecutive_errors = 0  # reset on any successful iteration

                except sr.WaitTimeoutError:
                    # Timeout is normal during silence
                    consecutive_errors = 0
                    continue
                except sr.UnknownValueError:
                    # Speech detected but too quiet; try next chunk
                    consecutive_errors = 0
                    continue
                except sr.RequestError:
                    # No internet (shouldn't happen since we're CPU-only)
                    consecutive_errors = 0
                    continue
                except Exception as e:
                    # A persistent bug here (e.g. a bad API kwarg) would otherwise spin
                    # in a tight loop and flood jarvis.log. Back off, and after enough
                    # consecutive failures, give up and let the caller fall back.
                    consecutive_errors += 1
                    log.warning("Audio processing error (%d in a row): %s", consecutive_errors, e)
                    if consecutive_errors >= 10:
                        giveup_reason = f"Repeated audio processing errors: {e}"
                        log.error("%s — giving up on wake word listener", giveup_reason)
                        break  # leave the `with` first so the mic is closed before any fallback opens it
                    time.sleep(min(0.5 * consecutive_errors, 5))
                    continue

        # Mic is released here. Only now hand off to the fallback, which opens its own stream.
        if giveup_reason and on_error:
            on_error(giveup_reason)
        return

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
