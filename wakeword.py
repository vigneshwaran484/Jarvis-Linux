"""
wakeword.py — CPU-only always-listening wake word detection via openWakeWord.

Replaces Google STT polling with a lightweight, offline wake word detector.
Runs on a background thread; on detection, transitions HUD to LISTENING state
and hands off to the STT pipeline (exactly as the hotkey path does).

The hotkey (SUPER+J) remains as a fallback if wake word fails.

Threading: runs on 'wakeword' daemon thread, enqueues state changes through
the same HUD queue as agent_v2's main loop — no race conditions.
"""
import os
import threading
import time
import numpy as np
import pyaudio

import collections
import re

from config import STT_LANGUAGE, WAKEWORD_ENABLED, WAKEWORD_THRESHOLD, WAKEWORD_CONFIRM_THRESHOLD
from jarvis_log import log
import speech_recognition as sr

try:
    import audioop  # stdlib on Python <3.13; the audioop-lts backport on 3.13+
except ImportError:
    audioop = None

WAKEWORD_SAMPLE_RATE = 16000  # openWakeWord's models are trained on 16kHz and hard-assume it
WAKE_BUFFER_SECONDS = 2.5     # audio kept for the Whisper confirmation of a borderline score
WAKE_CONFIRM_TAIL_FRAMES = 4  # frames (~0.3 s) to keep recording after a borderline score, so "Jarvis" is complete
WAKE_CONFIRM_COOLDOWN = 1.5   # seconds between confirmations, so room chatter cannot keep Whisper busy
MIC_SETTLE_SECONDS = 0.3      # audio discarded right after the mic opens (start-up click)
WAKEWORD_FRAME_MS = 80        # openWakeWord's native frame: 1280 samples at 16 kHz


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


class HighPass:
    """Streaming high-pass filter for 16-bit mono PCM.

    Some laptop microphones (this machine's built-in digital mic among them)
    deliver audio riding on a large constant offset: about 6400 out of 32768
    here. That offset dominates any loudness measurement, so an energy-based
    voice detector cannot tell speech from silence, and it wastes headroom for
    the wake word and Whisper models. Removing everything below ~80 Hz takes
    out the offset and fan rumble and leaves speech untouched."""

    def __init__(self, sample_rate, cutoff_hz=80.0):
        from scipy.signal import butter, sosfilt_zi
        self._sos = butter(2, cutoff_hz, btype="highpass", fs=sample_rate, output="sos")
        self._zi_unit = sosfilt_zi(self._sos)
        self._zi = None

    def process(self, raw_bytes):
        from scipy.signal import sosfilt
        x = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float64)
        if x.size == 0:
            return raw_bytes
        if self._zi is None:
            # Start in the steady state for the first sample so the offset
            # does not show up as a loud click at the start of the stream.
            self._zi = self._zi_unit * x[0]
        y, self._zi = sosfilt(self._sos, x, zi=self._zi)
        return np.clip(y, -32768, 32767).astype(np.int16).tobytes()


class _FilteredStream:
    """Wraps SpeechRecognition's microphone stream; read() returns filtered audio."""

    def __init__(self, inner, sample_rate):
        self._inner = inner
        self._filter = HighPass(sample_rate)

    def read(self, size):
        return self._filter.process(self._inner.read(size))

    def close(self):
        return self._inner.close()


class FilteredMicrophone(sr.Microphone):
    """sr.Microphone whose audio passes through HighPass, so Recognizer's
    energy threshold and everything downstream see offset-free audio."""

    def __enter__(self):
        super().__enter__()
        self.stream = _FilteredStream(self.stream, self.SAMPLE_RATE)
        # The offset ramps up over the first moments after the device opens,
        # which gets through the filter as a loud click. Throw that away.
        for _ in range(max(1, int(MIC_SETTLE_SECONDS * self.SAMPLE_RATE / self.CHUNK))):
            self.stream.read(self.CHUNK)
        return self


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
        self._pred_key = None   # key openwakeword uses for this model in predict() output
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

            import openwakeword
            from openwakeword.utils import download_models

            if self.model_name not in openwakeword.MODELS:
                log.warning("Wake word model '%s' unknown. Available: %s. Using 'hey_jarvis'.",
                            self.model_name, list(openwakeword.MODELS))
                self.model_name = "hey_jarvis"

            # openwakeword >= 0.6 ships NO model files in its wheel — every model
            # (the wake word itself plus the shared melspectrogram/embedding
            # feature models) must be fetched once from the project's GitHub
            # releases. download_models() is idempotent: it only downloads
            # files that aren't already present, so this is a no-op after the
            # first run. It needs internet exactly once (~10 MB total).
            onnx_path = openwakeword.MODELS[self.model_name]["model_path"].replace(".tflite", ".onnx")
            if not os.path.exists(onnx_path):
                log.info("Wake word model files missing — downloading '%s' (first run only)", self.model_name)
                download_models([self.model_name])

            # ONNX only. The library's default is inference_framework="tflite",
            # which needs tflite-runtime — a package with no wheels for
            # Python 3.12+ on Linux, so a clean install can never satisfy it.
            # The ONNX path uses onnxruntime (CPU wheels for every current
            # Python) and every pre-trained model is published in both formats.
            # Load ONLY the one model we listen for: cheaper than loading all
            # of them, and immune to any other model's file being absent.
            self.model = Model(wakeword_models=[onnx_path], inference_framework="onnx")
            # When a model is loaded by path, openwakeword keys its predictions by the
            # file stem ("hey_jarvis_v0.1"), not the friendly name ("hey_jarvis").
            # Look the key up once here rather than guessing it in is_detected().
            self._pred_key = next(iter(self.model.models))
            self.enabled = True

            elapsed = time.time() - t0
            log.info("Wake word model '%s' loaded (ONNX, CPU, key=%s) in %.1fs",
                     self.model_name, self._pred_key, elapsed)

        except ImportError:
            log.error(
                "openwakeword not installed. Wake word detection disabled. "
                'Install with: pip install --no-deps "openwakeword>=0.6.0"'
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

    def reset(self):
        """Clear openWakeWord's rolling audio buffer and score history."""
        if self.model is not None:
            with self._lock:
                self.model.reset()

    def score(self, audio_data: np.ndarray) -> float:
        """Wake word score (0-1) for the most recent audio, given one more frame."""
        predictions = self.predict(audio_data)
        return float(predictions.get(self._pred_key, 0.0)) if predictions else 0.0

    def is_detected(self, audio_data: np.ndarray) -> bool:
        """Check if the wake word was detected above threshold."""
        predictions = self.predict(audio_data)
        if not predictions or self._pred_key not in predictions:
            return False
        score = predictions[self._pred_key]
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


# How Whisper tends to spell the name. Deliberately excludes real words such as
# "service" or "travis": this check must not pass on ordinary conversation.
_WAKE_NAME_RE = re.compile(r"\b(jarvis|jervis|jarves|jarvus|javis|jarviss|jar vis)\b")


def confirms_wake(text):
    """True if a transcript contains the assistant's name."""
    return bool(_WAKE_NAME_RE.search(re.sub(r"[^a-z ]", " ", (text or "").lower())))


def _whisper_confirms(frames_16k):
    """Transcribe the buffered audio locally and check it for the wake name."""
    try:
        import voice_local
        if not voice_local.stt_ready():
            return False
        audio = sr.AudioData(b"".join(frames_16k), WAKEWORD_SAMPLE_RATE, 2)
        text, _conf = voice_local.transcribe(audio)
    except Exception as e:
        log.warning("Wake confirmation failed: %s", e)
        return False
    ok = confirms_wake(text)
    log.info("Wake confirmation: heard %r -> %s", text, "wake" if ok else "ignore")
    return ok


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

    # openWakeWord is a STREAMING detector: it keeps its own rolling audio
    # buffer and expects a steady feed of 80 ms frames (1280 samples at 16 kHz),
    # scoring the most recent window on every call. Handing it one whole
    # "phrase" per call (as an earlier version did via Recognizer.listen)
    # scores only the trailing silence and never fires. So read fixed-size
    # frames straight from PyAudio and feed every one of them.
    frames_native = int(native_rate * WAKEWORD_FRAME_MS / 1000)
    pa = pyaudio.PyAudio()
    stream = None
    giveup_reason = None
    try:
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=native_rate,
                         input=True, input_device_index=mic_idx,
                         frames_per_buffer=frames_native)
        _detector.reset()            # forget anything heard before (e.g. Jarvis's own voice)
        highpass = HighPass(native_rate)
        for _ in range(max(1, int(MIC_SETTLE_SECONDS * 1000 / WAKEWORD_FRAME_MS))):
            highpass.process(stream.read(frames_native, exception_on_overflow=False))
        resample_state = None        # audioop.ratecv carries state across calls
        consecutive_errors = 0
        recent = collections.deque(maxlen=int(WAKE_BUFFER_SECONDS * 1000 / WAKEWORD_FRAME_MS))
        confirm_in = None            # frames left before a borderline score is checked with Whisper
        last_confirm = 0.0
        while True:
            try:
                raw_bytes = highpass.process(stream.read(frames_native, exception_on_overflow=False))
                if native_rate != WAKEWORD_SAMPLE_RATE:
                    raw_bytes, resample_state = audioop.ratecv(
                        raw_bytes, 2, 1, native_rate, WAKEWORD_SAMPLE_RATE, resample_state
                    )
                # openWakeWord requires raw int16 PCM — NOT normalized float32.
                audio_data = np.frombuffer(raw_bytes, dtype=np.int16)
                recent.append(raw_bytes)
                score = _detector.score(audio_data)
                if score >= _detector.threshold:
                    print("✅ Wake word detected!")
                    log.info("Wake word detected! Score: %.3f (threshold: %.2f)", score, _detector.threshold)
                    break
                if (confirm_in is None and score >= WAKEWORD_CONFIRM_THRESHOLD
                        and time.time() - last_confirm >= WAKE_CONFIRM_COOLDOWN):
                    log.info("Borderline wake score %.3f; confirming with Whisper", score)
                    confirm_in = WAKE_CONFIRM_TAIL_FRAMES
                elif confirm_in is not None:
                    confirm_in -= 1
                    if confirm_in <= 0:
                        confirm_in, last_confirm = None, time.time()
                        if _whisper_confirms(list(recent)):
                            print("✅ Wake word detected (confirmed by Whisper)!")
                            break
                        _detector.reset()     # audio piled up while transcribing; start clean
                consecutive_errors = 0
            except Exception as e:
                # A persistent failure would otherwise spin and flood jarvis.log.
                consecutive_errors += 1
                log.warning("Audio processing error (%d in a row): %s", consecutive_errors, e)
                if consecutive_errors >= 10:
                    giveup_reason = f"Repeated audio processing errors: {e}"
                    log.error("%s — giving up on wake word listener", giveup_reason)
                    break
                time.sleep(min(0.5 * consecutive_errors, 5))
    except Exception as e:
        giveup_reason = f"Microphone error: {e}"
        log.exception(giveup_reason)
    finally:
        # Release the mic BEFORE calling back: both on_detected's caller and
        # the Google STT fallback open their own stream next.
        try:
            if stream is not None:
                stream.stop_stream()
                stream.close()
        finally:
            pa.terminate()

    if giveup_reason:
        if on_error:
            on_error(giveup_reason)
        return
    on_detected()


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

    with FilteredMicrophone(device_index=mic_idx, sample_rate=sample_rate) as source:
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
