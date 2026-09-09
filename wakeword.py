import speech_recognition as sr
from config import WAKE_WORDS, STT_LANGUAGE

_recognizer = sr.Recognizer()
_recognizer.dynamic_energy_threshold = True
_recognizer.energy_threshold = 300
_recognizer.pause_threshold = 0.6


def get_microphone_details():
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

def listen_for_wake_word_multi(wake_words=None):
    """
    Listens continuously until one of the wake_words is detected.
    Uses Google STT — same engine as the main listen(), so accuracy is high.
    Requires internet. No API key or account needed.
    """
    if wake_words is None:
        wake_words = WAKE_WORDS

    print("👂 Waiting for wake word...")

    mic_idx, sample_rate = get_microphone_details()
    with sr.Microphone(device_index=mic_idx, sample_rate=sample_rate) as source:
        # Calibrate once at the start of this wake-word session
        _recognizer.adjust_for_ambient_noise(source, duration=0.5)


        while True:
            try:
                audio = _recognizer.listen(source, timeout=5, phrase_time_limit=4)

                text = _recognizer.recognize_google(
                    audio, language=STT_LANGUAGE
                ).lower().strip()

                print(f"🔈 Heard: '{text}'")

                if any(wake in text for wake in wake_words):
                    print("✅ Wake word detected!")
                    return True

            except sr.WaitTimeoutError:
                # Nothing heard in 5s — perfectly normal, just keep listening
                continue
            except sr.UnknownValueError:
                # Speech detected but not understood — normal, keep going
                continue
            except sr.RequestError as e:
                print(f"⚠️  Google STT error: {e} — retrying in 2s...")
                import time
                time.sleep(2)
            except Exception as e:
                print(f"⚠️  Unexpected error in wake word: {e}")
                continue