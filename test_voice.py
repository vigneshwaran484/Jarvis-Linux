import speech_recognition as sr
import edge_tts
import asyncio
import pygame
import os

# ─────────────────────────────────────
# SPEAKING
# ─────────────────────────────────────
async def speak_edge(text):
    await edge_tts.Communicate(text, "en-AU-WilliamNeural").save("response.mp3")

def speak(text):
    asyncio.run(speak_edge(text))
    pygame.mixer.init()
    pygame.mixer.music.load("response.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pass
    pygame.mixer.music.unload()
    os.remove("response.mp3")

# ─────────────────────────────────────
# LISTENING
# ─────────────────────────────────────
def listen():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("🎤 Adjusting for background noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("🎤 Listening... speak now!")

        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            print(f"✅ You said: {text}")
            return text

        except sr.WaitTimeoutError:
            print("⏰ Timeout — didn't hear anything")
            return None
        except sr.UnknownValueError:
            print("❓ Couldn't understand that")
            return None

# ─────────────────────────────────────
# TEST
# ─────────────────────────────────────
if __name__ == "__main__":
    speak("Hello! I am Jarvis. Say something and I will repeat it back.")

    result = listen()

    if result:
        speak(f"You said: {result}")
    else:
        speak("I didn't catch that. Try again.")