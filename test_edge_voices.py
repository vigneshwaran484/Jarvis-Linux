import edge_tts
import asyncio
import pygame
import os

async def speak_edge(text, voice="en-AU-WilliamNeural"):
    await edge_tts.Communicate(text, voice).save("response.mp3")

def speak(text):
    asyncio.run(speak_edge(text))
    pygame.mixer.init()
    pygame.mixer.music.load("response.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pass
    pygame.mixer.music.unload()   # ← add this line
    os.remove("response.mp3")

# Test it
speak("Hello! I am Jarvis. How can I assist you today?")
"""voices:
en-US-GuyNeural
en-US-AriaNeural
en-US-DavisNeural
en-GB-RyanNeural
"""