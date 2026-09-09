import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty('voices')

for i, voice in enumerate(voices):
    print(f"{i} → {voice.name}")
engine.setProperty('voice', voices[1].id) 