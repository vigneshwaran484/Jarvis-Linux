<div align="center">

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Ollama-llama3.2-black?style=for-the-badge&logo=ollama&logoColor=white"/>
<img src="https://img.shields.io/badge/Linux-Hyprland-FCC624?style=for-the-badge&logo=linux&logoColor=black"/>
<img src="https://img.shields.io/badge/Offline-100%25-00d4ff?style=for-the-badge"/>
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>

<br/><br/>

```
   ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
   ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
   ██║███████║██████╔╝██║   ██║██║███████╗
██ ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
 ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝
```

# Jarvis-Linux — Offline Voice AI Assistant

**Fully offline. Iron Man-style. Runs entirely on your Linux machine.**

*Local LLM (Ollama) · ReAct Reasoning Loop · Persistent Memory (FAISS) · Live HUD State · Neural TTS*

<br/>

</div>

---

## 🧠 What is Jarvis-Linux?

Jarvis-Linux is a **privacy-first, fully offline voice assistant** for Linux/Hyprland. Say *"Hey Jarvis"* — it wakes up, listens to your command, reasons using a local LLM (Ollama), retrieves from persistent memory, calls system tools, and responds in neural speech.

**Key features:**
- 🐧 Linux-native (Hyprland, PulseAudio, systemd)
- 🧠 Persistent **short-term** (conversation window) and **long-term** (FAISS semantic search) memory
- 🎨 **Live HUD state machine** (IDLE → LISTENING → THINKING → SPEAKING → ERROR)
- 📝 Timestamped error logging for reliability analysis
- 🔇 Graceful fallbacks for STT/Ollama failures with spoken feedback
- ⚡ CPU-only FAISS + MiniLM embeddings (zero GPU VRAM overhead)

No cloud. No API keys. No internet required. Everything runs on your machine.

---

## 🚀 Quick Start

### Prerequisites
- **Linux** (tested on Arch/Garuda with Hyprland; should work on any distro)
- **Ollama** running locally: `ollama pull llama3.2 && ollama serve`
- **Python 3.10+**
- **mpv** for audio playback (or `playerctl` for media control)
- **grim** (Wayland screenshot) or fallback to pyautogui

### Installation

```bash
# Clone the repo
git clone https://github.com/vigneshwaran484/Jarvis-Linux.git
cd Jarvis-Linux

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install CPU torch first (keeps VRAM free for Ollama)
pip install --index-url https://download.pytorch.org/whl/cpu torch

# Install the rest
pip install -r requirements.txt
```

### Run

```bash
# Interactive mode (console + HUD window) — use the venv's python, NOT the system one
.venv/bin/python agent_v2.py

# or activate the venv first, then plain `python` works:
source .venv/bin/activate
python agent_v2.py

# Or via hotkey (Hyprland, bound in ~/.config/hypr/hyprland/keybinds.lua)
super+shift+j
```

> `ModuleNotFoundError: No module named 'speech_recognition'` means you ran the
> system `python` instead of the venv's — every dependency lives in `.venv/`.

Say **"Hey Jarvis"** to wake up, then speak your command.

---

## 📚 Memory System

Two-tier memory keeps context alive across sessions:

| Store | Capacity | Trigger | Storage |
|-------|----------|---------|---------|
| **Short-term** | Last 10 exchanges | Auto-logged | `memory_data/short_term.json` |
| **Long-term** | Semantic search (FAISS) | "Remember that..." | `memory_data/long_term.json` + embeddings |

**Say one of these to store facts:**
- "Remember that my project uses Postgres"
- "Don't forget I have a meeting at 5"
- "Note that my favorite editor is Neovim"

**The LLM can also call these tools:**
- `remember|fact` — store explicitly
- `recall_memory|query` — search stored facts
- `forget_memory|query` — delete a fact

All stored facts persist across restarts and are searched semantically (cosine similarity on MiniLM L6-v2 embeddings).

---

## 🎨 HUD States

The Arc Reactor GUI shows live pipeline state:

```
  🔵 IDLE       →  Waiting for wake word (slow blue pulse)
  🩵 LISTENING  →  Mic open, STT running (sonar ripple)
  🟠 THINKING   →  ReAct loop (orbit ring + tool name caption)
  🟣 SPEAKING   →  TTS playing (fast purple pulse)
  🔴 ERROR      →  STT/Ollama failure (red strobe + reason)
```

**Thread-safe:** Voice loop enqueues state changes; Tkinter main thread applies them. Every transition is mirrored to `/tmp/jarvis_state` for Waybar integration.

---

## 🛠️ Available Tools

| Tool | Use |
|------|-----|
| `open_app` | Launch desktop apps (Kate, Firefox, Spotify, etc.) |
| `close_app` | Kill running processes |
| `search_web` | Open Google search in your browser |
| `get_system_stats` | CPU, RAM, disk usage |
| `get_weather` | Current weather (via Open-Meteo, no API key) |
| `take_screenshot` | Save to `~/Pictures/jarvis/` |
| `start_recording` / `stop_recording` | Screen capture (wf-recorder) |
| `write_file` / `read_file` | Safe file I/O (.txt, .md, .json, .csv, .log only) |
| `get_datetime` | Current date and time |
| `get_battery` | Battery % and charge status |
| `set_volume` / `get_volume` / `mute_toggle` | Audio control |
| `set_brightness` | Screen brightness (0-100) |
| `lock_screen` | Lock the session |
| `media_play_pause` / `media_next` / `media_previous` | Playback control |
| `change_window` / `change_desktop` | Workspace navigation |
| `special_screen` | Toggle Hyprland special workspace |

---

## 📋 Configuration

Edit `.env` or environment variables:

```bash
# LLM
LLM_MODEL=llama3.2                    # Ollama model name
LLM_ENDPOINT=http://localhost:11434/api/chat
LLM_TIMEOUT=30                        # seconds before fallback
LLM_MAX_STEPS=6                       # ReAct loop limit

# Voice
TTS_VOICE=en-AU-WilliamNeural         # Microsoft Edge TTS voice
STT_LANGUAGE=en-IN                    # Google STT language
PHRASE_TIME_LIMIT=10                  # max seconds per utterance
STT_MIN_CONFIDENCE=0.5                # reject low-confidence transcripts

# Memory
MEMORY_DIR=memory_data
SHORT_TERM_WINDOW=10                  # last N exchanges in context
SHORT_TERM_PERSIST=true               # save short-term across restarts
LONG_TERM_TOP_K=3                     # memories retrieved per query
EMBEDDING_MODEL=all-MiniLM-L6-v2      # sentence-transformers model

# Output
LOG_FILE=jarvis.log
LOG_LEVEL=INFO
SCREENSHOT_DIR=~/Pictures/jarvis
VIDEO_DIR=~/Videos/jarvis
```

---

## 🧪 Testing

```bash
# Memory persistence & retrieval (no mic/LLM needed)
.venv/bin/python test_memory.py

# HUD state cycling (visual demo)
timeout 10 .venv/bin/python reactor.py

# Full integration (requires Ollama running)
.venv/bin/python agent_v2.py
```

---

## 📖 Architecture

```
agent_v2.py
  ├─ listen()           → STT (Google) with confidence floor
  ├─ speak(text)        → TTS (edge-tts) + mpv playback
  ├─ ask_llm(messages)  → Ollama ReAct loop
  ├─ run_agent(goal)    → [explicit memory] → [context injection] → [LLM] → [tools] → [logging]
  └─ agent_loop()       → wake-word → listen → agent → repeat
       │
       └─ hud(state, detail)  → reactor.set_state() (thread-safe queue)

reactor.py
  ├─ HUDState enum     (IDLE, LISTENING, THINKING, SPEAKING, ERROR)
  ├─ ArcReactor        (Tkinter + Tk.Canvas animation)
  └─ _animate()        → drains queue, draws state-specific visuals

memory.py
  ├─ Embedder          (sentence-transformers CPU, with hashed fallback)
  ├─ LongTermMemory    (FAISS IndexFlatIP + JSON metadata)
  ├─ ShortTermMemory   (deque + optional JSON persist)
  └─ MemoryManager     (unified get_context, add_memory, search_memory, forget, etc.)

jarvis_log.py
  └─ rotating logger   (jarvis.log + stderr, timestamps, thread names)
```

**Graceful degradation:** If faiss-cpu or sentence-transformers are missing, the embedder falls back to deterministic hashing and NumPy brute-force search, so Jarvis still boots and remembers facts (less accurately).

---

## 🔮 Future Ideas (TODOs in code)

- **Wake-word:** Replace Google STT polling with offline Porcupine or openWakeWord
- **Hyprland IPC:** Direct socket communication instead of shelling out to hyprctl/ydotool
- **Streaming TTS:** Start playback before synthesis finishes (edge_tts.stream())
- **Web search tool:** Integrate DuckDuckGo or SearXNG for retrieval-augmented queries

---

## 📊 File Manifest

| File | Role |
|------|------|
| `agent_v2.py` | Main ReAct loop, tool definitions, voice I/O, error handling |
| `reactor.py` | Tkinter HUD with state machine and thread-safe animation |
| `memory.py` | FAISS-backed persistent memory (long-term + short-term) |
| `jarvis_log.py` | Rotating logger for reliability audits |
| `wakeword.py` | Wake-word detection (Google STT polling) |
| `config.py` | Centralized config + environment loading |
| `requirements.txt` | Dependencies (torch CPU, faiss-cpu, sentence-transformers, edge-tts, etc.) |
| `jarvis-waybar.sh` | Waybar module script (shows HUD state in the bar) |
| `.env.example` | Template for environment variables |
| `test_memory.py` | Standalone memory persistence test |

---

## 🐛 Reliability & Logging

Every error is logged to `jarvis.log` with:
- Timestamp
- Log level (INFO, WARNING, ERROR, EXCEPTION)
- Thread name
- Message

Useful for:
- **Debugging** STT/Ollama timeouts
- **Demo video** commentary ("see how gracefully it falls back")
- **Academic papers** on reliability & robustness

Example:
```
2026-09-09 22:42:31 | INFO     | MainThread | GOAL: what database does Orion use?
2026-09-09 22:42:31 | INFO     | MainThread | Retrieved 1 long-term memories for 'what database does Orion use?': [('3b2fd32d', 0.68)]
2026-09-09 22:42:37 | ERROR    | MainThread | Ollama timed out after 30s (model=llama3.2)
```

---

## 📜 License

MIT License — free to use, modify, and redistribute. See [LICENSE](LICENSE).

---

## 🙋 Author

Built by Vigneshwaran for offline, privacy-first AI on Linux.
