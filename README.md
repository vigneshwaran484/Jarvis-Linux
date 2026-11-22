<div align="center">

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Ollama-qwen3%3A4b--instruct-black?style=for-the-badge&logo=ollama&logoColor=white"/>
<img src="https://img.shields.io/badge/Linux-Hyprland-FCC624?style=for-the-badge&logo=linux&logoColor=black"/>
<img src="https://img.shields.io/badge/Voice%20%2B%20LLM-on--device-00d4ff?style=for-the-badge"/>
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

**Iron Man-style. Voice and LLM run entirely on your Linux machine.**

*Local LLM (Ollama) · On-device Whisper STT + Piper TTS · ReAct Reasoning Loop · Calendar & Daily Briefing · Persistent Memory (FAISS) · Live HUD State*

<br/>

</div>

---

## 🧠 What is Jarvis-Linux?

Jarvis-Linux is a **privacy-first, fully offline voice assistant** for Linux/Hyprland. Say *"Hey Jarvis"* — it wakes up, listens to your command, reasons using a local LLM (Ollama), retrieves from persistent memory, calls system tools, and responds in neural speech.

**Key features:**
- 🎙️ **On-device voice**: openWakeWord wake word, Whisper speech-to-text and Kokoro (or Piper) text-to-speech all run on the CPU; the LLM runs on the GPU through Ollama
- 📅 **Calendar and daily briefing**: reads the local khal calendar, flags clashing events, and "good morning" gives weather + schedule + unread mail + battery in one reply
- 🐧 Linux-native (Hyprland, PulseAudio, systemd)
- 🧠 Persistent **short-term** (conversation window) and **long-term** (FAISS semantic search) memory
- 🎨 **Live HUD state machine** (IDLE → LISTENING → THINKING → SPEAKING → ERROR)
- 📝 Timestamped error logging for reliability analysis
- 🔇 Graceful fallbacks for STT/Ollama failures with spoken feedback
- ⚡ CPU-only FAISS + MiniLM embeddings (zero GPU VRAM overhead)

No cloud and no API keys. Listening, thinking and speaking all happen on your machine.

**What still uses the internet:** the tools that fetch live data (`get_weather`, `search_web`, `get_unread_email`), and the online speech engines (Google STT, edge-tts), which are used only as a fallback if a local speech model is missing or fails.

---

## 🚀 Quick Start

### Prerequisites
- **Linux** (tested on Arch/Garuda with Hyprland; should work on any distro)
- **Ollama** running locally: `ollama pull qwen3:4b-instruct && ollama serve` (2.5 GB, fits a 6 GB GPU; scored 27/28 on `bench_tools.py` against 26/28 for `llama3.2` at similar speed)
- **khal** for the calendar tools (optional)
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

# One-shot install (CPU torch first so VRAM stays free for Ollama, then
# requirements.txt, then openwakeword without its tflite-runtime dependency)
./install.sh
```

Or the same steps by hand:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt
pip install --no-deps "openwakeword>=0.6.0"
```

> **Why `--no-deps` for openwakeword?** Its PyPI metadata requires
> `tflite-runtime` on Linux, which has no wheels for Python 3.12+. Jarvis runs
> openWakeWord in ONNX-only mode (`inference_framework="onnx"`), which never
> imports tflite, so that dependency is simply skipped. `pip check` will keep
> reporting it as missing — that's expected and harmless.
>
> **Model files:** openwakeword ≥ 0.6 ships no model files in its wheel.
> `install.sh` downloads the `hey_jarvis` model (~10 MB, from the project's
> GitHub releases), and `wakeword.py` re-downloads automatically if the files
> are ever missing — so the very first run needs internet once. Everything
> after that is fully offline.

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
| `get_calendar` | Read the local khal calendar for today, tomorrow, a weekday, a date or the week, and point out clashing events |
| `daily_briefing` | "Good morning": weather, today's calendar with clashes, what's next, unread mail and battery in one reply |
| `get_unread_email` | Unread count plus sender and subject of the newest few (IMAP, read-only, optional) |
| `open_app` | Launch desktop apps (Kate, Firefox, Spotify, etc.) |
| `close_app` | Kill running processes |
| `search_web` | Look up facts and speak the answer (Tavily → DuckDuckGo → Wikipedia; opens the browser if nothing is found) |
| `open_in_browser` | Open a Google search in your browser |
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
LLM_MODEL=qwen3:4b-instruct           # Ollama model name
LLM_ENDPOINT=http://localhost:11434/api/chat
LLM_TIMEOUT=30                        # seconds before fallback
LLM_MAX_STEPS=6                       # ReAct loop limit
LLM_CIRCUIT_FAILURE_THRESHOLD=2       # Ollama failures in a row before failing fast
LLM_CIRCUIT_RECOVERY_SECONDS=30       # fail-fast window before retrying Ollama

# Web search (optional key; works without one)
TAVILY_API_KEY=                       # enables full web results
WEB_SEARCH_TIMEOUT=8

LLM_THINK=false                       # reasoning models (qwen3...) skip the thinking phase
HOME_CITY=                            # weather city when none is spoken (empty = guess from IP)
JARVIS_PERSONA=...                    # personality line in the system prompt (empty disables)

# Voice
STT_ENGINE=whisper                    # whisper (on-device) | google (online)
TTS_ENGINE=kokoro                     # kokoro | piper (on-device) | edge (online)
KOKORO_VOICE=bm_george                # try others with: python voice_demo.py
WHISPER_MODEL=base.en                 # tiny.en | base.en | small.en | small
PIPER_VOICE=en_GB-alan-medium         # voice file under voices/
TTS_VOICE=en-AU-WilliamNeural         # edge-tts voice (TTS_ENGINE=edge, or fallback)
STT_LANGUAGE=en-IN                    # Google STT language (STT_ENGINE=google, or fallback)
PHRASE_TIME_LIMIT=10                  # max seconds per utterance
WAKEWORD_THRESHOLD=0.5                # wake word score that wakes Jarvis directly
WAKEWORD_CONFIRM_THRESHOLD=0.15       # borderline scores above this are checked by Whisper for "Jarvis"
STT_MIN_CONFIDENCE=0.5                # reject low-confidence transcripts

# Email (optional, read-only; Gmail needs an app password)
MAIL_ADDRESS=
MAIL_APP_PASSWORD=
MAIL_IMAP_HOST=imap.gmail.com

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

# Circuit breaker, web search fallbacks, command matching (no mic/LLM/network)
.venv/bin/python test_reliability.py

# Calendar date phrases, clash detection, spoken summary (no khal needed)
.venv/bin/python test_calendar.py

# Mail header decoding and the IMAP flow against a fake server (no network)
.venv/bin/python test_mail.py

# Tool-selection accuracy and latency of one or more Ollama models
.venv/bin/python bench_tools.py llama3.2 qwen3:4b

# Wake word path with this laptop's mic defects simulated (no mic needed)
.venv/bin/python test_wakeword.py

# Can Jarvis hear you? Records a few seconds and reports levels, wake score, transcript
.venv/bin/python mic_check.py

# Listen to the available local voices
.venv/bin/python voice_demo.py

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

- **Hyprland IPC:** Direct socket communication instead of shelling out to hyprctl/ydotool
- **Calendar writes:** add events by voice (the calendar tool is read-only today)
- **Richer HUD:** a larger animated interface than the current reactor orb

---

## 📊 File Manifest

| File | Role |
|------|------|
| `agent_v2.py` | Main ReAct loop, tool definitions, voice I/O, error handling |
| `reactor.py` | Tkinter HUD with state machine and thread-safe animation |
| `memory.py` | FAISS-backed persistent memory (long-term + short-term) |
| `jarvis_log.py` | Rotating logger for reliability audits |
| `wakeword.py` | Wake-word detection (openWakeWord, with Google STT polling as the fallback) |
| `config.py` | Centralized config + environment loading |
| `requirements.txt` | Dependencies (torch CPU, faiss-cpu, sentence-transformers, edge-tts, etc.) |
| `jarvis-waybar.sh` | Waybar module script (shows HUD state in the bar) |
| `.env.example` | Template for environment variables |
| `calendar_tool.py` | Reads the khal calendar, resolves spoken dates, finds clashing events |
| `mail_tool.py` | Read-only unread-mail summary over IMAP (headers only, nothing marked as read) |
| `voice_local.py` | On-device speech: Whisper (STT) and Kokoro or Piper (TTS), all on the CPU |
| `tts_stream.py` | Sentence-by-sentence streaming speech playback |
| `mic_check.py` | Records from the mic and reports noise, voice level, wake word score and transcript |
| `voice_demo.py` | Plays a sample line in each local voice |
| `bench_tools.py` | Scores how reliably an Ollama model picks the right tool |
| `circuit_breaker.py` | Fails fast while Ollama is down instead of waiting out every timeout |
| `web_search.py` | Keyless web lookup backends (Tavily / DuckDuckGo / Wikipedia) |
| `test_memory.py` | Standalone memory persistence test |
| `test_reliability.py` | Breaker, web search and command-matching tests |
| `test_calendar.py` | Date-phrase, clash-detection and summary tests |
| `test_mail.py` | Mail parsing and IMAP-flow tests against a fake server |

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

**Circuit breaker:** after `LLM_CIRCUIT_FAILURE_THRESHOLD` Ollama failures in a row (unreachable, timeout or 5xx), Jarvis stops calling it and answers *"My brain is still offline"* instantly for `LLM_CIRCUIT_RECOVERY_SECONDS`. After that it sends one probe request: if that works, normal operation resumes, and if not, the fail-fast window restarts. A missing model (HTTP 404) gets its own message telling you to `ollama pull` it.

---

## 📜 License

MIT License — free to use, modify, and redistribute. See [LICENSE](LICENSE).

---

## 🙋 Author

Built by Vigneshwaran for offline, privacy-first AI on Linux.
