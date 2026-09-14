import json
import re
import requests
import subprocess
import webbrowser
import speech_recognition as sr
import edge_tts
import asyncio
import os
import shutil
import threading
import time
import psutil
from pathlib import Path
from datetime import datetime

from reactor import ArcReactor, HUDState
from wakeword import (
    listen_for_wake_word_continuous, listen_for_wake_word_google_stt,
    init_detector, is_detector_ready,
    get_microphone_details
)
from memory import MemoryManager
from tts_stream import StreamingSpeaker
from jarvis_log import log
from config import (
    SCREENSHOT_DIR, VIDEO_DIR,
    LLM_MODEL, LLM_ENDPOINT, LLM_MAX_STEPS, LLM_TIMEOUT,
    TTS_VOICE, STT_LANGUAGE, PHRASE_TIME_LIMIT, STT_MIN_CONFIDENCE,
    WAKEWORD_ENABLED, WAKEWORD_MODEL, WAKEWORD_THRESHOLD,
    WAKE_WORDS, ALLOWED_APPS, ALLOWED_WRITE_EXTENSIONS,
)

# DONE(wake-word): implemented openWakeWord-based detection in wakeword.py
#   Falls back to Google STT if openWakeWord model fails to load.
#   Hotkey (SUPER+J) remains as a manual override/fallback.
# TODO(hyprland-ipc): add tools that talk to Hyprland over its IPC socket
#   ($XDG_RUNTIME_DIR/hypr/$HYPRLAND_INSTANCE_SIGNATURE/.socket.sock) instead of
#   shelling out to hyprctl/ydotool — e.g. focus_window|title, move_to_workspace|n,
#   list_windows|. Register them in TOOLS and the SYSTEM_PROMPT tool list.
# DONE(streaming-tts): sentence-pipelined playback implemented in
#   tts_stream.py (StreamingSpeaker), fed from ask_llm()'s Ollama streaming
#   mode via the _DoneSpeechRouter below. speak() itself is unchanged and
#   still used for short deterministic messages and as the fallback path.

# Suppress ALSA warnings emitted by C libraries during audio initialization
import ctypes
try:
    ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
    def py_error_handler(filename, line, function, err, fmt):
        pass
    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    asound = ctypes.cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(c_error_handler)
except Exception:
    pass

# Create output folders on startup
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# Module-level reactor / memory references — set in __main__
_reactor = None
_memory: MemoryManager = None


def hud(state, detail=""):
    """Push a HUD state from the voice thread (no-op when running headless)."""
    if _reactor:
        _reactor.set_state(state, detail)


# ─────────────────────────────────────────
# 1. SPEAK  (fixes speaking state + temp file cleanup)
# ─────────────────────────────────────────

async def _speak_edge(text, tmp_path):
    await edge_tts.Communicate(text, TTS_VOICE).save(str(tmp_path))

def speak(text, then=None, gen_start_ts=None):
    """Synthesise + play `text` as a single blocking clip. HUD shows SPEAKING,
    then returns to `then` (or whatever state was active before the call).

    Used for short deterministic messages (memory confirmations, errors) and
    as the fallback when streaming TTS (see run_agent/StreamingSpeaker) isn't
    used or its first chunk fails. gen_start_ts, if given, is logged against
    the moment playback actually starts — lets jarvis.log show this old
    blocking path's full latency for comparison against streaming's
    'first audio playback started' log line.
    """
    print(f"🔊 Jarvis: {text}")
    log.info("SPEAK: %s", text)

    prev_state = then or (_reactor.state if _reactor else None)
    hud(HUDState.SPEAKING)

    tmp_path = Path("_jarvis_response.mp3")
    try:
        asyncio.run(_speak_edge(text, tmp_path))
        if gen_start_ts is not None:
            log.info("Blocking TTS: audio playback started %.2fs after generation began",
                     time.time() - gen_start_ts)
        # Use mpv for reliable audio playback on Linux
        subprocess.run(["mpv", "--no-terminal", "--quiet", str(tmp_path)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"⚠️  TTS error: {e}")
        log.exception("TTS failure")
    finally:
        # Always clean up temp file, even if playback crashes
        if tmp_path.exists():
            tmp_path.unlink()
        if prev_state:
            hud(prev_state)


# ─────────────────────────────────────────
# 2. LISTEN
# ─────────────────────────────────────────

_listen_recognizer = sr.Recognizer()
_listen_recognizer.dynamic_energy_threshold = True
_listen_recognizer.pause_threshold = 0.8

def listen():
    """Capture one utterance. Returns (text, reason): text is None on failure and
    `reason` is one of timeout | unclear | low_confidence | stt_error | mic_error."""
    hud(HUDState.LISTENING)
    try:
        mic_idx, sample_rate = get_microphone_details()
        mic = sr.Microphone(device_index=mic_idx, sample_rate=sample_rate)
    except Exception:
        log.exception("Microphone init failed")
        return None, "mic_error"

    try:
        mic.__enter__()  # opening the stream can itself raise (device busy, unplugged)
    except Exception:
        log.exception("Microphone open failed")
        return None, "mic_error"

    try:
        source = mic
        print("🎤 Listening...")
        _listen_recognizer.adjust_for_ambient_noise(source, duration=0.3)
        try:
            audio = _listen_recognizer.listen(
                source, timeout=6, phrase_time_limit=PHRASE_TIME_LIMIT
            )
            # show_all=True exposes Google's confidence so we can reject mumbling
            raw = _listen_recognizer.recognize_google(audio, language=STT_LANGUAGE, show_all=True)
            alternatives = raw.get("alternative", []) if isinstance(raw, dict) else []
            if not alternatives or not alternatives[0].get("transcript", "").strip():
                print("❓ Empty transcript")
                log.warning("STT returned empty transcript")
                return None, "unclear"
            best = alternatives[0]
            text = best["transcript"].strip()
            confidence = float(best.get("confidence", 1.0))  # Google omits it on later alternatives
            if confidence < STT_MIN_CONFIDENCE:
                print(f"❓ Low confidence ({confidence:.2f}): {text}")
                log.warning("STT low confidence %.2f for %r", confidence, text)
                return None, "low_confidence"
            print(f"✅ You said: {text}  (conf {confidence:.2f})")
            log.info("STT: %r conf=%.2f", text, confidence)
            return text, None
        except sr.WaitTimeoutError:
            print("⏰ Timeout — nothing heard")
            return None, "timeout"
        except sr.UnknownValueError:
            print("❓ Couldn't understand")
            log.warning("STT could not understand audio")
            return None, "unclear"
        except sr.RequestError as e:
            print(f"⚠️  STT error: {e}")
            log.error("STT request error: %s", e)
            return None, "stt_error"
    finally:
        try:
            mic.__exit__(None, None, None)
        except Exception:
            pass


# ─────────────────────────────────────────
# 3. LLM
# ─────────────────────────────────────────

class LLMError(RuntimeError):
    """Raised by ask_llm; message is a short reason (unreachable | timeout | error: ...)."""


def ask_llm(messages, on_token=None):
    """
    Call Ollama's /api/chat in streaming mode. If `on_token` is given, it's
    invoked with each incremental text chunk as it arrives — this is what
    lets run_agent() pipeline TTS playback while generation is still in
    progress (see _DoneSpeechRouter below).

    Always returns the FULL accumulated response text (stripped), exactly
    like the old blocking (stream=False) version did — the ReAct loop's
    TOOL:/DONE: parsing further down is untouched, it still runs on this
    complete string after ask_llm() returns. Only how the bytes arrive
    changed, not the response contract.
    """
    try:
        resp = requests.post(
            LLM_ENDPOINT,
            json={"model": LLM_MODEL, "messages": messages, "stream": True},
            timeout=LLM_TIMEOUT,
            stream=True,
        )
        resp.raise_for_status()

        parts = []
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                log.warning("Ollama stream: unparseable line, skipping: %r", line[:200])
                continue

            token = data.get("message", {}).get("content", "")
            if token:
                parts.append(token)
                if on_token:
                    on_token(token)
            if data.get("done"):
                break

        return "".join(parts).strip()
    except requests.exceptions.ConnectionError as e:
        log.error("Ollama unreachable at %s: %s", LLM_ENDPOINT, e)
        raise LLMError("unreachable")
    except requests.exceptions.Timeout:
        log.error("Ollama timed out after %.0fs (model=%s)", LLM_TIMEOUT, LLM_MODEL)
        raise LLMError("timeout")
    except LLMError:
        raise
    except Exception as e:
        log.exception("LLM error")
        raise LLMError(f"error: {e}")


# ─────────────────────────────────────────
# 4. TOOLS
# ─────────────────────────────────────────

def open_app(app_name):
    app_name = app_name.strip().lower()

    # System apps for Linux/Garuda (using generic names and some KDE defaults)
    system_apps = {
        "notepad":    "kate",      # or gedit, kwrite
        "calculator": "kcalc",     # or gnome-calculator
        "explorer":   "thunar",    # default for Celestia rice
        "terminal":   "konsole",
        "browser":    "firefox",
    }
    
    # Check common synonyms
    for key, cmd in system_apps.items():
        if key in app_name:
            try:
                subprocess.Popen([cmd])
                return f"Opened {key}"
            except FileNotFoundError:
                pass

    # Search for .desktop files
    desktop_paths = [
        Path("/usr/share/applications"),
        Path(os.path.expanduser("~/.local/share/applications"))
    ]
    
    for folder in desktop_paths:
        if not folder.exists(): continue
        for desktop_file in folder.rglob("*.desktop"):
            if app_name in desktop_file.stem.lower():
                try:
                    with open(desktop_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.startswith("Exec="):
                                cmd = line.split("=", 1)[1].strip()
                                # Clean up formatting codes like %f, %u
                                cmd = " ".join(arg for arg in cmd.split() if not arg.startswith("%"))
                                subprocess.Popen(cmd, shell=True)
                                return f"Opened {app_name}"
                except Exception as e:
                    print(f"Error reading desktop file: {e}")

    # Fallback: try running directly
    try:
        subprocess.Popen([app_name])
        return f"Opened {app_name}"
    except FileNotFoundError:
        return f"Could not find '{app_name}' on this PC."

def close_app(app_name):
    app_name = app_name.strip().lower()
    # Build a set of possible exact process names to match against
    targets = set()
    mapped = ALLOWED_APPS.get(app_name)
    if mapped:
        targets.add(mapped.lower())
    targets.add(app_name + ".exe")
    targets.add(app_name)

    if len(app_name) < 2:
        return f"Cannot close app with name '{app_name}' (name is too short/ambiguous)."

    killed = 0
    procs_to_kill = []
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            pname = proc.info["name"].lower()
            # Safe matching: exact target match, or prefix match like 'app_name.exe'
            if pname in targets or pname.startswith(app_name + "."):
                procs_to_kill.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    for proc in procs_to_kill:
        try:
            proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Wait for graceful termination, then force-kill survivors
    gone, alive = psutil.wait_procs(procs_to_kill, timeout=3)
    for proc in alive:
        try:
            proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    killed = len(gone) + len(alive)
    if killed:
        return f"Closed {app_name} ({killed} process{'es' if killed > 1 else ''})"
    return f"Could not close {app_name} — it may not be running."

def search_web(query):
    import urllib.parse
    webbrowser.open(f"https://www.google.com/search?q={urllib.parse.quote(query)}")
    return f"Searched Google for: {query}"

def write_file(filename, content):
    path = Path(filename.strip())
    # Security: only allow safe extensions and no path traversal
    if path.suffix.lower() not in ALLOWED_WRITE_EXTENSIONS:
        return f"Cannot write '{path.suffix}' files. Allowed: {ALLOWED_WRITE_EXTENSIONS}"
    if path.is_absolute() or ".." in path.parts:
        return "Cannot write to that path. Relative paths only."
    path.write_text(content, encoding="utf-8")
    return f"File '{filename}' created"

def read_file(filename):
    path = Path(filename.strip())
    if path.is_absolute() or ".." in path.parts:
        return "Cannot read that path."
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"File '{filename}' not found"
    except Exception as e:
        return f"Could not read file: {e}"

def get_system_stats():
    cpu = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    return (
        f"CPU usage is {cpu}%. "
        f"RAM usage is {ram.percent}% ({ram.used // 1024**2} MB / {ram.total // 1024**2} MB). "
        f"Root disk has {disk.free // 1024**3} GB free out of {disk.total // 1024**3} GB."
    )

def take_screenshot(filename=None):
    try:
        # Always use datetime-based naming to prevent overwrites
        filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        full_path = SCREENSHOT_DIR / filename
        
        # Use grim on Wayland (Hyprland) if available
        if shutil.which("grim"):
            subprocess.run(["grim", str(full_path)])
            return f"Screenshot saved to {full_path}"
        else:
            # Fallback to pyautogui
            import pyautogui
            pyautogui.screenshot().save(str(full_path))
            return f"Screenshot saved to {full_path} (used fallback)"
    except Exception as e:
        return f"Could not take screenshot: {e}"

# Recording — using wf-recorder for Wayland/Hyprland
_recording_process = None

def start_recording(dummy=None):
    global _recording_process
    
    if _recording_process is not None and _recording_process.poll() is None:
        return "Already recording."
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = VIDEO_DIR / f"recording_{timestamp}.mp4"
    
    if shutil.which("wf-recorder"):
        print(f"🎥 Recording via wf-recorder → {out_path}")
        _recording_process = subprocess.Popen(
            ["wf-recorder", "-f", str(out_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return "Screen recording started."
    else:
        return "wf-recorder is not installed. Please install it (sudo pacman -S wf-recorder) to record on Hyprland."

def stop_recording(dummy=None):
    global _recording_process
    import signal
    
    if _recording_process is None or _recording_process.poll() is not None:
        return "No recording in progress."
        
    # wf-recorder stops gracefully on SIGINT
    _recording_process.send_signal(signal.SIGINT)
    try:
        _recording_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _recording_process.kill()
        
    _recording_process = None
    return "Screen recording stopped and saved to your video folder."

def get_weather(city=None):
    try:
        if not city:
            loc = requests.get("https://ipwho.is/", timeout=5).json()
            if loc.get("success"):
                lat, lon, city = loc["latitude"], loc["longitude"], loc["city"]
            else:
                lat, lon, city = 13.0827, 80.2707, "Chennai"
        else:
            # Use Open-Meteo geocoding instead of a hardcoded dict
            geo = requests.get(
                f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1",
                timeout=5,
            ).json()
            results = geo.get("results")
            if not results:
                return f"Could not find city '{city}'."
            lat  = results[0]["latitude"]
            lon  = results[0]["longitude"]
            city = results[0]["name"]

        data = requests.get(
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current_weather=true&hourly=relative_humidity_2m",
            timeout=5,
        ).json()

        w    = data["current_weather"]
        humidity = data["hourly"]["relative_humidity_2m"][0]
        desc = {
            0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
            45: "foggy", 48: "foggy", 51: "light drizzle", 53: "moderate drizzle",
            61: "light rain", 63: "moderate rain", 65: "heavy rain",
            80: "rain showers", 95: "thunderstorm",
        }.get(w["weathercode"], "mixed conditions")

        return (
            f"Weather in {city}: {desc}. "
            f"Temperature {w['temperature']}°C. "
            f"Wind {w['windspeed']} km/h. "
            f"Humidity {humidity}%."
        )
    except Exception as e:
        return f"Could not fetch weather: {e}"

def set_volume(level):
    """Set system volume to a percentage (0-100)."""
    try:
        level = max(0, min(100, int(level)))
        # Try wpctl (Pipewire) first, then amixer
        if shutil.which("wpctl"):
            subprocess.run(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{level/100:.2f}"])
        elif shutil.which("amixer"):
            subprocess.run(["amixer", "-D", "pulse", "sset", "Master", f"{level}%"])
        else:
            return "No supported volume control tool found (wpctl/amixer)."
        return f"Volume set to {level}%"
    except Exception as e:
        return f"Could not set volume: {e}"

def get_volume(dummy=None):
    """Get current system volume percentage."""
    try:
        if shutil.which("wpctl"):
            res = subprocess.check_output(["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"]).decode().strip()
            # Example output: Volume: 0.50 [MUTED]
            parts = res.split()
            vol = int(float(parts[1]) * 100)
            muted = " (muted)" if "[MUTED]" in res else ""
            return f"Volume is at {vol}%{muted}"
        elif shutil.which("amixer"):
            res = subprocess.check_output(["amixer", "-D", "pulse", "sget", "Master"]).decode()
            # Parse amixer output (e.g. [50%] [on])
            import re
            vol_match = re.search(r"\[(\d+)%\]", res)
            mute_match = re.search(r"\[(on|off)\]", res)
            if vol_match:
                vol = vol_match.group(1)
                muted = " (muted)" if mute_match and mute_match.group(1) == "off" else ""
                return f"Volume is at {vol}%{muted}"
        return "No supported volume control tool found."
    except Exception as e:
        return f"Could not get volume: {e}"

def mute_toggle(dummy=None):
    """Toggle system mute on/off."""
    try:
        if shutil.which("wpctl"):
            subprocess.run(["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "toggle"])
            return "Mute toggled"
        elif shutil.which("amixer"):
            subprocess.run(["amixer", "-D", "pulse", "sset", "Master", "toggle"])
            return "Mute toggled"
        return "No supported volume control tool found."
    except Exception as e:
        return f"Could not toggle mute: {e}"

def set_brightness(level):
    """Set screen brightness (0-100)."""
    try:
        level = max(0, min(100, int(level)))
        try:
            import screen_brightness_control as sbc
            sbc.set_brightness(level)
            return f"Brightness set to {level}%"
        except Exception:
            if shutil.which("brightnessctl"):
                subprocess.run(["brightnessctl", "set", f"{level}%"])
                return f"Brightness set to {level}%"
            else:
                return "Could not set brightness: screen_brightness_control failed and brightnessctl not found."
    except Exception as e:
        return f"Could not set brightness: {e}"

def get_datetime(dummy=None):
    """Get current date and time."""
    now = datetime.now()
    return f"Date: {now.strftime('%A, %B %d, %Y')}. Time: {now.strftime('%I:%M %p')}."

def get_battery(dummy=None):
    """Get battery percentage and charging status."""
    battery = psutil.sensors_battery()
    if battery is None:
        return "No battery detected (desktop PC)."
    status = "charging" if battery.power_plugged else "discharging"
    return f"Battery is at {int(battery.percent)}%, {status}."

def lock_screen(dummy=None):
    try:
        subprocess.run(["loginctl", "lock-session"])
        return "Screen locked."
    except Exception as e:
        return f"Could not lock screen: {e}"

def media_play_pause(dummy=None):
    try:
        if shutil.which("playerctl"):
            subprocess.run(["playerctl", "play-pause"])
            return "Media toggled (play/pause)."
        return "playerctl not installed."
    except Exception as e:
        return f"Could not toggle media: {e}"

def media_next(dummy=None):
    try:
        if shutil.which("playerctl"):
            subprocess.run(["playerctl", "next"])
            return "Next track playing."
        return "playerctl not installed."
    except Exception as e:
        return f"Could not skip track: {e}"

def media_previous(dummy=None):
    try:
        if shutil.which("playerctl"):
            subprocess.run(["playerctl", "previous"])
            return "Previous track playing."
        return "playerctl not installed."
    except Exception as e:
        return f"Could not skip track: {e}"

def simulate_keybind(combo):
    keycodes = {
        "win": "125", "ctrl": "29", "s": "31",
        "1": "2", "2": "3", "3": "4", "4": "5", "5": "6",
        "6": "7", "7": "8", "8": "9", "9": "10", "0": "11"
    }
    keys = combo.lower().split("+")
    args = ["ydotool", "key"]
    for k in keys:
        if k in keycodes: args.append(f"{keycodes[k]}:1")
    for k in reversed(keys):
        if k in keycodes: args.append(f"{keycodes[k]}:0")
    subprocess.run(args)

def change_window(number):
    """Change window (win+number)."""
    simulate_keybind(f"win+{number}")
    return f"Changed window to {number}."

def change_desktop(number):
    """Change desktop (ctrl+win+number)."""
    simulate_keybind(f"ctrl+win+{number}")
    return f"Changed desktop to {number}."

def special_screen(dummy=None):
    """Change to special screen (win+s)."""
    try:
        subprocess.run(["hyprctl", "dispatch", "togglespecialworkspace"])
        return "Changed to special screen."
    except Exception as e:
        return f"Could not toggle special screen: {e}"


_PLACEHOLDER_ARGS = {"fact", "text", "query", "memory", "argument", "value"}

def remember(text=""):
    """Store an explicit long-term memory."""
    text = (text or "").strip()
    # Small models sometimes echo the prompt's placeholder ("remember|fact")
    # instead of a real fact — refuse rather than store junk.
    if not text or text.lower() in _PLACEHOLDER_ARGS:
        return "Nothing to remember — say the fact itself, e.g. 'remember that my meeting is at 5'."
    entry = _memory.add_memory(text)
    return f"Remembered: {entry.text}"

_current_goal = ""   # set by run_agent; lets memory tools fall back to the user's own words

def recall_memory(query=""):
    """Semantic search over long-term memories."""
    query = (query or "").strip()
    if not query or query.lower() in {"query", "text", "fact", "memory"}:  # model echoed the placeholder
        query = _current_goal
    hits = _memory.search_memory(query)
    if not hits:
        return "No stored memories match that."
    return " | ".join(f"{e.text} (saved {e.timestamp[:10]})" for e, _ in hits)

def forget_memory(query=""):
    """Delete the single best-matching long-term memory."""
    query = (query or "").strip() or _current_goal
    entry = _memory.forget(query)
    return f"Forgot: {entry.text}" if entry else "No stored memory matches that."


TOOLS = {
    "remember":        remember,
    "recall_memory":   recall_memory,
    "forget_memory":   forget_memory,
    "open_app":        open_app,
    "close_app":       close_app,
    "search_web":      search_web,
    "write_file":      write_file,
    "read_file":       read_file,
    "get_system_stats": get_system_stats,
    "get_weather":     get_weather,
    "take_screenshot": take_screenshot,
    "start_recording": start_recording,
    "stop_recording":  stop_recording,
    "set_volume":      set_volume,
    "get_volume":      get_volume,
    "mute_toggle":     mute_toggle,
    "set_brightness":  set_brightness,
    "get_datetime":    get_datetime,
    "get_battery":     get_battery,
    "lock_screen":     lock_screen,
    "media_play_pause":media_play_pause,
    "media_next":      media_next,
    "media_previous":  media_previous,
    "change_window":   change_window,
    "change_desktop":  change_desktop,
    "special_screen":  special_screen,
}


# ─────────────────────────────────────────
# 5. SYSTEM PROMPT
# ─────────────────────────────────────────

SYSTEM_PROMPT = """You are Jarvis, a personal AI assistant.

AVAILABLE TOOLS:
- remember|fact              (store something the user asks you to remember)
- recall_memory|query        (search things the user told you earlier)
- forget_memory|query        (delete a stored memory)
- open_app|app_name
- close_app|app_name
- search_web|query
- write_file|filename|content
- read_file|filename
- get_system_stats|
- get_weather|city
- take_screenshot|filename
- start_recording|
- stop_recording|
- set_volume|level           (0-100, e.g. 50)
- get_volume|
- mute_toggle|
- set_brightness|level       (0-100, e.g. 70)
- get_datetime|
- get_battery|
- lock_screen|
- media_play_pause|
- media_next|
- media_previous|
- change_window|number       (1-0)
- change_desktop|number      (1-0)
- special_screen|

STRICT RESPONSE RULES:
Rule 1: ALWAYS use a tool to get real information. NEVER make up or assume answers.
Rule 1b: The ONLY exception: if the user asks about something that is literally written in the
  KNOWN FACTS or RECENT CONVERSATION context, answer it directly with DONE:.
  Time, date, weather, battery, volume and system stats are NEVER in context — always call the tool.
Rule 2: For weather — call get_weather first. For system stats — call get_system_stats first.
Rule 3: Your ENTIRE response must be ONLY one of these two formats (include the colon):
  TOOL: tool_name|argument
  DONE: one short sentence summarising what happened
Rule 4: No explanation, no thinking, no extra text — only the single line above.
Rule 5: Only ONE tool per response.
Rule 6: After a tool result is given, ALWAYS respond with DONE: followed by a short sentence.

EXAMPLES:
User: what's the weather?
You: TOOL: get_weather|

User: Tool result: Weather in Chennai...
You: DONE: It is 32 degrees with clear skies in Chennai.

User: open spotify
You: TOOL: open_app|spotify

User: Tool result: Opened spotify
You: DONE: Spotify is now open.

User: how much RAM am I using?
You: TOOL: get_system_stats|

User: Tool result: CPU 13%...
You: DONE: Your CPU is at 13 percent and RAM is at 52 percent.

User: what time is it?
You: TOOL: get_datetime|

User: Tool result: Date: Monday... Time: 04:30 PM.
You: DONE: It is 4:30 in the afternoon.

User: what time is my meeting?   (KNOWN FACTS contains: "I have a meeting at 5 pm")
You: DONE: Your meeting is at 5 pm.

User: what database does my project use?   (KNOWN FACTS contains: "my project uses Postgres")
You: DONE: Your project uses Postgres.
"""


# ─────────────────────────────────────────
# 5b. STREAMING-SPEECH ROUTER
# ─────────────────────────────────────────

class _DoneSpeechRouter:
    """
    Watches the incremental tokens of ONE ask_llm() call (one ReAct step).
    If the response turns out to start with 'DONE' — i.e. it's the final
    spoken reply, not a TOOL: call — it engages a StreamingSpeaker and feeds
    it the spoken text as it streams in, so playback of the first sentence
    starts while the LLM is still generating the rest.

    This only ever affects DONE: responses. A TOOL: call (or anything else)
    causes zero audio side effects — ask_llm() still returns the complete
    accumulated text exactly as before, and run_agent()'s existing
    TOOL:/DONE: parsing runs on it completely unchanged.
    """
    _PREFIX = "DONE"

    def __init__(self, gen_start_ts):
        self._gen_start_ts = gen_start_ts
        self._buffer = ""
        self._decided = False
        self.is_done_response = False
        self.speaker: StreamingSpeaker | None = None

    def on_token(self, token: str):
        if self._decided:
            if self.is_done_response:
                self.speaker.feed(token)
            return

        self._buffer += token
        stripped = self._buffer.lstrip()
        if len(stripped) < len(self._PREFIX):
            return  # not enough yet to tell TOOL:/DONE apart

        self._decided = True
        if stripped.startswith(self._PREFIX):
            self.is_done_response = True
            # Mirror the exact stripping run_agent() applies to the full
            # response text below: response[4:].lstrip(":. ")
            remainder = stripped[len(self._PREFIX):].lstrip(":. ")
            self.speaker = StreamingSpeaker(
                voice=TTS_VOICE,
                on_first_audio=lambda: hud(HUDState.SPEAKING),
                on_finished=lambda: hud(HUDState.LISTENING),
                gen_start_ts=self._gen_start_ts,
            )
            if remainder:
                self.speaker.feed(remainder)
        # else: a TOOL: call or a malformed response — no speech, no speaker.


# ─────────────────────────────────────────
# 6. AGENT LOOP
# ─────────────────────────────────────────

def run_agent(user_goal):
    """One ReAct episode. Returns the final spoken reply (or None on failure)."""
    print(f"\n🎯 Goal: {user_goal}")
    print("─" * 40)
    log.info("GOAL: %s", user_goal)
    global _current_goal
    _current_goal = user_goal
    hud(HUDState.THINKING)

    # Explicit memory trigger — bypass the LLM entirely so "remember that..." is deterministic
    fact = _memory.extract_explicit_memory(user_goal)
    if fact:
        entry = _memory.add_memory(fact)
        reply = f"Got it. I'll remember that {entry.text}."
        _memory.record_exchange(user_goal, reply)
        speak(reply)
        return reply

    # Explicit forget trigger — same reasoning: deletion should be deterministic
    what = _memory.extract_forget_command(user_goal)
    if what:
        entry = _memory.forget(what)
        reply = f"Okay, I've forgotten that {entry.text}." if entry else "I couldn't find a memory matching that."
        _memory.record_exchange(user_goal, reply)
        speak(reply)
        return reply

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    context = _memory.get_context(user_goal)
    if context:
        messages.append({"role": "system", "content": "CONTEXT (use if relevant, do not repeat verbatim):\n" + context})
    messages.append({"role": "user", "content": user_goal})

    for step in range(LLM_MAX_STEPS):
        hud(HUDState.THINKING, f"step {step+1}: ollama")
        step_start = time.time()
        log.info("Generation started (step %d)", step + 1)
        router = _DoneSpeechRouter(gen_start_ts=step_start)
        try:
            response = ask_llm(messages, on_token=router.on_token)
        except LLMError as e:
            reason = str(e)
            if router.speaker is not None:
                # Drop anything not yet queued; already-fully-formed sentences
                # already playing finish naturally rather than cutting off
                # mid-word. Don't speak the trailing fragment — the error
                # message below is what should be heard next.
                router.speaker.abort()
            hud(HUDState.ERROR, f"ollama {reason}")
            if reason == "unreachable":
                speak("I'm having trouble thinking right now. Ollama doesn't seem to be running.", then=HUDState.ERROR)
            else:
                speak("I'm having trouble thinking right now. Please try again in a moment.", then=HUDState.ERROR)
            return None

        print(f"\n🧠 [{step+1}] {response}")
        log.info("LLM[%d]: %s", step + 1, response)
        messages.append({"role": "assistant", "content": response})

        # Lenient parse: small models sometimes drop the "TOOL:" prefix or write
        # "FORGET MEMORY|..." — normalise the head and see if it names a tool.
        head = response.split("|", 1)[0].strip()
        head_norm = re.sub(r"^tool\s*:\s*", "", head, flags=re.I).strip().lower().replace(" ", "_")
        if response.startswith("TOOL:") or head_norm in TOOLS:
            # A tool call is always a single line. If the model tacks extra
            # lines on (e.g. "TOOL: remember|fact\nDONE: ..."), ignore them
            # instead of stuffing them into the tool's argument.
            first_line = response.strip().splitlines()[0]
            body      = re.sub(r"^tool\s*:\s*", "", first_line, flags=re.I).strip()
            parts     = body.split("|")
            tool_name = parts[0].strip().lower().replace(" ", "_")
            # tolerate "key=value" / "key: value" style args the model sometimes emits
            tool_args = [re.sub(r"^\w+\s*[=:]\s*", "", a.strip()) for a in parts[1:]]

            if tool_name in TOOLS:
                print(f"🔧 {tool_name}({tool_args})")
                hud(HUDState.THINKING, f"calling: {tool_name}")
                try:
                    result = TOOLS[tool_name](*tool_args)
                except Exception as e:
                    log.exception("Tool %s failed", tool_name)
                    result = f"Tool {tool_name} failed: {e}"
                print(f"✅ Result: {result}")
                log.info("TOOL %s(%s) -> %s", tool_name, tool_args, str(result)[:200])
                messages.append({
                    "role": "user",
                    "content": f"Tool result: {result}. Now respond with DONE: if complete.",
                })
            else:
                messages.append({
                    "role": "user",
                    "content": (
                        f"Tool '{tool_name}' does not exist. "
                        f"Available: {', '.join(TOOLS)}. Try again."
                    ),
                })

        elif response.startswith("DONE"):
            # Accept "DONE: text", "DONE.", or bare "DONE"
            final = response[4:].lstrip(":. ").strip()
            if not final:
                final = "Done."
            _memory.record_exchange(user_goal, final)

            # router.speaker was fed this exact text as it streamed in, and
            # (per HUD requirement) already flipped the HUD to SPEAKING the
            # moment the first sentence started playing — well before this
            # point, which is only reached once the FULL response finished
            # generating. finish() blocks until all queued audio has played,
            # then returns False only if the very first chunk failed to
            # synthesize, in which case we fall back to one plain blocking
            # speak() call so the reply is never silently dropped.
            spoken_via_stream = False
            if router.speaker is not None:
                spoken_via_stream = router.speaker.finish(default_text_if_empty=final)
            if not spoken_via_stream:
                speak(final, gen_start_ts=step_start)
            return final

        else:
            log.warning("Malformed LLM response, re-prompting: %r", response[:120])
            messages.append({
                "role": "user",
                "content": "You must respond with ONLY 'TOOL: ...' or 'DONE: ...'. Try again.",
            })

    log.error("ReAct loop exhausted %d steps for goal %r", LLM_MAX_STEPS, user_goal)
    hud(HUDState.ERROR, "max steps reached")
    speak("I had trouble completing that. Please try again.", then=HUDState.ERROR)
    return None


# ─────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    log.info("=== Jarvis starting (model=%s, stt=%s, tts=%s, wake_word=%s) ===",
             LLM_MODEL, STT_LANGUAGE, TTS_VOICE, WAKEWORD_MODEL if WAKEWORD_ENABLED else "disabled")
    reactor = ArcReactor()
    _reactor = reactor                     # module globals used by speak()/hud()
    _memory = MemoryManager()              # loads FAISS + JSON stores, warms embedder in background
    init_detector()                        # initialize openWakeWord model (CPU-only, background thread)

    # Spoken prompts for each listen() failure reason (Part 3.1)
    LISTEN_ERRORS = {
        "timeout":        None,  # silence is normal — just keep listening
        "unclear":        "I didn't catch that. Could you repeat?",
        "low_confidence": "Sorry, I'm not sure I heard that right. Can you say it again?",
        "stt_error":      "Speech recognition is unavailable right now. Check the internet connection.",
        "mic_error":      "I can't access the microphone.",
    }

    def agent_loop():
        time.sleep(1)
        speak("Jarvis is online. Say Hey Jarvis to wake me up.", then=HUDState.IDLE)

        while True:
            hud(HUDState.IDLE)
            print("\n😴 Idle — say 'Hey Jarvis' to wake up (or press Super+J)")

            # Wake word detection (openWakeWord or Google STT fallback)
            wake_detected = False

            def on_wake_word_detected():
                """Callback when openWakeWord detects the wake word."""
                nonlocal wake_detected
                wake_detected = True

            def on_wake_word_error(reason):
                """Callback if wake word detection fails; fall back to Google STT."""
                nonlocal wake_detected
                log.warning("Wake word detection failed (%s); falling back to Google STT", reason)
                hud(HUDState.ERROR, f"wake word: {reason}")
                time.sleep(1)
                # Fall back to Google STT polling
                try:
                    if listen_for_wake_word_google_stt(WAKE_WORDS):
                        wake_detected = True
                except Exception as e:
                    log.exception("Google STT fallback also failed: %s", e)
                    wake_detected = False

            # Try openWakeWord-based detection first
            if is_detector_ready():
                log.info("Using openWakeWord for wake word detection (model=%s, threshold=%.2f)",
                         WAKEWORD_MODEL, WAKEWORD_THRESHOLD)
                listen_for_wake_word_continuous(on_wake_word_detected, on_wake_word_error)
            else:
                # Fall back to Google STT if openWakeWord is disabled or failed to load
                log.info("openWakeWord not ready; using Google STT fallback")
                if listen_for_wake_word_google_stt(WAKE_WORDS):
                    wake_detected = True

            if wake_detected:
                print("✅ Wake word detected!")
                log.info("Wake word detected — transitioning to listening")
                speak("Yes, I'm listening.", then=HUDState.LISTENING)
            else:
                log.warning("Wake word detection ended without trigger (may be user abort or timeout)")
                time.sleep(1)
                continue

            silent_rounds = 0
            while True:
                user_input, reason = listen()

                if not user_input:
                    if reason == "timeout":
                        silent_rounds += 1
                        if silent_rounds >= 3:      # ~20s of silence → back to idle
                            log.info("No speech for 3 rounds, returning to idle")
                            break
                        continue
                    msg = LISTEN_ERRORS.get(reason)
                    hud(HUDState.ERROR, f"stt: {reason}")
                    if msg:
                        speak(msg, then=HUDState.LISTENING)
                    else:
                        hud(HUDState.LISTENING)
                    continue
                silent_rounds = 0

                lower = user_input.lower()
                is_memory_cmd = (_memory.extract_explicit_memory(lower) is not None
                                 or _memory.extract_forget_command(lower) is not None)

                if not is_memory_cmd and any(w in lower for w in ["bye", "goodbye", "sleep", "that's all"]):
                    speak("Okay, going to sleep. Call me when you need me.", then=HUDState.IDLE)
                    break

                if not is_memory_cmd and any(w in lower for w in ["shutdown", "exit", "quit"]):
                    speak("Shutting down. Goodbye!", then=HUDState.IDLE)
                    log.info("Shutdown requested by user")
                    reactor.request_close()
                    return

                try:
                    run_agent(user_input)
                except Exception:
                    log.exception("Unhandled error in run_agent")
                    hud(HUDState.ERROR, "internal error")
                    speak("Something went wrong with that request.", then=HUDState.ERROR)
                hud(HUDState.LISTENING)

    threading.Thread(target=agent_loop, name="voice", daemon=True).start()

    # Tkinter must run on the main thread
    reactor.start()
