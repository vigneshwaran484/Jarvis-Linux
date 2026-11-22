"""
reactor.py — J.A.R.V.I.S. Tactical Command Center & Arc Reactor HUD

A dual-mode native desktop interface (Compact Mini-Orb <-> Full Command Center):
- Center Hub: Multi-ring rotating Arc Reactor core, state reticles, 36-bar audio
  spectrum analyzer, real-time streaming response ticker, and neural/tool
  network topology graph (6 category hubs + 29 tool satellites with animated
  synaptic data packets when tools fire).
- Left Panel: Live system telemetry (CPU/RAM sparklines, disk, battery),
  khal Calendar Agenda with automatic conflict/clash flagging, and FAISS
  Semantic Memory Vault inspector.
- Right Panel: Subsystem status matrix (LLM + Circuit Breaker, STT, TTS,
  Wake Word), live ReAct execution pipeline & conversation trace, quick-action
  tactical triggers, and an interactive command console (type or click to talk
  to Jarvis without using the mic).

Thread-safe: all state, event, and telemetry updates from worker threads pass
through lock-free queues and are rendered on the Tkinter main thread at ~33 FPS.
Every state transition is mirrored to /tmp/jarvis_state for Waybar integration.
"""
from __future__ import annotations

import json
import math
import os
import queue
import random
import threading
import time
import tkinter as tk
from collections import deque
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

try:
    import psutil
except ImportError:
    psutil = None

STATE_FILE = "/tmp/jarvis_state"


class HUDState(str, Enum):
    IDLE      = "idle"        # waiting for wake word / hotkey
    LISTENING = "listening"   # mic open, STT running
    THINKING  = "thinking"    # ReAct loop / Ollama; detail = tool being called
    SPEAKING  = "speaking"    # TTS playing
    ERROR     = "error"       # STT failure, Ollama timeout, ...

    @classmethod
    def coerce(cls, value):
        if isinstance(value, cls):
            return value
        v = str(value).lower()
        if v == "sleeping":          # legacy name used by older agent code
            return cls.IDLE
        return cls(v)


# Visual palette & motion profile per state
_VISUALS = {
    HUDState.IDLE: dict(
        rgb=(0, 175, 235),
        glow=(8, 45, 75),
        speed=1.4,
        radius=50,
        swing=2.5,
        label="STANDBY // ONLINE",
        sub="AWAITING WAKE WORD OR COMMAND",
        badge_bg="#062236",
    ),
    HUDState.LISTENING: dict(
        rgb=(0, 245, 212),
        glow=(0, 75, 68),
        speed=3.8,
        radius=55,
        swing=5.0,
        label="VOICE CAPTURE ACTIVE",
        sub="HIGH-PASS AUDIO STREAMING TO STT",
        badge_bg="#042f2b",
    ),
    HUDState.THINKING: dict(
        rgb=(255, 158, 0),
        glow=(85, 48, 0),
        speed=5.5,
        radius=54,
        swing=6.0,
        label="REACT NEURAL REASONING",
        sub="SYNTHESIZING TOOL CALLS & CONTEXT",
        badge_bg="#331d00",
    ),
    HUDState.SPEAKING: dict(
        rgb=(185, 103, 255),
        glow=(58, 22, 90),
        speed=8.5,
        radius=58,
        swing=9.0,
        label="VOCAL SYNTHESIS ACTIVE",
        sub="STREAMING NEURAL AUDIO OUTPUT",
        badge_bg="#270e40",
    ),
    HUDState.ERROR: dict(
        rgb=(255, 56, 76),
        glow=(90, 12, 20),
        speed=12.0,
        radius=55,
        swing=8.0,
        label="SYSTEM FAULT DETECTED",
        sub="DIAGNOSTIC ALERT // CHECK PIPELINE",
        badge_bg="#3b0810",
    ),
}

# Names written to the waybar state file (keeps the old "sleeping" contract)
_FILE_NAMES = {HUDState.IDLE: "sleeping"}

# Tool Topology Categories (all 29 tools from agent_v2.py)
_TOOL_CLUSTERS = [
    ("BRIEF & SCHEDULE", "#00e5ff", -148, [
        "daily_briefing", "get_calendar", "get_unread_email", "get_datetime",
    ]),
    ("MEMORY VAULT", "#b967ff", -92, [
        "remember", "recall_memory", "forget_memory",
    ]),
    ("WEB & INTEL", "#38bdf8", -32, [
        "search_web", "get_weather", "open_in_browser",
    ]),
    ("SYSTEM & POWER", "#00f5d4", 32, [
        "get_system_stats", "get_battery", "set_brightness", "lock_screen",
    ]),
    ("AUDIO & MEDIA", "#ffbe0b", 92, [
        "set_volume", "get_volume", "mute_toggle",
        "media_play_pause", "media_next", "media_previous",
    ]),
    ("DESKTOP & FILES", "#ff6b6b", 148, [
        "open_app", "close_app", "write_file", "read_file",
        "take_screenshot", "start_recording", "stop_recording",
        "change_window", "change_desktop", "special_screen",
    ]),
]


def rgb_to_hex(rgb) -> str:
    return "#%02x%02x%02x" % (
        max(0, min(255, int(rgb[0]))),
        max(0, min(255, int(rgb[1]))),
        max(0, min(255, int(rgb[2]))),
    )


def blend_rgb(c1, c2, t: float):
    t = max(0.0, min(1.0, float(t)))
    return (
        c1[0] + (c2[0] - c1[0]) * t,
        c1[1] + (c2[1] - c1[1]) * t,
        c1[2] + (c2[2] - c1[2]) * t,
    )


class ArcReactor:
    FULL_W, FULL_H = 1240, 760
    MINI_W, MINI_H = 240, 260

    def __init__(self, start_compact: bool = False):
        self.root = tk.Tk()
        self.root.title("J.A.R.V.I.S // TACTICAL COMMAND HUD")
        self._compact = bool(start_compact or os.getenv("JARVIS_HUD_MODE", "").lower() == "mini")
        w = self.MINI_W if self._compact else self.FULL_W
        h = self.MINI_H if self._compact else self.FULL_H
        self.W, self.H = w, h

        self.root.geometry(f"{w}x{h}")
        self.root.minsize(self.MINI_W, self.MINI_H)
        self.root.configure(bg="#03070e")
        self.root.resizable(True, True)

        self.canvas = tk.Canvas(
            self.root, width=w, height=h, bg="#03070e", highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True)

        # Input bar frame (embedded at bottom of full view)
        self._input_frame = tk.Frame(
            self.root, bg="#071324", highlightbackground="#00b4d8", highlightthickness=1
        )
        self._prompt_lbl = tk.Label(
            self._input_frame,
            text=" COMMAND > ",
            bg="#071324",
            fg="#00e5ff",
            font=("JetBrains Mono", 9, "bold"),
        )
        self._prompt_lbl.pack(side="left", padx=(6, 2))

        self._entry_var = tk.StringVar()
        self._entry = tk.Entry(
            self._input_frame,
            textvariable=self._entry_var,
            bg="#040b16",
            fg="#e2f1ff",
            insertbackground="#00e5ff",
            relief="flat",
            font=("JetBrains Mono", 10),
        )
        self._entry.pack(side="left", fill="x", expand=True, padx=4, pady=5)
        self._entry.bind("<Return>", self._on_submit_entry)

        self._send_btn = tk.Label(
            self._input_frame,
            text=" EXECUTE ↵ ",
            bg="#003554",
            fg="#00e5ff",
            font=("JetBrains Mono", 8, "bold"),
            cursor="hand2",
            padx=8,
            pady=4,
        )
        self._send_btn.pack(side="right", padx=6, pady=3)
        self._send_btn.bind("<Button-1>", self._on_submit_entry)

        # Bindings
        self.canvas.bind("<Configure>", self._on_configure)
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.root.bind("<F11>", lambda _e: self.toggle_mode())
        self.root.bind("<Escape>", self._on_escape)

        # Core state
        self.pulse = 0
        self._state = HUDState.IDLE
        self.detail = ""
        self._state_since = time.time()
        self._boot_time = time.time()
        self._queue: "queue.Queue" = queue.Queue()
        self._event_q: "queue.Queue" = queue.Queue()
        self._close_requested = False
        self.running = True

        # Interactive callbacks & hitboxes
        self._command_callback: Optional[Callable[[str], None]] = None
        self._wake_callback: Optional[Callable[[], None]] = None
        self._memory_delete_callback: Optional[Callable[[str], bool]] = None
        self._event_callback: Optional[Callable[[dict], None]] = None
        self._stop_callback: Optional[Callable[[], bool]] = None
        self._armed_delete: tuple[Optional[str], float] = (None, 0.0)   # (memory id, armed at)
        self._refresh_slow = False          # force the 15s memory/calendar tick to run now
        self._hitboxes: list[tuple[float, float, float, float, str, object]] = []
        self._hover_id: Optional[str] = None

        # Live visual data
        self._active_tool: str = ""
        self._active_tool_ts: float = 0.0
        self._tool_history: dict[str, float] = {}     # tool_name -> last_called_ts
        self._packets: list[dict] = []                 # animated synaptic packets
        self._step_info: str = "READY"
        self._stream_text: str = ""
        self._last_user: str = ""
        self._last_reply: str = ""
        self._trace_log: deque[dict] = deque(maxlen=18)
        self._audio_bars: list[float] = [0.08] * 36
        self._audio_level: float = 0.0

        # Background telemetry cache
        self._sys_lock = threading.Lock()
        self._cpu_hist: deque[float] = deque([12.0] * 36, maxlen=36)
        self._ram_hist: deque[float] = deque([40.0] * 36, maxlen=36)
        self._cpu_pct: float = 0.0
        self._ram_pct: float = 0.0
        self._ram_used_gb: float = 0.0
        self._ram_total_gb: float = 0.0
        self._disk_pct: float = 0.0
        self._disk_free_gb: float = 0.0
        self._bat_pct: Optional[int] = None
        self._bat_charging: bool = False
        self._calendar_items: list[dict] = []
        self._calendar_clashes: list[str] = []
        self._memories: list[dict] = []
        self._subsystems: dict[str, tuple[str, str]] = {
            "LLM BRAIN": ("qwen3:4b", "ONLINE"),
            "CIRCUIT": ("CLOSED", "OK"),
            "STT VOICE": ("WHISPER", "READY"),
            "TTS VOICE": ("PIPER", "READY"),
            "WAKE WORD": ("ONNX", "ARMED"),
            "FAISS MEM": ("MINILM", "SYNCED"),
        }

        self._place_input_bar()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._write_state_file(self._state)

        # Log startup trace
        self.log_event("SYS", "J.A.R.V.I.S. Tactical HUD initialized.")

        # Start background telemetry sampler
        threading.Thread(target=self._telemetry_loop, name="hud-telemetry", daemon=True).start()
        self.animate()

    # ──────────────────────────────────────────────────────────────────────
    # Public API (100% backward-compatible + rich extensions, thread-safe)
    # ──────────────────────────────────────────────────────────────────────

    @property
    def state(self) -> str:
        """Current state as a plain string ('idle', 'listening', ...)."""
        return self._state.value

    def set_state(self, state, detail: str = ""):
        """Request a state change from any thread. Applied on Tk thread within one frame."""
        st = HUDState.coerce(state)
        self._queue.put((st, detail or ""))
        self._write_state_file(st)

    def set_detail(self, detail: str):
        """Update only the detail caption without changing state."""
        self._queue.put((None, detail or ""))

    def flash_error(self, detail: str = "", seconds: float = 2.5, then=HUDState.IDLE):
        """Show ERROR for a moment, then revert. Non-blocking, safe from any thread."""
        self.set_state(HUDState.ERROR, detail)
        threading.Timer(seconds, lambda: self.set_state(then)).start()

    def request_close(self):
        """Ask the Tk thread to close the window (safe from any thread)."""
        self._close_requested = True

    def set_command_callback(self, callback: Callable[[str], None]):
        """Register a callback invoked on a worker thread when user submits text or clicks a quick action."""
        self._command_callback = callback

    def set_wake_callback(self, callback: Callable[[], None]):
        """Register a callback to trigger immediate voice listening."""
        self._wake_callback = callback

    def set_memory_delete_callback(self, callback: Callable[[str], bool]):
        """Register the function that deletes a long-term memory by id. It must
        go through the agent's MemoryManager: editing long_term.json from here
        would be overwritten by the agent's in-memory copy on its next save."""
        self._memory_delete_callback = callback

    def refresh_panels(self):
        """Re-read memories and calendar on the next telemetry tick (~1s)."""
        self._refresh_slow = True

    def set_stop_callback(self, callback: Callable[[], bool]):
        """Register barge-in: called on Esc or an orb click; returns True if
        speech was playing (an orb click on a silent Jarvis keeps its usual
        wake behaviour)."""
        self._stop_callback = callback

    def set_event_callback(self, callback: Callable[[dict], None]):
        """Register a callback for a click on a calendar row (gets the event dict)."""
        self._event_callback = callback

    def log_event(self, kind: str, text: str):
        """Append an entry to the live ReAct / conversation trace feed (thread-safe)."""
        if not text:
            return
        self._event_q.put(("trace", (kind.upper(), str(text).strip())))

    def push_token(self, token: str):
        """Feed live streaming LLM tokens to the HUD ticker (thread-safe)."""
        if token:
            self._event_q.put(("token", token))

    def clear_stream(self):
        """Reset the streaming token buffer at the start of a new generation step."""
        self._event_q.put(("clear_stream", None))

    def notify_tool(self, tool_name: str, args_str: str = "", result_str: str = ""):
        """Highlight a tool node in the neural graph and optionally log its output."""
        self._event_q.put(("tool", (tool_name, args_str, result_str)))

    def set_audio_level(self, level: float):
        """Push a normalized 0..1 microphone or speaker amplitude for the waveform."""
        self._audio_level = max(0.0, min(1.0, float(level)))

    def update_subsystem(self, name: str, value: str, status: str = "OK"):
        """Update one cell in the subsystem status matrix."""
        self._event_q.put(("subsystem", (name, value, status)))

    def toggle_mode(self):
        """Switch between Compact Mini-Orb and Full Tactical Command Center."""
        self._compact = not self._compact
        if self._compact:
            self.root.geometry(f"{self.MINI_W}x{self.MINI_H}")
        else:
            self.root.geometry(f"{self.FULL_W}x{self.FULL_H}")
        self._place_input_bar()

    # ──────────────────────────────────────────────────────────────────────
    # Internals & Background Telemetry
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _write_state_file(state: HUDState):
        try:
            with open(STATE_FILE, "w") as f:
                f.write(_FILE_NAMES.get(state, state.value))
        except Exception:
            pass

    def _place_input_bar(self):
        if self._compact or self.W < 640 or self.H < 420:
            self._input_frame.place_forget()
        else:
            bar_w = max(320, min(560, int(self.W * 0.44)))
            bar_x = (self.W - bar_w) // 2
            bar_y = self.H - 44
            self._input_frame.place(x=bar_x, y=bar_y, width=bar_w, height=32)

    def _on_configure(self, event):
        if event.widget == self.canvas:
            self.W = max(160, event.width)
            self.H = max(180, event.height)
            self._compact = self.W < 520 or self.H < 360
            self._place_input_bar()

    def _focus_canvas(self):
        self.canvas.focus_set()

    def _on_escape(self, _event=None):
        if self._stop_callback:
            threading.Thread(target=self._stop_callback, name="hud-stop", daemon=True).start()
        self._focus_canvas()

    def _on_orb_click(self):
        """Orb click: interrupt speech if Jarvis is talking, otherwise wake it."""
        if self._stop_callback and self._stop_callback():
            return
        if self._wake_callback:
            self._wake_callback()
        else:
            self.log_event("SYS", "Arc Reactor core pinged — systems nominal.")

    def _on_submit_entry(self, _event=None):
        text = self._entry_var.get().strip()
        if not text:
            return
        self._entry_var.set("")
        self._dispatch_user_command(text)

    def _dispatch_user_command(self, text: str):
        self.log_event("USER", text)
        if self._command_callback:
            threading.Thread(
                target=self._command_callback, args=(text,), name="hud-cmd", daemon=True
            ).start()
        else:
            # Demo mode simulation when running standalone (python reactor.py)
            threading.Thread(
                target=self._simulate_command, args=(text,), name="hud-demo-cmd", daemon=True
            ).start()

    def _simulate_command(self, text: str):
        low = text.lower()
        self.set_state(HUDState.THINKING, "step 1: ollama")
        time.sleep(0.6)
        tool = "search_web"
        if "weather" in low:
            tool = "get_weather"
        elif "calendar" in low or "schedule" in low:
            tool = "get_calendar"
        elif "brief" in low or "morning" in low:
            tool = "daily_briefing"
        elif "stat" in low or "cpu" in low or "ram" in low:
            tool = "get_system_stats"
        elif "screenshot" in low:
            tool = "take_screenshot"
        elif "remember" in low:
            tool = "remember"
        elif "volume" in low or "mute" in low:
            tool = "get_volume"

        self.set_state(HUDState.THINKING, f"calling: {tool}")
        self.notify_tool(tool, text)
        time.sleep(0.9)
        reply = f"DONE: Executed {tool} for '{text}', sir. All systems nominal."
        self.clear_stream()
        self.set_state(HUDState.SPEAKING, tool)
        for word in reply[6:].split():
            self.push_token(word + " ")
            time.sleep(0.08)
        self.log_event("DONE", reply[6:])
        time.sleep(0.8)
        self.set_state(HUDState.IDLE)

    def _on_canvas_motion(self, event):
        x, y = event.x, event.y
        hovered = None
        for x1, y1, x2, y2, hid, _payload in self._hitboxes:
            if x1 <= x <= x2 and y1 <= y <= y2:
                hovered = hid
                break
        if hovered != self._hover_id:
            self._hover_id = hovered
            self.canvas.config(cursor="hand2" if hovered else "")

    def _on_canvas_click(self, event):
        x, y = event.x, event.y
        for x1, y1, x2, y2, hid, payload in self._hitboxes:
            if x1 <= x <= x2 and y1 <= y <= y2:
                if hid == "toggle_mode":
                    self.toggle_mode()
                elif hid.startswith("quick:"):
                    self._dispatch_user_command(str(payload))
                elif hid.startswith("state:"):
                    st = payload
                    self.set_state(st, f"manual: {st.value}")
                elif hid.startswith("tool:"):
                    tname = str(payload)
                    self.notify_tool(tname, "manual probe")
                    self.log_event("TOOL", f"Probed node: {tname}")
                elif hid.startswith("memdel:"):
                    self._on_memory_delete_click(str(payload))
                elif hid.startswith("cal:"):
                    self._on_calendar_click(payload)
                elif hid == "orb_core":
                    threading.Thread(target=self._on_orb_click, name="hud-orb", daemon=True).start()
                return

    DELETE_ARM_SECONDS = 4.0

    def _on_memory_delete_click(self, memory_id: str):
        """First click arms the [×], a second click within DELETE_ARM_SECONDS deletes."""
        armed_id, armed_at = self._armed_delete
        if armed_id != memory_id or time.time() - armed_at > self.DELETE_ARM_SECONDS:
            self._armed_delete = (memory_id, time.time())
            return
        self._armed_delete = (None, 0.0)
        if not self._memory_delete_callback:
            self.log_event("ERR", "Memory delete is not wired to the agent.")
            return
        text = next((m.get("text", "") for m in self._memories if m.get("id") == memory_id), memory_id)

        def _run():
            try:
                ok = self._memory_delete_callback(memory_id)
            except Exception as e:
                self.log_event("ERR", f"Memory delete failed: {e}")
                return
            self.log_event("MEM", f"Deleted: {text}" if ok else f"Already gone: {text}")
            self._refresh_slow = True

        threading.Thread(target=_run, name="hud-memdel", daemon=True).start()

    def _on_calendar_click(self, item):
        if not isinstance(item, dict):
            return
        when = item.get("time_range") or item.get("time", "")
        line = f"{item.get('day_long', item.get('day', ''))} {when}: {item.get('title', '')}"
        if item.get("calendar"):
            line += f" [{item['calendar']}]"
        if item.get("clash_with"):
            line += " — CLASHES with " + ", ".join(item["clash_with"])
        self.log_event("CAL", line)
        if self._event_callback:
            threading.Thread(target=self._event_callback, args=(item,), name="hud-cal", daemon=True).start()

    def _drain_queue(self):
        while True:
            try:
                st, detail = self._queue.get_nowait()
            except queue.Empty:
                break
            prev_detail = self.detail
            if st is not None and st != self._state:
                self._state = st
                self._state_since = time.time()
                if st == HUDState.IDLE:
                    self._step_info = "READY"
                elif st == HUDState.LISTENING:
                    self.clear_stream()
                elif st == HUDState.ERROR and detail:
                    self._trace_log.append({
                        "ts": datetime.now().strftime("%H:%M:%S"),
                        "kind": "ERR",
                        "text": detail,
                    })
            if detail != prev_detail and detail:
                if detail.startswith("calling:"):
                    tname = detail.split(":", 1)[1].strip()
                    self._trigger_tool_visual(tname)
                    self._trace_log.append({
                        "ts": datetime.now().strftime("%H:%M:%S"),
                        "kind": "TOOL",
                        "text": f"{tname}()",
                    })
                elif detail.startswith("step "):
                    self._step_info = detail.upper()
                    self._trace_log.append({
                        "ts": datetime.now().strftime("%H:%M:%S"),
                        "kind": "STEP",
                        "text": detail,
                    })
            self.detail = detail

        while True:
            try:
                etype, payload = self._event_q.get_nowait()
            except queue.Empty:
                break
            if etype == "trace":
                kind, text = payload
                if kind == "USER":
                    self._last_user = text
                elif kind in ("DONE", "JARVIS"):
                    self._last_reply = text
                    self._step_info = "READY"
                self._trace_log.append({
                    "ts": datetime.now().strftime("%H:%M:%S"),
                    "kind": kind,
                    "text": text,
                })
            elif etype == "token":
                self._stream_text += payload
                if len(self._stream_text) > 220:
                    self._stream_text = "…" + self._stream_text[-210:]
            elif etype == "clear_stream":
                self._stream_text = ""
            elif etype == "tool":
                tname, args_str, res_str = payload
                self._trigger_tool_visual(tname)
                msg = f"{tname}({args_str})" if args_str else f"{tname}()"
                if res_str:
                    msg += f" → {res_str[:65]}"
                self._trace_log.append({
                    "ts": datetime.now().strftime("%H:%M:%S"),
                    "kind": "TOOL",
                    "text": msg,
                })
            elif etype == "subsystem":
                k, val, st = payload
                self._subsystems[k] = (val, st)

    def _trigger_tool_visual(self, tool_name: str):
        tool_name = (tool_name or "").strip().lower()
        self._active_tool = tool_name
        self._active_tool_ts = time.time()
        self._tool_history[tool_name] = time.time()
        # Spawn animated synaptic packets from core -> hub -> satellite
        for i in range(4):
            self._packets.append({
                "tool": tool_name,
                "progress": -i * 0.18,
                "speed": 0.045 + i * 0.005,
            })

    def _telemetry_loop(self):
        """Background daemon sampling CPU/RAM/Battery, FAISS memory store, and khal calendar."""
        last_slow = 0.0
        base_dir = Path(__file__).parent
        mem_file = base_dir / "memory_data" / "long_term.json"
        short_file = base_dir / "memory_data" / "short_term.json"
        seen_short_ts = ""

        # Read config metadata once if available
        try:
            import config
            self.update_subsystem("LLM BRAIN", config.LLM_MODEL[:14], "ONLINE")
            self.update_subsystem("STT VOICE", config.STT_ENGINE.upper(), "READY")
            self.update_subsystem("TTS VOICE", config.TTS_ENGINE.upper(), "READY")
            ww = config.WAKEWORD_MODEL if config.WAKEWORD_ENABLED else "OFF"
            self.update_subsystem("WAKE WORD", ww[:12].upper(), "ARMED" if config.WAKEWORD_ENABLED else "OFF")
        except Exception:
            pass

        while self.running:
            now = time.time()
            # Fast tick (1.2s): CPU, RAM, Disk, Battery, Short-term conversation sync
            if psutil is not None:
                try:
                    cpu = psutil.cpu_percent(interval=None)
                    vm = psutil.virtual_memory()
                    disk = psutil.disk_usage("/")
                    bat = psutil.sensors_battery()
                    with self._sys_lock:
                        self._cpu_pct = cpu
                        self._ram_pct = vm.percent
                        self._ram_used_gb = vm.used / (1024 ** 3)
                        self._ram_total_gb = vm.total / (1024 ** 3)
                        self._disk_pct = disk.percent
                        self._disk_free_gb = disk.free / (1024 ** 3)
                        self._cpu_hist.append(cpu)
                        self._ram_hist.append(vm.percent)
                        if bat is not None:
                            self._bat_pct = int(bat.percent)
                            self._bat_charging = bool(bat.power_plugged)
                except Exception:
                    pass

            # Sync recent exchange from short_term.json if modified externally
            try:
                if short_file.exists():
                    turns = json.loads(short_file.read_text(encoding="utf-8"))
                    if turns:
                        last = turns[-1]
                        ts = last.get("timestamp", "")
                        if ts and ts != seen_short_ts:
                            seen_short_ts = ts
                            u = last.get("user", "")
                            a = last.get("assistant", "")
                            if u and u != self._last_user:
                                self.log_event("USER", u)
                            if a and a != self._last_reply:
                                self.log_event("DONE", a)
            except Exception:
                pass

            # Slow tick (15s): Long-term FAISS memories + khal calendar events
            if now - last_slow >= 15.0 or self._refresh_slow:
                last_slow = now
                self._refresh_slow = False
                try:
                    if mem_file.exists():
                        raw_m = json.loads(mem_file.read_text(encoding="utf-8"))
                        with self._sys_lock:
                            self._memories = list(reversed(raw_m[-8:]))
                        self.update_subsystem("FAISS MEM", f"{len(raw_m)} FACTS", "SYNCED")
                except Exception:
                    pass

                try:
                    import calendar_tool
                    today = date.today()
                    evts = calendar_tool.fetch_events(today, 2)
                    clashes = calendar_tool.find_conflicts(evts)
                    formatted = []
                    for e in evts[:6]:
                        day_tag = "TODAY" if e.day == today else "TMRW"
                        timed = not (e.all_day or not e.start)
                        t_str = e.start if timed else "ALL DAY"
                        clash_with = [b.title if a == e else a.title
                                      for a, b in clashes if e in (a, b)]
                        formatted.append({
                            "day": day_tag,
                            "day_long": "Today" if e.day == today else "Tomorrow",
                            "date": e.day.isoformat(),
                            "time": t_str,
                            "time_range": (f"{e.start}-{e.end}" if timed and e.end and e.end != e.start
                                           else t_str),
                            "start": e.start, "end": e.end, "all_day": not timed,
                            "title": e.title,
                            "calendar": e.calendar,
                            "clash": bool(clash_with),
                            "clash_with": clash_with,
                        })
                    clash_strs = [f"{a.title[:14]} ⚡ {b.title[:14]}" for a, b in clashes[:2]]
                    with self._sys_lock:
                        self._calendar_items = formatted
                        self._calendar_clashes = clash_strs
                except Exception:
                    pass

            time.sleep(1.2)

    def on_close(self):
        self.running = False
        try:
            with open(STATE_FILE, "w") as f:
                f.write("offline")
        except Exception:
            pass
        self.root.destroy()

    def start(self):
        self.root.mainloop()

    # ──────────────────────────────────────────────────────────────────────
    # Rendering Engine
    # ──────────────────────────────────────────────────────────────────────

    def draw_orb(self):
        """Main render dispatch (kept named draw_orb for backward compatibility)."""
        c = self.canvas
        c.delete("all")
        self._hitboxes.clear()

        if self._compact:
            self._draw_compact_hud(c)
        else:
            self._draw_full_command_center(c)

    # ── 1. COMPACT MINI-ORB MODE ──────────────────────────────────────────

    def _draw_compact_hud(self, c: tk.Canvas):
        w, h = self.W, self.H
        spec = _VISUALS[self._state]
        base = spec["rgb"]
        hex_col = rgb_to_hex(base)

        c.create_rectangle(0, 0, w, h, fill="#03070e", outline="")
        # Subtle grid
        for gx in range(0, w, 24):
            c.create_line(gx, 0, gx, h, fill="#071220")
        for gy in range(0, h, 24):
            c.create_line(0, gy, w, gy, fill="#071220")

        # Top mini bar
        c.create_text(12, 14, text="J.A.R.V.I.S", anchor="w", fill="#00e5ff", font=("JetBrains Mono", 8, "bold"))
        btn_x1, btn_y1, btn_x2, btn_y2 = w - 68, 5, w - 6, 23
        c.create_rectangle(btn_x1, btn_y1, btn_x2, btn_y2, fill="#071a2e", outline="#00b4d8")
        c.create_text((btn_x1 + btn_x2) / 2, 14, text="EXPAND ⤢", fill="#00e5ff", font=("JetBrains Mono", 7, "bold"))
        self._hitboxes.append((btn_x1, btn_y1, btn_x2, btn_y2, "toggle_mode", None))

        cx, cy = w // 2, int(h * 0.44)
        self._draw_reactor_core(c, cx, cy, scale=0.72)

        # State & detail caption
        c.create_text(cx, h - 46, text=spec["label"].split("//")[0].strip(), fill=hex_col, font=("JetBrains Mono", 9, "bold"))
        if self.detail:
            txt = self.detail if len(self.detail) <= 26 else self.detail[:25] + "…"
            c.create_text(cx, h - 28, text=txt, fill="#8fa3b8", font=("JetBrains Mono", 8))
        else:
            c.create_text(cx, h - 28, text=f"CPU {self._cpu_pct:.0f}% · RAM {self._ram_pct:.0f}%", fill="#5c768d", font=("JetBrains Mono", 7))

    # ── 2. FULL TACTICAL COMMAND CENTER ───────────────────────────────────

    def _draw_full_command_center(self, c: tk.Canvas):
        w, h = self.W, self.H
        spec = _VISUALS[self._state]
        base = spec["rgb"]
        accent_hex = rgb_to_hex(base)

        # Background & perspective tactical grid
        c.create_rectangle(0, 0, w, h, fill="#03070e", outline="")
        grid_step = 36
        for gx in range(0, w, grid_step):
            c.create_line(gx, 0, gx, h, fill="#06101e")
        for gy in range(0, h, grid_step):
            c.create_line(0, gy, w, gy, fill="#06101e")

        # Ambient corner radial vignette lines
        c.create_line(0, 44, w, 44, fill="#0d2840", width=1)
        c.create_line(0, h - 54, w, h - 54, fill="#0d2840", width=1)

        # ── TOP HEADER BAR ──
        self._draw_top_bar(c, w, accent_hex, spec)

        # Layout columns
        pad = 14
        top_y = 54
        bot_y = h - 60
        panel_h = max(200, bot_y - top_y)
        left_w = max(270, min(335, int(w * 0.265)))
        right_w = max(290, min(360, int(w * 0.285)))

        left_x1, left_x2 = pad, pad + left_w
        right_x1, right_x2 = w - pad - right_w, w - pad
        center_x1, center_x2 = left_x2 + pad, right_x1 - pad
        center_cx = (center_x1 + center_x2) / 2
        center_cy = top_y + panel_h * 0.45

        # ── CENTER: NEURAL TOOL TOPOLOGY + ARC REACTOR + WAVEFORM ──
        self._draw_center_hub(c, center_x1, top_y, center_x2, bot_y, center_cx, center_cy, spec)

        # ── LEFT COLUMN: TELEMETRY, CALENDAR & CLASHES, MEMORY VAULT ──
        self._draw_left_column(c, left_x1, top_y, left_x2, bot_y, accent_hex)

        # ── RIGHT COLUMN: SUBSYSTEMS, LIVE REACT TRACE, QUICK ACTIONS ──
        self._draw_right_column(c, right_x1, top_y, right_x2, bot_y, accent_hex)

        # ── BOTTOM FOOTER TELEMETRY BAR (around command input) ──
        self._draw_bottom_bar(c, w, h, accent_hex)

    def _draw_top_bar(self, c: tk.Canvas, w: int, accent_hex: str, spec: dict):
        c.create_rectangle(0, 0, w, 44, fill="#050d1a", outline="")
        # Glowing accent strip
        c.create_line(0, 43, w, 43, fill=accent_hex, width=1)

        # Brand identity
        c.create_oval(16, 15, 28, 27, fill=accent_hex, outline="")
        c.create_text(
            36, 21,
            text="J.A.R.V.I.S. // MARK VII TACTICAL AI MATRIX",
            anchor="w",
            fill="#e2f1ff",
            font=("JetBrains Mono", 11, "bold"),
        )

        # Center state pill
        pill_w = 270
        px1, px2 = w / 2 - pill_w / 2, w / 2 + pill_w / 2
        c.create_rectangle(px1, 8, px2, 36, fill=spec["badge_bg"], outline=accent_hex, width=1)
        c.create_text(
            w / 2, 22,
            text=f"● {spec['label']}",
            fill=accent_hex,
            font=("JetBrains Mono", 9, "bold"),
        )

        # Right clock, uptime & mode toggle
        now_str = datetime.now().strftime("%a %d %b  %H:%M:%S").upper()
        uptime_s = int(time.time() - self._boot_time)
        up_m, up_s = divmod(uptime_s, 60)
        up_h, up_m = divmod(up_m, 60)
        c.create_text(
            w - 115, 22,
            text=f"UP {up_h:02d}:{up_m:02d}:{up_s:02d}  |  {now_str}",
            anchor="e",
            fill="#7fa6c4",
            font=("JetBrains Mono", 8),
        )

        # Mini-mode toggle button
        bx1, by1, bx2, by2 = w - 102, 9, w - 14, 35
        is_hov = self._hover_id == "toggle_mode"
        c.create_rectangle(
            bx1, by1, bx2, by2,
            fill="#0c2640" if is_hov else "#071728",
            outline="#00e5ff" if is_hov else "#184366",
        )
        c.create_text((bx1 + bx2) / 2, 22, text="MINI ORB ⤡", fill="#00e5ff", font=("JetBrains Mono", 8, "bold"))
        self._hitboxes.append((bx1, by1, bx2, by2, "toggle_mode", None))

    def _draw_panel_frame(self, c: tk.Canvas, x1: float, y1: float, x2: float, y2: float, title: str, accent: str = "#00b4d8", right_tag: str = ""):
        """Sci-fi chamfered HUD panel with corner brackets and header bar."""
        c.create_rectangle(x1, y1, x2, y2, fill="#050c17", outline="#0e2942", width=1)
        # Header strip
        c.create_rectangle(x1, y1, x2, y1 + 24, fill="#08172b", outline="")
        c.create_line(x1, y1 + 24, x2, y1 + 24, fill="#123556")

        # Corner tactical brackets
        bl = 10
        for cx, cy, dx, dy in [
            (x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)
        ]:
            c.create_line(cx, cy, cx + dx * bl, cy, fill=accent, width=2)
            c.create_line(cx, cy, cx, cy + dy * bl, fill=accent, width=2)

        c.create_text(
            x1 + 10, y1 + 12,
            text=f"▸ {title}",
            anchor="w",
            fill="#c8e6ff",
            font=("JetBrains Mono", 8, "bold"),
        )
        if right_tag:
            c.create_text(
                x2 - 10, y1 + 12,
                text=right_tag,
                anchor="e",
                fill=accent,
                font=("JetBrains Mono", 7, "bold"),
            )

    # ── CENTER HUB: NEURAL NETWORK + ARC REACTOR + AUDIO SPECTRUM ─────────

    def _draw_center_hub(self, c: tk.Canvas, x1: float, y1: float, x2: float, y2: float, cx: float, cy: float, spec: dict):
        w_span = x2 - x1
        h_span = y2 - y1
        scale = max(0.75, min(1.22, min(w_span / 520.0, h_span / 620.0)))

        # Outer tactical radar rings
        max_r = min(w_span, h_span) * 0.45
        for frac, col in [(0.98, "#081c30"), (0.76, "#0b243d"), (0.54, "#0d2c49")]:
            rr = max_r * frac
            c.create_oval(cx - rr, cy - rr, cx + rr, cy + rr, outline=col, width=1)

        # Crosshair axis ticks
        c.create_line(cx - max_r, cy, cx - max_r * 0.38, cy, fill="#0b233b", dash=(3, 5))
        c.create_line(cx + max_r * 0.38, cy, cx + max_r, cy, fill="#0b233b", dash=(3, 5))
        c.create_line(cx, cy - max_r, cx, cy - max_r * 0.38, fill="#0b233b", dash=(3, 5))
        c.create_line(cx, cy + max_r * 0.38, cx, cy + max_r, fill="#0b233b", dash=(3, 5))

        # 1. Draw Neural & Tool Network Graph around the Reactor
        self._draw_tool_network(c, cx, cy, max_r, scale)

        # 2. Draw Central Arc Reactor Core
        self._draw_reactor_core(c, cx, cy, scale=scale)

        # 3. State Banner & Active Tool Callout below Reactor
        banner_y = cy + max_r * 0.72
        accent_hex = rgb_to_hex(spec["rgb"])
        c.create_text(
            cx, banner_y,
            text=spec["label"],
            fill=accent_hex,
            font=("JetBrains Mono", 12, "bold"),
        )
        sub_text = self.detail.upper() if self.detail else spec["sub"]
        c.create_text(
            cx, banner_y + 19,
            text=sub_text[:54],
            fill="#8cb4d4",
            font=("JetBrains Mono", 8, "bold"),
        )

        # 4. Audio Frequency Spectrum Analyzer (36 bars)
        spec_top = banner_y + 38
        self._draw_audio_spectrum(c, cx, spec_top, min(380, w_span - 30), 34, spec["rgb"])

        # 5. Live Streaming Token / Speech Ticker Box
        ticker_y1 = y2 - 54
        ticker_y2 = y2 - 4
        c.create_rectangle(x1 + 6, ticker_y1, x2 - 6, ticker_y2, fill="#050e1c", outline="#123556")
        c.create_text(
            x1 + 16, ticker_y1 + 14,
            text="VOICE / LLM STREAM ▸",
            anchor="w",
            fill="#00e5ff",
            font=("JetBrains Mono", 7, "bold"),
        )
        live_line = self._stream_text.strip() or self._last_reply or "Say 'Hey Jarvis' or type a command below..."
        if len(live_line) > 76:
            live_line = "…" + live_line[-75:]
        c.create_text(
            x1 + 16, ticker_y1 + 33,
            text=live_line,
            anchor="w",
            fill="#d9f0ff",
            font=("JetBrains Mono", 9),
        )

    def _draw_tool_network(self, c: tk.Canvas, cx: float, cy: float, max_r: float, scale: float):
        now = time.time()
        hub_r = max_r * 0.62
        sat_r = max_r * 0.90

        tool_coords: dict[str, tuple[float, float, float, float, str]] = {}

        for cluster_name, col_hex, base_deg, tools in _TOOL_CLUSTERS:
            # Slow orbital drift
            drift = math.sin(self.pulse * 0.015 + base_deg) * 2.5
            ang = math.radians(base_deg + drift)
            hx = cx + hub_r * math.cos(ang)
            hy = cy + hub_r * math.sin(ang)

            # Check if any tool in this cluster was recently active
            cluster_hot = any(
                (now - self._tool_history.get(t, 0.0)) < 6.0 or self._active_tool == t
                for t in tools
            )

            # Spoke from reactor core to cluster hub
            c.create_line(
                cx, cy, hx, hy,
                fill=col_hex if cluster_hot else "#0e2a44",
                width=2 if cluster_hot else 1,
                dash=() if cluster_hot else (2, 4),
            )

            # Fan out individual tool satellites
            n = len(tools)
            spread = min(46.0, max(22.0, n * 6.2))
            for idx, tname in enumerate(tools):
                offset_deg = (idx - (n - 1) / 2.0) * (spread / max(1, n - 1)) if n > 1 else 0.0
                tang = math.radians(base_deg + offset_deg + drift * 0.6)
                # Alternate radius slightly so labels never collide
                r_tweak = sat_r * (0.94 if idx % 2 == 0 else 1.03)
                tx = cx + r_tweak * math.cos(tang)
                ty = cy + r_tweak * math.sin(tang)

                tool_coords[tname] = (hx, hy, tx, ty, col_hex)
                age = now - self._tool_history.get(tname, 0.0)
                is_active = (self._active_tool == tname and age < 4.0)
                is_recent = age < 12.0
                is_hov = self._hover_id == f"tool:{tname}"

                line_col = col_hex if (is_active or is_hov) else ("#1a4b70" if is_recent else "#0a1e32")
                c.create_line(hx, hy, tx, ty, fill=line_col, width=2 if is_active else 1)

                node_r = (5.5 if is_active else (4.2 if is_hov or is_recent else 2.8)) * scale
                fill_col = col_hex if (is_active or is_recent or is_hov) else "#0d2840"
                if is_active:
                    # Outer pulse halo on active tool node
                    halo_r = node_r + 4 + 3 * math.sin(self.pulse * 0.3)
                    c.create_oval(tx - halo_r, ty - halo_r, tx + halo_r, ty + halo_r, outline=col_hex, width=1)

                c.create_oval(tx - node_r, ty - node_r, tx + node_r, ty + node_r, fill=fill_col, outline="")
                self._hitboxes.append((tx - 9, ty - 9, tx + 9, ty + 9, f"tool:{tname}", tname))

                # Show label for active, hovered, or key tools
                if is_active or is_hov or is_recent or idx in (0, n - 1):
                    short_lbl = tname.replace("get_", "").replace("media_", "")
                    anchor = "w" if math.cos(tang) >= 0 else "e"
                    lx = tx + (8 if anchor == "w" else -8)
                    c.create_text(
                        lx, ty,
                        text=short_lbl,
                        anchor=anchor,
                        fill="#ffffff" if (is_active or is_hov) else ("#9be7ff" if is_recent else "#3f6582"),
                        font=("JetBrains Mono", 7, "bold" if is_active else "normal"),
                    )

            # Draw cluster hub node & badge
            hr = (7.0 if cluster_hot else 5.0) * scale
            c.create_oval(hx - hr, hy - hr, hx + hr, hy + hr, fill="#051324", outline=col_hex, width=2)
            c.create_oval(hx - 2.5, hy - 2.5, hx + 2.5, hy + 2.5, fill=col_hex, outline="")
            c.create_text(
                hx, hy - hr - 7,
                text=cluster_name,
                fill=col_hex if cluster_hot else "#5682a3",
                font=("JetBrains Mono", 7, "bold"),
            )

        # Animate synaptic data packets along active tool pathways
        alive_packets = []
        for pkt in self._packets:
            pkt["progress"] += pkt["speed"]
            p = pkt["progress"]
            tname = pkt["tool"]
            if p <= 2.0 and tname in tool_coords:
                alive_packets.append(pkt)
                if p >= 0.0:
                    hx, hy, tx, ty, col_hex = tool_coords[tname]
                    if p <= 1.0:
                        px = cx + (hx - cx) * p
                        py = cy + (hy - cy) * p
                    else:
                        p2 = p - 1.0
                        px = hx + (tx - hx) * p2
                        py = hy + (ty - hy) * p2
                    c.create_oval(px - 3.5, py - 3.5, px + 3.5, py + 3.5, fill="#ffffff", outline=col_hex, width=1)
        self._packets = alive_packets

    def _draw_reactor_core(self, c: tk.Canvas, cx: float, cy: float, scale: float = 1.0):
        spec = _VISUALS[self._state]
        base = spec["rgb"]
        p_val = math.sin(math.radians(self.pulse * spec["speed"]))
        r = (spec["radius"] + spec["swing"] * p_val) * scale

        if self._state == HUDState.ERROR:
            dim = 0.55 + 0.45 * (p_val + 1) / 2
            base = tuple(int(ch * dim) for ch in base)

        hex_col = rgb_to_hex(base)
        dim_hex = rgb_to_hex(blend_rgb(base, (4, 12, 24), 0.62))

        # 1. Outer counter-rotating segmented sci-fi bezel
        outer_r = r * 1.62
        rot1 = (self.pulse * 1.4) % 360
        for k in range(6):
            c.create_arc(
                cx - outer_r, cy - outer_r, cx + outer_r, cy + outer_r,
                start=rot1 + k * 60, extent=34,
                style="arc", outline=dim_hex, width=max(2, int(3 * scale)),
            )

        # 2. Inner high-speed telemetry ring
        mid_r = r * 1.36
        rot2 = (-self.pulse * 2.6) % 360
        for k in range(4):
            c.create_arc(
                cx - mid_r, cy - mid_r, cx + mid_r, cy + mid_r,
                start=rot2 + k * 90, extent=52,
                style="arc", outline=hex_col, width=max(1, int(2 * scale)),
            )

        # 3. Tick marks around reactor housing (24 ticks)
        tick_r1 = r * 1.18
        tick_r2 = r * 1.26
        for t in range(24):
            tang = math.radians(t * 15 + self.pulse * 0.4)
            x1 = cx + tick_r1 * math.cos(tang)
            y1 = cy + tick_r1 * math.sin(tang)
            x2 = cx + tick_r2 * math.cos(tang)
            y2 = cy + tick_r2 * math.sin(tang)
            c.create_line(x1, y1, x2, y2, fill=hex_col if t % 3 == 0 else dim_hex, width=1)

        # 4. State-specific dynamic effects
        if self._state == HUDState.THINKING:
            ang = math.radians(self.pulse * 5.5)
            ring_r = r * 1.12
            for k in range(3):
                a = ang + k * 2 * math.pi / 3
                x = cx + ring_r * math.cos(a)
                y = cy + ring_r * math.sin(a)
                dot_r = 4.5 * scale
                c.create_oval(x - dot_r, y - dot_r, x + dot_r, y + dot_r, fill=hex_col, outline="#ffffff")

        if self._state in (HUDState.LISTENING, HUDState.SPEAKING):
            for wave_idx in range(2):
                ripple = ((self.pulse * 1.8 + wave_idx * 18) % 36) * scale
                rr = r + ripple
                fade = max(0.0, 1.0 - ripple / (36 * scale))
                wave_col = rgb_to_hex(blend_rgb((3, 7, 14), base, fade))
                c.create_oval(cx - rr, cy - rr, cx + rr, cy + rr, outline=wave_col, width=2)

        # 5. 3D Gradient Energy Sphere
        steps = 22
        for i in range(steps):
            ratio = i / steps
            color = rgb_to_hex(tuple(base[j] + (255 - base[j]) * (ratio ** 1.25) for j in range(3)))
            cur_r = r * (1 - ratio * 0.88)
            ox = cx - (r * ratio * 0.25)
            oy = cy - (r * ratio * 0.25)
            c.create_oval(ox - cur_r, oy - cur_r, ox + cur_r, oy + cur_r, fill=color, outline="")

        # 6. Geometric triangular Arc Reactor lattice overlay
        tri_r = r * 0.72
        pts = []
        for k in range(3):
            a = math.radians(-90 + k * 120 + self.pulse * 0.8)
            pts.extend([cx + tri_r * math.cos(a), cy + tri_r * math.sin(a)])
        c.create_polygon(*pts, fill="", outline=hex_col, width=1)

        self._hitboxes.append((cx - r, cy - r, cx + r, cy + r, "orb_core", None))

    def _draw_audio_spectrum(self, c: tk.Canvas, cx: float, top_y: float, width: float, height: float, rgb: tuple):
        n = len(self._audio_bars)
        bar_w = width / n
        start_x = cx - width / 2
        mid_y = top_y + height / 2

        for i in range(n):
            center_dist = abs(i - (n - 1) / 2) / (n / 2)
            envelope = max(0.15, 1.0 - center_dist * 0.75)
            if self._state == HUDState.SPEAKING:
                target = envelope * (0.35 + 0.65 * abs(math.sin(self.pulse * 0.35 + i * 0.45)))
            elif self._state == HUDState.LISTENING:
                target = envelope * (0.22 + 0.55 * abs(math.sin(self.pulse * 0.22 + i * 0.3)))
            elif self._state == HUDState.THINKING:
                target = envelope * (0.15 + 0.25 * abs(math.sin(self.pulse * 0.18 + i * 0.6)))
            else:
                target = 0.08 + 0.05 * math.sin(self.pulse * 0.08 + i * 0.2)

            if self._audio_level > 0.02:
                target = max(target, self._audio_level * envelope)

            self._audio_bars[i] = self._audio_bars[i] * 0.65 + target * 0.35
            bh = max(2.0, self._audio_bars[i] * height)
            bx = start_x + i * bar_w
            col = rgb_to_hex(blend_rgb((15, 45, 75), rgb, min(1.0, self._audio_bars[i] * 1.3)))
            c.create_rectangle(
                bx + 1, mid_y - bh / 2, bx + bar_w - 1.5, mid_y + bh / 2,
                fill=col, outline="",
            )

    # ── LEFT COLUMN: TELEMETRY + CALENDAR + MEMORY VAULT ──────────────────

    def _draw_left_column(self, c: tk.Canvas, x1: float, y1: float, x2: float, y2: float, accent_hex: str):
        total_h = y2 - y1
        gap = 10
        h1 = int(total_h * 0.34)
        h2 = int(total_h * 0.34)

        # Panel 1: Host System Telemetry
        p1_y1, p1_y2 = y1, y1 + h1
        self._draw_panel_frame(c, x1, p1_y1, x2, p1_y2, "SYSTEM TELEMETRY", accent_hex, "LIVE PSUTIL")
        with self._sys_lock:
            cpu = self._cpu_pct
            ram = self._ram_pct
            ram_u, ram_t = self._ram_used_gb, self._ram_total_gb
            disk = self._disk_pct
            disk_f = self._disk_free_gb
            bat = self._bat_pct
            charging = self._bat_charging
            cpu_hist = list(self._cpu_hist)
            ram_hist = list(self._ram_hist)

        inner_x1, inner_x2 = x1 + 12, x2 - 12
        pw = inner_x2 - inner_x1
        cy_cursor = p1_y1 + 36

        # CPU Gauge + Sparkline
        self._draw_metric_row(c, inner_x1, cy_cursor, pw, "CPU CORE LOAD", f"{cpu:4.1f}%", cpu / 100.0, "#00e5ff", cpu_hist)
        cy_cursor += 46

        # RAM Gauge + Sparkline
        ram_lbl = f"{ram:4.1f}% ({ram_u:.1f}/{ram_t:.0f}G)" if ram_t else f"{ram:4.1f}%"
        self._draw_metric_row(c, inner_x1, cy_cursor, pw, "MEMORY ALLOC", ram_lbl, ram / 100.0, "#b967ff", ram_hist)
        cy_cursor += 46

        # Disk & Battery compact bars
        half_w = (pw - 10) / 2
        self._draw_mini_bar(c, inner_x1, cy_cursor, half_w, "ROOT DISK", f"{disk_f:.0f}G FREE", disk / 100.0, "#38bdf8")
        bat_val = (bat / 100.0) if bat is not None else 1.0
        bat_txt = f"{bat}% {'⚡' if charging else 'BAT'}" if bat is not None else "AC POWER"
        bat_col = "#00f5d4" if (bat is None or bat > 30 or charging) else "#ff384c"
        self._draw_mini_bar(c, inner_x1 + half_w + 10, cy_cursor, half_w, "POWER CELL", bat_txt, bat_val, bat_col)

        # Panel 2: khal Calendar & Clash Detection
        p2_y1, p2_y2 = p1_y2 + gap, p1_y2 + gap + h2
        with self._sys_lock:
            cal_items = list(self._calendar_items)
            clashes = list(self._calendar_clashes)

        clash_tag = f"⚠ {len(clashes)} CLASH" if clashes else "KHAL SYNC"
        clash_col = "#ffbe0b" if clashes else "#00e5ff"
        self._draw_panel_frame(c, x1, p2_y1, x2, p2_y2, "SCHEDULE & BRIEFING", clash_col, clash_tag)

        cy_cal = p2_y1 + 34
        if clashes:
            c.create_rectangle(inner_x1, cy_cal, inner_x2, cy_cal + 20, fill="#2b1600", outline="#ffbe0b")
            c.create_text(
                inner_x1 + 6, cy_cal + 10,
                text=f"CLASH: {clashes[0][:34]}",
                anchor="w",
                fill="#ffbe0b",
                font=("JetBrains Mono", 7, "bold"),
            )
            cy_cal += 25

        if not cal_items:
            c.create_text(
                inner_x1, cy_cal + 14,
                text="No upcoming events on khal calendar.",
                anchor="w",
                fill="#567691",
                font=("JetBrains Mono", 8),
            )
        else:
            for item in cal_items[:5]:
                if cy_cal + 20 > p2_y2 - 6:
                    break
                row_col = "#ffbe0b" if item["clash"] else "#00e5ff"
                c.create_text(
                    inner_x1, cy_cal + 8,
                    text=f"{item['day']:5s} {item['time']:7s}",
                    anchor="w",
                    fill=row_col,
                    font=("JetBrains Mono", 7, "bold"),
                )
                title = item["title"]
                if len(title) > 22:
                    title = title[:21] + "…"
                c.create_text(
                    inner_x1 + 92, cy_cal + 8,
                    text=title,
                    anchor="w",
                    fill="#d2e8ff",
                    font=("JetBrains Mono", 8),
                )
                hid = f"cal:{item.get('date')}:{item.get('time')}:{item['title']}"
                if self._hover_id == hid:
                    c.create_rectangle(inner_x1 - 3, cy_cal - 1, inner_x2, cy_cal + 17, outline=row_col)
                self._hitboxes.append((inner_x1 - 3, cy_cal - 1, inner_x2, cy_cal + 17, hid, item))
                cy_cal += 21

        # Panel 3: FAISS Semantic Memory Vault
        p3_y1, p3_y2 = p2_y2 + gap, y2
        with self._sys_lock:
            mems = list(self._memories)
        self._draw_panel_frame(c, x1, p3_y1, x2, p3_y2, "FAISS SEMANTIC MEMORY VAULT", "#b967ff", f"{len(mems)} RECENT")

        cy_mem = p3_y1 + 34
        if not mems:
            c.create_text(
                inner_x1, cy_mem + 14,
                text="No stored long-term facts yet.\nSay 'Remember that...' to store one.",
                anchor="nw",
                fill="#567691",
                font=("JetBrains Mono", 8),
            )
        else:
            for m in mems:
                if cy_mem + 22 > p3_y2 - 6:
                    break
                cat = (m.get("category") or "fact").upper()[:7]
                txt = m.get("text") or ""
                mid = m.get("id") or ""
                armed_id, armed_at = self._armed_delete
                armed = bool(mid) and armed_id == mid and time.time() - armed_at <= self.DELETE_ARM_SECONDS
                if len(txt) > 27:
                    txt = txt[:26] + "…"
                c.create_rectangle(inner_x1, cy_mem, inner_x1 + 48, cy_mem + 15, fill="#190d2c", outline="#7b3fe4")
                c.create_text(inner_x1 + 24, cy_mem + 7.5, text=cat, fill="#d0a2ff", font=("JetBrains Mono", 6, "bold"))
                c.create_text(inner_x1 + 55, cy_mem + 7.5, text=txt, anchor="w", fill="#cbe2f8", font=("JetBrains Mono", 8))
                if mid:
                    # [×] delete: first click arms (turns red, "DEL?"), second click deletes
                    bw = 30 if armed else 16
                    bx1, bx2 = inner_x2 - bw, inner_x2
                    hid = f"memdel:{mid}"
                    hot = armed or self._hover_id == hid
                    c.create_rectangle(bx1, cy_mem, bx2, cy_mem + 15,
                                       fill="#3a0610" if armed else "#140a22",
                                       outline="#ff384c" if hot else "#5a3a8a")
                    c.create_text((bx1 + bx2) / 2, cy_mem + 7.5, text="DEL?" if armed else "×",
                                  fill="#ff384c" if hot else "#a07cd0",
                                  font=("JetBrains Mono", 7 if armed else 9, "bold"))
                    self._hitboxes.append((bx1, cy_mem, bx2, cy_mem + 15, hid, mid))
                cy_mem += 21

    def _draw_metric_row(self, c: tk.Canvas, x: float, y: float, w: float, label: str, val_str: str, ratio: float, color: str, hist: list[float]):
        ratio = max(0.0, min(1.0, ratio))
        c.create_text(x, y, text=label, anchor="w", fill="#7fa6c4", font=("JetBrains Mono", 7, "bold"))
        c.create_text(x + w, y, text=val_str, anchor="e", fill=color, font=("JetBrains Mono", 8, "bold"))

        # Progress bar on left 55%, sparkline on right 42%
        bar_w = w * 0.54
        by1, by2 = y + 11, y + 21
        c.create_rectangle(x, by1, x + bar_w, by2, fill="#08182b", outline="#123252")
        if ratio > 0:
            c.create_rectangle(x + 1, by1 + 1, x + max(3, bar_w * ratio), by2 - 1, fill=color, outline="")

        # Sparkline
        sx1 = x + w * 0.58
        sw = w * 0.42
        sh = 16
        sy_base = y + 23
        c.create_rectangle(sx1, sy_base - sh, sx1 + sw, sy_base, fill="#061221", outline="#0e263f")
        if len(hist) >= 2:
            pts = []
            step = sw / (len(hist) - 1)
            for idx, v in enumerate(hist):
                px = sx1 + idx * step
                py = sy_base - 1 - (max(0.0, min(100.0, v)) / 100.0) * (sh - 2)
                pts.extend([px, py])
            c.create_line(*pts, fill=color, width=1.5)

    def _draw_mini_bar(self, c: tk.Canvas, x: float, y: float, w: float, label: str, val_str: str, ratio: float, color: str):
        ratio = max(0.0, min(1.0, ratio))
        c.create_text(x, y, text=label, anchor="w", fill="#6d92b0", font=("JetBrains Mono", 7))
        c.create_text(x + w, y, text=val_str, anchor="e", fill=color, font=("JetBrains Mono", 7, "bold"))
        c.create_rectangle(x, y + 10, x + w, y + 18, fill="#08182b", outline="#123252")
        c.create_rectangle(x + 1, y + 11, x + max(2, w * ratio), y + 17, fill=color, outline="")

    # ── RIGHT COLUMN: SUBSYSTEMS + REACT TRACE + QUICK DECK ───────────────

    def _draw_right_column(self, c: tk.Canvas, x1: float, y1: float, x2: float, y2: float, accent_hex: str):
        total_h = y2 - y1
        gap = 10
        h1 = 128
        h3 = 112
        h2 = total_h - h1 - h3 - gap * 2

        # Panel 1: Subsystem Status Matrix (2x3 Grid)
        p1_y1, p1_y2 = y1, y1 + h1
        self._draw_panel_frame(c, x1, p1_y1, x2, p1_y2, "NEURAL & AUDIO SUBSYSTEMS", accent_hex, "LOCAL ON-DEVICE")

        inner_x1, inner_x2 = x1 + 10, x2 - 10
        cell_w = (inner_x2 - inner_x1 - 8) / 2
        cell_h = 26
        items = list(self._subsystems.items())
        for idx, (k, (val, st)) in enumerate(items[:6]):
            row, col = divmod(idx, 2)
            cx1 = inner_x1 + col * (cell_w + 8)
            cy1 = p1_y1 + 32 + row * (cell_h + 5)
            ok = st in ("ONLINE", "OK", "READY", "ARMED", "SYNCED", "CLOSED")
            dot_col = "#00f5d4" if ok else "#ffbe0b"
            c.create_rectangle(cx1, cy1, cx1 + cell_w, cy1 + cell_h, fill="#071526", outline="#11314f")
            c.create_oval(cx1 + 6, cy1 + 10, cx1 + 12, cy1 + 16, fill=dot_col, outline="")
            c.create_text(cx1 + 17, cy1 + 8, text=k, anchor="w", fill="#6c92b2", font=("JetBrains Mono", 6, "bold"))
            c.create_text(cx1 + 17, cy1 + 19, text=val, anchor="w", fill="#d8eeff", font=("JetBrains Mono", 7, "bold"))

        # Panel 2: Live ReAct Loop & Conversation Trace
        p2_y1, p2_y2 = p1_y2 + gap, p1_y2 + gap + h2
        self._draw_panel_frame(c, x1, p2_y1, x2, p2_y2, "REACT REASONING & TRACE LOG", "#00e5ff", self._step_info)

        trace_y = p2_y1 + 34
        max_rows = max(4, int((p2_y2 - trace_y - 8) / 24))
        visible_trace = list(self._trace_log)[-max_rows:]

        badge_colors = {
            "USER": ("#062c43", "#38bdf8"),
            "STEP": ("#2e1c00", "#ff9e00"),
            "TOOL": ("#032b26", "#00f5d4"),
            "DONE": ("#210d38", "#b967ff"),
            "ERR":  ("#3b0810", "#ff384c"),
            "SYS":  ("#0b2038", "#7fa6c4"),
        }

        for entry in visible_trace:
            kind = entry["kind"]
            bg_c, fg_c = badge_colors.get(kind, ("#0b2038", "#00e5ff"))
            c.create_text(inner_x1, trace_y + 7, text=entry["ts"], anchor="w", fill="#476782", font=("JetBrains Mono", 7))
            c.create_rectangle(inner_x1 + 52, trace_y, inner_x1 + 92, trace_y + 15, fill=bg_c, outline=fg_c)
            c.create_text(inner_x1 + 72, trace_y + 7.5, text=kind[:4], fill=fg_c, font=("JetBrains Mono", 6, "bold"))

            msg = entry["text"].replace("\n", " ")
            if len(msg) > 29:
                msg = msg[:28] + "…"
            c.create_text(inner_x1 + 98, trace_y + 7.5, text=msg, anchor="w", fill="#d6ecff", font=("JetBrains Mono", 8))
            trace_y += 23

        # Panel 3: Tactical Quick-Command Deck (Clickable)
        p3_y1, p3_y2 = p2_y2 + gap, y2
        self._draw_panel_frame(c, x1, p3_y1, x2, p3_y2, "TACTICAL QUICK DECK", "#00f5d4", "CLICK TO RUN")

        quick_cmds = [
            ("☀ BRIEF ME", "good morning"),
            ("📅 CALENDAR", "what is on my calendar today"),
            ("🌦 WEATHER", "what is the weather"),
            ("📊 SYS STATS", "how much CPU and RAM am I using"),
            ("📸 SCREENSHOT", "take a screenshot"),
            ("🔇 MUTE AUDIO", "mute toggle"),
        ]
        btn_w = (inner_x2 - inner_x1 - 12) / 3
        btn_h = 28
        for idx, (label, cmd_text) in enumerate(quick_cmds):
            r_idx, c_idx = divmod(idx, 3)
            bx1 = inner_x1 + c_idx * (btn_w + 6)
            by1 = p3_y1 + 32 + r_idx * (btn_h + 7)
            bx2, by2 = bx1 + btn_w, by1 + btn_h
            hid = f"quick:{idx}"
            hov = self._hover_id == hid
            c.create_rectangle(
                bx1, by1, bx2, by2,
                fill="#0d2f4f" if hov else "#081a2e",
                outline="#00f5d4" if hov else "#163e63",
            )
            c.create_text(
                (bx1 + bx2) / 2, (by1 + by2) / 2,
                text=label,
                fill="#00f5d4" if hov else "#b8dcff",
                font=("JetBrains Mono", 7, "bold"),
            )
            self._hitboxes.append((bx1, by1, bx2, by2, hid, cmd_text))

    # ── BOTTOM FOOTER BAR ─────────────────────────────────────────────────

    def _draw_bottom_bar(self, c: tk.Canvas, w: int, h: int, accent_hex: str):
        y_top = h - 54
        c.create_rectangle(0, y_top, w, h, fill="#040a14", outline="")
        c.create_line(0, y_top, w, y_top, fill="#102c47", width=1)

        # Left: State selector pills (clickable for testing/manual override)
        sx = 14
        sy = h - 28
        c.create_text(sx, sy, text="STATE:", anchor="w", fill="#587b99", font=("JetBrains Mono", 7, "bold"))
        sx += 44
        for st in (HUDState.IDLE, HUDState.LISTENING, HUDState.THINKING, HUDState.SPEAKING, HUDState.ERROR):
            col = rgb_to_hex(_VISUALS[st]["rgb"])
            active = self._state == st
            pw = 54
            bx1, by1, bx2, by2 = sx, sy - 10, sx + pw, sy + 10
            c.create_rectangle(
                bx1, by1, bx2, by2,
                fill=_VISUALS[st]["badge_bg"] if active else "#071322",
                outline=col if active else "#14314d",
            )
            c.create_text(
                (bx1 + bx2) / 2, sy,
                text=st.value.upper()[:6],
                fill=col if active else "#6688a6",
                font=("JetBrains Mono", 6, "bold"),
            )
            self._hitboxes.append((bx1, by1, bx2, by2, f"state:{st.value}", st))
            sx += pw + 5

        # Right: Keyboard shortcut & Waybar sync indicator
        c.create_text(
            w - 16, sy,
            text="F11: MINI/FULL  ·  WAYBAR: /tmp/jarvis_state  ·  29 TOOLS ARMED",
            anchor="e",
            fill="#587b99",
            font=("JetBrains Mono", 7),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Animation Loop
    # ──────────────────────────────────────────────────────────────────────

    def animate(self):
        if self._close_requested and self.running:
            self.on_close()
            return
        if not self.running:
            return
        self._drain_queue()
        self.pulse += 1
        self.draw_orb()
        self.root.after(33, self.animate)


if __name__ == "__main__":
    # Interactive Demo: cycles states & tool calls, or click any button / type commands!
    app = ArcReactor()

    def demo_cycle():
        seq = [
            (HUDState.LISTENING, "", None),
            (HUDState.THINKING, "step 1: ollama", None),
            (HUDState.THINKING, "calling: daily_briefing", "daily_briefing"),
            (HUDState.SPEAKING, "Good morning, sir. It is 29 degrees in Chennai with clear skies.", None),
            (HUDState.THINKING, "calling: get_calendar", "get_calendar"),
            (HUDState.SPEAKING, "You have two events on your khal calendar today.", None),
            (HUDState.IDLE, "", None),
        ]
        time.sleep(1.0)
        while app.running:
            for st, detail, tool in seq:
                if not app.running:
                    return
                app.set_state(st, detail)
                if tool:
                    app.notify_tool(tool, "auto-demo")
                if st == HUDState.SPEAKING and detail:
                    app.clear_stream()
                    for word in detail.split():
                        app.push_token(word + " ")
                        time.sleep(0.12)
                    app.log_event("DONE", detail)
                time.sleep(2.4)

    threading.Thread(target=demo_cycle, daemon=True).start()
    app.start()
