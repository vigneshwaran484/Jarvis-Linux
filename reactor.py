"""
reactor.py — ArcReactor HUD for J.A.R.V.I.S.

State machine (HUDState) driven by the voice pipeline.  The voice loop runs on
a worker thread; Tkinter must stay on the main thread, so set_state() only
enqueues a request and the animation tick applies it on the Tk thread.

Every state change is also mirrored to /tmp/jarvis_state so jarvis-waybar.sh
keeps working unchanged (file protocol: one word, e.g. "listening").
"""
import math
import queue
import threading
import time
import tkinter as tk
from enum import Enum

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


# Visual spec per state: base colour, pulse speed multiplier, base radius, radius swing
_VISUALS = {
    HUDState.IDLE:      dict(rgb=(30, 80, 120),  speed=2,  radius=40, swing=2,  label="STANDBY"),
    HUDState.LISTENING: dict(rgb=(0, 200, 255),  speed=4,  radius=45, swing=4,  label="LISTENING"),
    HUDState.THINKING:  dict(rgb=(255, 140, 0),  speed=6,  radius=45, swing=6,  label="THINKING"),
    HUDState.SPEAKING:  dict(rgb=(170, 70, 255), speed=10, radius=50, swing=10, label="SPEAKING"),
    HUDState.ERROR:     dict(rgb=(230, 40, 40),  speed=14, radius=46, swing=8,  label="ERROR"),
}

# Names written to the waybar state file (keeps the old "sleeping" contract)
_FILE_NAMES = {HUDState.IDLE: "sleeping"}


def rgb_to_hex(rgb):
    return "#%02x%02x%02x" % (int(rgb[0]), int(rgb[1]), int(rgb[2]))


class ArcReactor:
    W, H = 160, 180

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("J.A.R.V.I.S")
        self.root.geometry(f"{self.W}x{self.H}")
        self.root.configure(bg="#010812")
        self.root.resizable(False, False)

        self.canvas = tk.Canvas(self.root, width=self.W, height=self.H, bg="#010812", highlightthickness=0)
        self.canvas.pack()

        self.cx, self.cy = self.W // 2, 75
        self.pulse = 0
        self._state = HUDState.IDLE
        self.detail = ""              # e.g. "calling: get_weather"
        self._state_since = time.time()
        self._queue = queue.Queue()   # (HUDState, detail) requests from any thread
        self._close_requested = False

        self.running = True
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._write_state_file(self._state)
        self.animate()

    # ── public API (thread-safe) ──
    @property
    def state(self) -> str:
        """Current state as a plain string ('idle', 'listening', ...)."""
        return self._state.value

    def set_state(self, state, detail: str = ""):
        """Request a state change from any thread. Applied on the Tk thread within one frame."""
        st = HUDState.coerce(state)
        self._queue.put((st, detail or ""))
        # Write the waybar file immediately so the bar doesn't lag behind the orb
        self._write_state_file(st)

    def set_detail(self, detail: str):
        """Update only the small caption (e.g. which tool is running) without changing state."""
        self._queue.put((None, detail or ""))

    def flash_error(self, detail: str = "", seconds: float = 2.5, then=HUDState.IDLE):
        """Show ERROR for a moment, then revert. Non-blocking, safe from any thread."""
        self.set_state(HUDState.ERROR, detail)
        threading.Timer(seconds, lambda: self.set_state(then)).start()

    def request_close(self):
        """Ask the Tk thread to close the window (safe from any thread)."""
        self._close_requested = True

    # ── internals ──
    @staticmethod
    def _write_state_file(state: HUDState):
        try:
            with open(STATE_FILE, "w") as f:
                f.write(_FILE_NAMES.get(state, state.value))
        except Exception:
            pass

    def _drain_queue(self):
        while True:
            try:
                st, detail = self._queue.get_nowait()
            except queue.Empty:
                return
            if st is not None and st != self._state:
                self._state = st
                self._state_since = time.time()
                self.pulse = 0
            self.detail = detail

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

    def draw_orb(self):
        c = self.canvas
        c.delete("all")
        c.create_rectangle(0, 0, self.W, self.H, fill="#010812", outline="")

        spec = _VISUALS[self._state]
        base = spec["rgb"]
        p_val = math.sin(math.radians(self.pulse * spec["speed"]))
        r = spec["radius"] + spec["swing"] * p_val

        if self._state == HUDState.ERROR:
            # hard red strobe: dim the sphere on the down-beat
            dim = 0.55 + 0.45 * (p_val + 1) / 2
            base = tuple(int(ch * dim) for ch in base)

        if self._state == HUDState.THINKING:
            # rotating orbit ring so "thinking" reads as activity even when the LLM is silent
            ang = math.radians(self.pulse * 5)
            ring_r = r + 9
            for k in range(3):
                a = ang + k * 2 * math.pi / 3
                x, y = self.cx + ring_r * math.cos(a), self.cy + ring_r * math.sin(a)
                c.create_oval(x - 3, y - 3, x + 3, y + 3, fill=rgb_to_hex(base), outline="")

        if self._state == HUDState.LISTENING:
            # expanding sonar ring
            ripple = (self.pulse * 1.5) % 30
            rr = r + ripple
            fade = int(120 * (1 - ripple / 30))
            c.create_oval(self.cx - rr, self.cy - rr, self.cx + rr, self.cy + rr,
                          outline=rgb_to_hex((0, fade, fade + 60)), width=2)

        # 3D gradient sphere (original visual)
        steps = 25
        for i in range(steps):
            ratio = i / steps
            color = rgb_to_hex(tuple(base[j] + (255 - base[j]) * ratio for j in range(3)))
            cur_r = r * (1 - ratio)
            ox = self.cx - (r * ratio * 0.4)
            oy = self.cy - (r * ratio * 0.4)
            c.create_oval(ox - cur_r, oy - cur_r, ox + cur_r, oy + cur_r, fill=color, outline="")

        # State label
        c.create_text(self.cx, 145, text=spec["label"], fill=rgb_to_hex(spec["rgb"]),
                      font=("Courier New", 10, "bold"))
        # Detail caption (tool name / error reason), truncated to fit
        if self.detail:
            text = self.detail if len(self.detail) <= 24 else self.detail[:23] + "…"
            c.create_text(self.cx, 163, text=text, fill="#8fa3b8", font=("Courier New", 8))

    def animate(self):
        if self._close_requested and self.running:
            self.on_close()
            return
        if not self.running:
            return
        self._drain_queue()
        self.pulse += 1
        self.draw_orb()
        self.root.after(30, self.animate)


if __name__ == "__main__":
    # Demo: cycle through every state so you can eyeball the visuals.
    app = ArcReactor()

    def cycle():
        seq = [
            (HUDState.LISTENING, ""),
            (HUDState.THINKING, "calling: get_weather"),
            (HUDState.SPEAKING, ""),
            (HUDState.ERROR, "ollama timeout"),
            (HUDState.IDLE, ""),
        ]
        while app.running:
            for st, detail in seq:
                app.set_state(st, detail)
                time.sleep(2.5)

    threading.Thread(target=cycle, daemon=True).start()
    app.start()
