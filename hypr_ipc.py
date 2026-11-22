"""
hypr_ipc.py — talk to Hyprland directly over its UNIX socket.

Replaces the ydotool keypress approach (which can only press keys and cannot
see what is open) with real queries and dispatches:

    list_windows()                 -> speech-friendly list of open windows
    focus_window(name)             -> jump to a window by app name or title
    move_window_to_workspace(n)    -> move the focused window to workspace n
    focus_workspace(n)             -> switch to workspace n
    toggle_special(name)           -> show / hide a special (scratchpad) workspace

Socket: $XDG_RUNTIME_DIR/hypr/$HYPRLAND_INSTANCE_SIGNATURE/.socket.sock
Queries use the "j/<name>" form and return JSON.

Hyprland 0.57 with a Lua config: `dispatch <x>` is evaluated as Lua
`hl.dispatch(<x>)`, so the old `dispatch workspace 2` syntax fails. Every
dispatch here is built as an hl.dsp.* expression, and every value placed in it
is either an integer we validated or a hex address Hyprland itself gave us,
so nothing the user (or the LLM) says ends up inside the Lua expression.

Window-targeting dispatches always name the window by address with
follow = false. Relying on "the active window" is unsafe: focus can change
between the query and the dispatch (it did during testing and moved the
wrong window).
"""
from __future__ import annotations

import json
import os
import re
import socket

SOCKET_TIMEOUT = 2.0   # seconds; Hyprland answers in well under 10 ms
MAX_WORKSPACE = 10

# Spoken names -> substrings of a window's class. Checked before fuzzy title
# matching so "browser" finds zen even when its title is a page name.
APP_ALIASES = {
    "browser": ["zen", "firefox", "chromium", "chrome", "brave"],
    "zen": ["zen"],
    "firefox": ["firefox"],
    "terminal": ["kitty", "alacritty", "foot", "wezterm"],
    "kitty": ["kitty"],
    "vs code": ["code"],
    "vscode": ["code"],
    "code": ["code"],
    "editor": ["code"],
    "spotify": ["spotify"],
    "music": ["spotify"],
    "files": ["yazi", "nautilus", "thunar", "dolphin"],
    "yazi": ["yazi"],
    "emulator": ["yuzu"],
    "yuzu": ["yuzu"],
    "game": ["yuzu", "sober", "steam_app"],
    "roblox": ["sober"],
    "discord": ["discord", "vesktop"],
}

# Friendlier spoken names for common window classes.
_CLASS_NAMES = {
    "zen": "Zen browser",
    "kitty": "Kitty terminal",
    "org.yuzu_emu.yuzu": "yuzu",
    "org.vinegarhq.sober": "Roblox",
    "spotify": "Spotify",
    "code": "VS Code",
}


class HyprError(RuntimeError):
    pass


# ── socket ──────────────────────────────────────────────────────────────────

def socket_path() -> str:
    sig = os.environ.get("HYPRLAND_INSTANCE_SIGNATURE")
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if not sig or not runtime:
        raise HyprError("not running inside a Hyprland session")
    path = os.path.join(runtime, "hypr", sig, ".socket.sock")
    if not os.path.exists(path):
        raise HyprError("Hyprland socket not found")
    return path


def request(command: str) -> str:
    """Send one raw command and return the whole reply."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(SOCKET_TIMEOUT)
            s.connect(socket_path())
            s.sendall(command.encode())
            chunks = []
            while True:
                data = s.recv(65536)
                if not data:
                    break
                chunks.append(data)
    except (OSError, socket.timeout) as e:
        raise HyprError(f"socket error: {e}") from e
    return b"".join(chunks).decode(errors="replace")


def query(name: str):
    """JSON query, e.g. query('clients'), query('activewindow')."""
    raw = request(f"j/{name}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise HyprError(f"bad reply to {name}: {raw[:120]}") from e


def dispatch(lua_expr: str) -> None:
    """Run one hl.dsp.* dispatcher. Hyprland replies 'ok' on success and an
    'error:' / 'warning:' line otherwise (a missing window is only a warning,
    so anything that is not 'ok' counts as failure)."""
    reply = request(f"dispatch {lua_expr}").strip()
    if reply != "ok":
        first = reply.splitlines()[0] if reply else "no reply"
        raise HyprError(first[:160])


# ── helpers ─────────────────────────────────────────────────────────────────

_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]+$")


def _address(win: dict) -> str:
    addr = str(win.get("address", ""))
    if not _ADDRESS_RE.match(addr):          # never put anything else in Lua
        raise HyprError(f"unexpected window address {addr!r}")
    return addr


_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
}


def _workspace_number(value) -> int:
    text = str(value or "").lower()
    m = re.search(r"\d+", text)
    if not m:
        word = next((w for w in re.findall(r"[a-z]+", text) if w in _NUMBER_WORDS), None)
        if word:
            return _NUMBER_WORDS[word]
        raise HyprError("say a workspace number from 1 to 10")
    n = int(m.group())
    if not 1 <= n <= MAX_WORKSPACE:
        raise HyprError(f"workspace {n} is out of range, use 1 to {MAX_WORKSPACE}")
    return n


def _app_name(win: dict) -> str:
    cls = (win.get("class") or "").strip()
    return _CLASS_NAMES.get(cls.lower(), cls or "an untitled window")


def _short_title(win: dict, limit: int = 45) -> str:
    title = (win.get("title") or "").strip()
    # "page — Zen Browser" style suffixes repeat the app name
    title = re.split(r"\s+[—–-]\s+(?:Zen Browser|Mozilla Firefox|Visual Studio Code)$", title)[0]
    return title if len(title) <= limit else title[:limit].rsplit(" ", 1)[0] + "…"


def _workspace_label(win: dict) -> str:
    ws = win.get("workspace") or {}
    name = str(ws.get("name", "?"))
    if name.startswith("special:"):
        return f"the {name.split(':', 1)[1]} scratchpad"
    return f"workspace {name}"


def _normal_windows():
    return [w for w in query("clients") if w.get("mapped", True) and w.get("workspace")]


def _ws_sort_key(win: dict):
    ws = win.get("workspace") or {}
    wid = ws.get("id")
    return (0, wid) if isinstance(wid, int) and wid > 0 else (1, str(ws.get("name")))


def find_window(name: str, windows=None):
    """Best window for a spoken name: alias -> class match -> title match.
    Returns (window, None) or (None, reason)."""
    name = re.sub(r"\s+", " ", (name or "").strip().lower())
    name = re.sub(r"^(the|my)\s+", "", name)
    name = re.sub(r"\s+(window|app)$", "", name)
    if not name:
        return None, "which window? say an app name like browser or terminal"
    windows = windows if windows is not None else _normal_windows()

    wanted = APP_ALIASES.get(name, [name])
    by_class = [w for w in windows
                if any(a in (w.get("class") or "").lower() for a in wanted)]
    if by_class:
        # most recently focused first (focusHistoryID 0 = current)
        by_class.sort(key=lambda w: w.get("focusHistoryID", 99))
        return by_class[0], None

    by_title = [w for w in windows if name in (w.get("title") or "").lower()]
    if by_title:
        by_title.sort(key=lambda w: w.get("focusHistoryID", 99))
        return by_title[0], None

    words = [t for t in re.findall(r"[a-z0-9]+", name) if len(t) > 2]
    scored = []
    for w in windows:
        hay = f"{w.get('class', '')} {w.get('title', '')}".lower()
        hits = sum(1 for t in words if t in hay)
        if hits:
            scored.append((hits, -w.get("focusHistoryID", 99), w))
    if scored:
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        return scored[0][2], None
    return None, f"no open window matches {name}"


# ── tools (return speech-friendly strings, never raise) ─────────────────────

def list_windows(_unused=None) -> str:
    try:
        windows = sorted(_normal_windows(), key=_ws_sort_key)
        active = query("activewindow") or {}
    except HyprError as e:
        return f"Could not read windows: {e}"
    if not windows:
        return "No windows are open."
    parts = [f"{len(windows)} windows open."]
    for w in windows:
        bit = f"{_workspace_label(w).capitalize()}: {_app_name(w)}"
        title = _short_title(w)
        if title and title.lower() != _app_name(w).lower():
            bit += f", {title}"
        if w.get("address") == active.get("address"):
            bit += " (focused)"
        parts.append(bit + ".")
    return " ".join(parts)


def focus_window(name: str = "") -> str:
    try:
        win, reason = find_window(name)
        if not win:
            return f"Could not focus: {reason}."
        dispatch(f'hl.dsp.focus({{ window = "address:{_address(win)}" }})')
    except HyprError as e:
        return f"Could not focus that window: {e}"
    return f"Focused {_app_name(win)} on {_workspace_label(win)}."


def move_window_to_workspace(number: str = "") -> str:
    try:
        n = _workspace_number(number)
        win = query("activewindow") or {}
        if not win.get("address"):
            return "There is no focused window to move."
        if "J.A.R.V.I.S" in (win.get("title") or ""):
            return "The focused window is my own HUD; focus the window you want moved first."
        dispatch(
            f'hl.dsp.window.move({{ window = "address:{_address(win)}", '
            f'workspace = "{n}", follow = false }})'
        )
    except HyprError as e:
        return f"Could not move the window: {e}"
    return f"Moved {_app_name(win)} to workspace {n}."


def focus_workspace(number: str = "") -> str:
    try:
        n = _workspace_number(number)
        dispatch(f'hl.dsp.focus({{ workspace = "{n}" }})')
    except HyprError as e:
        return f"Could not switch workspace: {e}"
    return f"Switched to workspace {n}."


def toggle_special(name: str = "special") -> str:
    name = (name or "special").strip().lower()
    if not re.fullmatch(r"[a-z0-9_-]{1,32}", name):
        return "Could not toggle that scratchpad: bad name."
    try:
        dispatch(f'hl.dsp.workspace.toggle_special("{name}")')
    except HyprError as e:
        return f"Could not toggle the scratchpad: {e}"
    return f"Toggled the {name} scratchpad."


if __name__ == "__main__":
    # Read-only smoke test: python hypr_ipc.py
    print(list_windows())
    for probe in ("browser", "terminal", "spotify", "nonexistent thing"):
        w, why = find_window(probe)
        print(f"{probe!r:22} -> {(_app_name(w) + ' on ' + _workspace_label(w)) if w else why}")
