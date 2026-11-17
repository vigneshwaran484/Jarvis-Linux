"""
screen_tool.py — let Jarvis read what is on the screen (OCR, fully local).

    inspect_screen(question) -> text for the LLM: the question plus the OCR'd
                                screen text it should answer from

Pipeline: grim captures the focused window (or the whole screen) -> ImageMagick
turns it grayscale, inverts dark themes (tesseract reads dark-on-light far
better) and upscales 2x -> tesseract -> cleaned text.

Why OCR and not a vision model: on a 6 GB GPU a vision model would compete
with the main LLM for VRAM and force Ollama to swap models on every question.
OCR runs on the CPU, takes about a second, and is accurate on the main use
case: error messages and code in a terminal or editor.

Nothing leaves the machine: the screenshot is a temp file deleted right after
OCR, and the text only goes to the local Ollama model.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import hypr_ipc

OCR_TIMEOUT = 20          # seconds, tesseract on a 2x-upscaled 1080p frame takes ~1-3s
MAX_CHARS = 2500          # keep the prompt small for a 4B model
_WHOLE_SCREEN_RE = re.compile(r"\b(whole|entire|full|all of (?:the|my))\s+(screen|display|monitor)\b", re.I)


class ScreenError(RuntimeError):
    pass


def _run(cmd, timeout=OCR_TIMEOUT, **kw):
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)
    except FileNotFoundError:
        raise ScreenError(f"{cmd[0]} is not installed")
    except subprocess.TimeoutExpired:
        raise ScreenError(f"{cmd[0]} timed out")
    if proc.returncode != 0:
        raise ScreenError(f"{cmd[0]} failed: {(proc.stderr or proc.stdout).strip()[:160]}")
    return proc.stdout


def target_region(whole_screen=False):
    """(grim geometry or None for the whole screen, spoken label)."""
    if whole_screen:
        return None, "the screen"
    try:
        win = hypr_ipc.query("activewindow") or {}
    except hypr_ipc.HyprError:
        return None, "the screen"
    title = win.get("title") or ""
    # Typing into the HUD focuses the HUD itself; reading it back is useless.
    if not win.get("address") or "J.A.R.V.I.S" in title:
        return None, "the screen"
    (x, y), (w, h) = win.get("at", (0, 0)), win.get("size", (0, 0))
    if w < 20 or h < 20:
        return None, "the screen"
    return f"{x},{y} {w}x{h}", hypr_ipc._app_name(win)


def capture(path, geometry=None):
    cmd = ["grim"]
    if geometry:
        cmd += ["-g", geometry]
    _run(cmd + [str(path)])


def preprocess(src, dst):
    """Grayscale, invert if the image is mostly dark, upscale 2x.
    Returns the path to OCR (src itself if ImageMagick is missing)."""
    magick = shutil.which("magick")
    if not magick:
        return src
    try:
        mean = float(_run([magick, str(src), "-colorspace", "Gray", "-format", "%[fx:mean]", "info:"]))
    except (ScreenError, ValueError):
        mean = 1.0
    cmd = [magick, str(src), "-colorspace", "Gray"]
    if mean < 0.5:
        cmd.append("-negate")
    cmd += ["-filter", "Lanczos", "-resize", "200%", str(dst)]
    try:
        _run(cmd)
        return dst
    except ScreenError:
        return src


def ocr(path):
    return _run(["tesseract", str(path), "stdout", "-l", "eng", "--psm", "6"])


def clean(text):
    """Drop empty and junk lines (borders, icon glyphs), collapse spaces."""
    lines = []
    for line in (text or "").splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if len(re.findall(r"[A-Za-z0-9]", line)) >= 2:
            lines.append(line)
    out = "\n".join(lines)
    if len(out) > MAX_CHARS:
        # errors usually sit at the bottom of a terminal: keep the end
        out = "…\n" + out[-MAX_CHARS:]
    return out


def read_screen(whole_screen=False):
    """(cleaned text, label). Raises ScreenError."""
    geometry, label = target_region(whole_screen)
    with tempfile.TemporaryDirectory(prefix="jarvis_ocr_") as tmp:
        raw, prepped = Path(tmp) / "shot.png", Path(tmp) / "prep.png"
        capture(raw, geometry)
        text = clean(ocr(preprocess(raw, prepped)))
    return text, label


def inspect_screen(question=""):
    """Tool: read the focused window (or whole screen) and hand the text to the LLM."""
    question = (question or "").strip()
    if question.lower() in {"question", "query", "argument"}:
        question = ""
    try:
        text, label = read_screen(whole_screen=bool(_WHOLE_SCREEN_RE.search(question)))
    except ScreenError as e:
        return f"Could not read the screen: {e}"
    if not text:
        return f"I captured {label} but found no readable text on it."
    q = question or "what is on the screen"
    return (f"Text read from {label} by OCR (may contain small misreads). "
            f"Answer the user's question '{q}' using only this text:\n{text}")


if __name__ == "__main__":
    import sys, time
    t = time.time()
    out = inspect_screen(" ".join(sys.argv[1:]))
    print(out)
    print(f"\n[{time.time() - t:.2f}s]")
