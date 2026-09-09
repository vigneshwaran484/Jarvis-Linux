"""
jarvis_log.py — central logging for J.A.R.V.I.S.

Every module does `from jarvis_log import log` and uses log.info / log.warning /
log.error / log.exception.  Output goes to:
  * jarvis.log (LOG_FILE in config) — timestamped, persists across sessions,
    useful for reliability analysis (demo video / IEEE write-up).
  * stderr — same lines, so the terminal view stays readable.
"""
import logging
import sys
from logging.handlers import RotatingFileHandler

from config import LOG_FILE, LOG_LEVEL

log = logging.getLogger("jarvis")

if not log.handlers:  # guard against double-init on re-import
    log.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(threadName)-10s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    log.addHandler(file_handler)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.WARNING)  # keep the terminal quiet; everything is in the file
    console.setFormatter(fmt)
    log.addHandler(console)

    log.propagate = False
