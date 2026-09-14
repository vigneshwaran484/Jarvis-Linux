import os
from pathlib import Path
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
SCREENSHOT_DIR = Path(os.getenv("SCREENSHOT_DIR", "~/Pictures/jarvis")).expanduser()
VIDEO_DIR      = Path(os.getenv("VIDEO_DIR", "~/Videos/jarvis")).expanduser()

# ── LLM ────────────────────────────────────────────────────────────────────
LLM_MODEL    = os.getenv("LLM_MODEL", "llama3.2")
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://localhost:11434/api/chat")
LLM_MAX_STEPS = int(os.getenv("LLM_MAX_STEPS", "6"))

# ── Voice ───────────────────────────────────────────────────────────────────
TTS_VOICE        = os.getenv("TTS_VOICE", "en-AU-WilliamNeural")
STT_LANGUAGE     = os.getenv("STT_LANGUAGE", "en-IN")        # Indian English — better accuracy for you
PHRASE_TIME_LIMIT = int(os.getenv("PHRASE_TIME_LIMIT", "10"))   # max seconds per spoken command

# ── Wake word (openWakeWord) ────────────────────────────────────────────────
WAKEWORD_ENABLED      = os.getenv("WAKEWORD_ENABLED", "true").lower() in ("1", "true", "yes")
WAKEWORD_MODEL        = os.getenv("WAKEWORD_MODEL", "hey_jarvis")  # see github.com/dscripka/openWakeWord for available models
WAKEWORD_THRESHOLD    = float(os.getenv("WAKEWORD_THRESHOLD", "0.5"))  # detection confidence (0-1, higher = stricter)

# ── Legacy wake words (Google STT fallback) ───────────────────────────────────
WAKE_WORDS = ["hey jarvis", "jarvis", "jar vis", "hey jervis", "jervis", "hi jarvis", "hey harvest", "travis", "service"]


# ── Allowed apps (security allowlist) ──────────────────────────────────────
# Maps a spoken app name to the Linux process name close_app() should match.
ALLOWED_APPS = {
    "notepad":        "kate",
    "calculator":     "kcalc",
    "explorer":       "thunar",
    "terminal":       "kitty",
    "spotify":        "spotify",
    "chrome":         "chrome",
    "brave":          "brave",
    "firefox":        "firefox",
    "vscode":         "code",
    "android studio": "studio.sh",
}

# ── Allowed file extensions for write_file ──────────────────────────────────
ALLOWED_WRITE_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".log"}

# ── Reliability ─────────────────────────────────────────────────────────────
LLM_TIMEOUT        = float(os.getenv("LLM_TIMEOUT", "30"))       # seconds before "having trouble thinking"
STT_MIN_CONFIDENCE = float(os.getenv("STT_MIN_CONFIDENCE", "0.5")) # Google STT confidence floor (0-1)
LOG_FILE           = Path(os.getenv("LOG_FILE", "jarvis.log"))
LOG_LEVEL          = os.getenv("LOG_LEVEL", "INFO")

# ── Memory ──────────────────────────────────────────────────────────────────
MEMORY_DIR            = Path(os.getenv("MEMORY_DIR", "memory_data"))
SHORT_TERM_WINDOW     = int(os.getenv("SHORT_TERM_WINDOW", "10"))   # last N exchanges kept in prompt
SHORT_TERM_PERSIST    = os.getenv("SHORT_TERM_PERSIST", "true").lower() in ("1", "true", "yes")
LONG_TERM_TOP_K       = int(os.getenv("LONG_TERM_TOP_K", "3"))      # memories retrieved per query
LONG_TERM_MIN_SCORE   = float(os.getenv("LONG_TERM_MIN_SCORE", "0.25"))  # cosine floor for retrieval
EMBEDDING_MODEL       = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
