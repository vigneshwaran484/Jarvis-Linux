import os
from pathlib import Path
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
SCREENSHOT_DIR = Path(os.getenv("SCREENSHOT_DIR", "~/Pictures/jarvis")).expanduser()
VIDEO_DIR      = Path(os.getenv("VIDEO_DIR", "~/Videos/jarvis")).expanduser()

# ── LLM ────────────────────────────────────────────────────────────────────
LLM_MODEL    = os.getenv("LLM_MODEL", "qwen3:4b-instruct")
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://localhost:11434/api/chat")
LLM_MAX_STEPS = int(os.getenv("LLM_MAX_STEPS", "6"))
# Reasoning models (qwen3, deepseek-r1, ...) "think" before replying. For a voice
# assistant that is dead air, so it is off by default. Set LLM_THINK=true to
# allow it, or LLM_THINK=auto to send nothing and use the model's own default.
_think = os.getenv("LLM_THINK", "false").strip().lower()
LLM_THINK = None if _think in ("auto", "") else _think in ("1", "true", "yes")
# Optional personality line added to the system prompt. Empty string disables it.
PERSONA = os.getenv(
    "JARVIS_PERSONA",
    "Speak like a calm, dry-witted British butler and address the user as 'sir'. "
    "Wit is welcome but never at the cost of the facts.",
).strip()

# City used for weather when none is spoken (also by the daily briefing).
# Empty = guess from the IP address, which is often the ISP's city, not yours.
HOME_CITY = os.getenv("HOME_CITY", "").strip()

# ── Voice ───────────────────────────────────────────────────────────────────
TTS_VOICE        = os.getenv("TTS_VOICE", "en-AU-WilliamNeural")
STT_LANGUAGE     = os.getenv("STT_LANGUAGE", "en-IN")        # Indian English — better accuracy for you
PHRASE_TIME_LIMIT = int(os.getenv("PHRASE_TIME_LIMIT", "10"))   # max seconds per spoken command

# Speech engines. "whisper" and "piper" run on this machine (CPU); "google" and
# "edge" are the older online engines and remain the automatic fallback when a
# local model is missing or fails.
STT_ENGINE       = os.getenv("STT_ENGINE", "whisper").strip().lower()   # whisper | google
TTS_ENGINE       = os.getenv("TTS_ENGINE", "piper").strip().lower()     # kokoro | piper | edge
WHISPER_MODEL    = os.getenv("WHISPER_MODEL", "base.en")   # tiny.en | base.en | small.en | small (multilingual)
# Whisper's confidence is exp(mean token log-probability): clear speech lands
# around 0.6-0.8, so it needs a lower floor than Google's 0.5.
WHISPER_MIN_CONFIDENCE = float(os.getenv("WHISPER_MIN_CONFIDENCE", "0.35"))
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")      # empty = auto-detect (multilingual models only)
PIPER_VOICE      = os.getenv("PIPER_VOICE", "en_GB-alan-medium")
# Kokoro: more natural than Piper, a little slower. Voices: bm_george, bm_fable,
# bm_lewis, bm_daniel (British male), am_michael, am_adam (US male), bf_emma,
# af_heart (female)... run `python voice_demo.py` to hear them.
KOKORO_VOICE     = os.getenv("KOKORO_VOICE", "bm_george")
KOKORO_SPEED     = float(os.getenv("KOKORO_SPEED", "1.0"))
KOKORO_MODEL     = os.getenv("KOKORO_MODEL", "kokoro-v1.0.onnx")        # file under VOICE_DIR
VOICE_DIR        = Path(os.getenv("VOICE_DIR", Path(__file__).parent / "voices")).expanduser()

# ── Wake word (openWakeWord) ────────────────────────────────────────────────
WAKEWORD_ENABLED      = os.getenv("WAKEWORD_ENABLED", "true").lower() in ("1", "true", "yes")
WAKEWORD_MODEL        = os.getenv("WAKEWORD_MODEL", "hey_jarvis")  # see github.com/dscripka/openWakeWord for available models
WAKEWORD_THRESHOLD    = float(os.getenv("WAKEWORD_THRESHOLD", "0.5"))  # detection confidence (0-1, higher = stricter)

# Scores between this and WAKEWORD_THRESHOLD are "maybe": the last couple of
# seconds are transcribed locally with Whisper and Jarvis wakes only if the
# text really contains "Jarvis". Helps accents and mics the wake model scores
# low, without lowering the main threshold. Set equal to WAKEWORD_THRESHOLD to disable.
WAKEWORD_CONFIRM_THRESHOLD = float(os.getenv("WAKEWORD_CONFIRM_THRESHOLD", "0.15"))

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
# Circuit breaker: after N consecutive Ollama failures, answer "I'm offline"
# instantly for RECOVERY seconds instead of waiting LLM_TIMEOUT every time.
LLM_CIRCUIT_FAILURE_THRESHOLD = int(os.getenv("LLM_CIRCUIT_FAILURE_THRESHOLD", "2"))
LLM_CIRCUIT_RECOVERY_SECONDS  = float(os.getenv("LLM_CIRCUIT_RECOVERY_SECONDS", "30"))

# ── Web search ──────────────────────────────────────────────────────────────
# Optional. With a Tavily key search_web returns real web results; without one
# it falls back to DuckDuckGo Instant Answers + Wikipedia (no key needed).
TAVILY_API_KEY     = os.getenv("TAVILY_API_KEY", "")
WEB_SEARCH_TIMEOUT = float(os.getenv("WEB_SEARCH_TIMEOUT", "8"))

# ── Email (optional, read-only) ─────────────────────────────────────────────
# Leave MAIL_ADDRESS empty to disable. For Gmail use an app password
# (myaccount.google.com/apppasswords), never the account password.
MAIL_ADDRESS      = os.getenv("MAIL_ADDRESS", "").strip()
MAIL_APP_PASSWORD = os.getenv("MAIL_APP_PASSWORD", "").strip()
MAIL_IMAP_HOST    = os.getenv("MAIL_IMAP_HOST", "imap.gmail.com").strip()
MAIL_MAX_SPOKEN   = int(os.getenv("MAIL_MAX_SPOKEN", "5"))   # newest N unread read aloud

# ── Memory ──────────────────────────────────────────────────────────────────
MEMORY_DIR            = Path(os.getenv("MEMORY_DIR", "memory_data"))
SHORT_TERM_WINDOW     = int(os.getenv("SHORT_TERM_WINDOW", "10"))   # last N exchanges kept in prompt
SHORT_TERM_PERSIST    = os.getenv("SHORT_TERM_PERSIST", "true").lower() in ("1", "true", "yes")
LONG_TERM_TOP_K       = int(os.getenv("LONG_TERM_TOP_K", "3"))      # memories retrieved per query
LONG_TERM_MIN_SCORE   = float(os.getenv("LONG_TERM_MIN_SCORE", "0.25"))  # cosine floor for retrieval
EMBEDDING_MODEL       = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
