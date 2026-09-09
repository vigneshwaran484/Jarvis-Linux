"""
memory.py — persistent memory layer for J.A.R.V.I.S.

Two stores, one interface (MemoryManager):

  Short-term  — rolling window of the last N user/assistant exchanges.
                Lives in a deque; optionally persisted to memory_data/short_term.json.
  Long-term   — explicit facts the user asked Jarvis to remember.
                FAISS IndexFlatIP over L2-normalised MiniLM embeddings (cosine),
                with a parallel, human-readable JSON metadata store
                (memory_data/long_term.json) so entries can be read and pruned.

Public interface (decoupled from the ReAct loop):
    add_memory(text, category=None)        -> LongTermEntry
    get_context(query)                      -> str   (prompt block: recent + relevant)
    search_memory(query, k=None)            -> list[(LongTermEntry, score)]
    record_exchange(user_text, reply_text)  -> None  (short-term)
    extract_explicit_memory(user_text)      -> str | None  ("remember that ..." trigger)
    list_memories(), delete_memory(id), prune(older_than_days), clear_short_term()

Heavy deps (faiss-cpu, sentence-transformers) are imported lazily and are
optional: without them the store degrades to a hashed bag-of-words embedding
and a NumPy brute-force search, so the assistant still boots.
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from config import (
    MEMORY_DIR, SHORT_TERM_WINDOW, SHORT_TERM_PERSIST,
    LONG_TERM_TOP_K, LONG_TERM_MIN_SCORE, EMBEDDING_MODEL,
)
from jarvis_log import log

EMBED_DIM = 384  # all-MiniLM-L6-v2 output size; fallback embedder matches it


# ─────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────

class Embedder:
    """Lazy sentence-transformers wrapper pinned to CPU (keeps VRAM for Ollama)."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()
        self.backend = "unloaded"

    def warmup(self):
        """Load the model in a background thread so startup isn't blocked."""
        threading.Thread(target=self._load, name="embed-warm", daemon=True).start()

    def _load(self):
        with self._lock:
            if self._model is not None:
                return
            try:
                from sentence_transformers import SentenceTransformer
                t0 = time.time()
                self._model = SentenceTransformer(self.model_name, device="cpu")
                self.backend = "sentence-transformers"
                log.info("Embedding model '%s' loaded on CPU in %.1fs", self.model_name, time.time() - t0)
            except Exception as e:  # ImportError, download failure, ...
                self._model = False
                self.backend = "hashed-fallback"
                log.warning("sentence-transformers unavailable (%s); using hashed fallback embeddings", e)

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            self._load()
        if self._model:
            vecs = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return np.asarray(vecs, dtype="float32")
        return self._hashed(texts)

    @staticmethod
    def _hashed(texts: list[str]) -> np.ndarray:
        """Deterministic bag-of-words hashing — crude, but works with zero deps."""
        out = np.zeros((len(texts), EMBED_DIM), dtype="float32")
        for row, text in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                out[row, hash(tok) % EMBED_DIM] += 1.0
            norm = np.linalg.norm(out[row])
            if norm:
                out[row] /= norm
        return out


# ─────────────────────────────────────────
# Long-term store
# ─────────────────────────────────────────

@dataclass
class LongTermEntry:
    id: str
    text: str
    timestamp: str      # ISO-8601
    category: str

    @property
    def created(self) -> datetime:
        return datetime.fromisoformat(self.timestamp)


_CATEGORY_RULES = [
    ("event",    r"\b(meeting|appointment|call|deadline|at \d{1,2}(:\d{2})?\s*(am|pm)?|tomorrow|tonight|on (mon|tues|wednes|thurs|fri|satur|sun)day)\b"),
    ("project",  r"\b(project|repo|codebase|uses|stack|database|postgres|mysql|api|backend|frontend)\b"),
    ("personal", r"\b(my name|i am|i'm|i like|i love|i hate|i prefer|favourite|favorite|birthday|wife|husband|mom|dad)\b"),
    ("todo",     r"\b(need to|have to|should|todo|to do|buy|pay|submit|send)\b"),
]

def guess_category(text: str) -> str:
    low = text.lower()
    for cat, pattern in _CATEGORY_RULES:
        if re.search(pattern, low):
            return cat
    return "fact"


class LongTermMemory:
    def __init__(self, directory: Path, embedder: Embedder):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.dir / "long_term.json"
        self.index_path = self.dir / "long_term.faiss"
        self.embedder = embedder
        self._lock = threading.RLock()
        self.entries: list[LongTermEntry] = []
        self._vectors = np.zeros((0, EMBED_DIM), dtype="float32")  # always kept; mirrors index
        self._index = None
        self._faiss = None
        try:
            import faiss
            self._faiss = faiss
        except Exception as e:
            log.warning("faiss unavailable (%s); using NumPy brute-force search", e)
        self._load()

    # ── persistence ──
    def _load(self):
        if self.meta_path.exists():
            try:
                raw = json.loads(self.meta_path.read_text(encoding="utf-8"))
                self.entries = [LongTermEntry(**e) for e in raw]
            except Exception as e:
                log.error("Corrupt long_term.json (%s) — starting empty", e)
                self.entries = []
        vec_path = self.dir / "long_term_vectors.npy"
        if self.entries and vec_path.exists():
            try:
                self._vectors = np.load(vec_path).astype("float32")
                if self._vectors.shape[0] != len(self.entries):
                    raise ValueError("vector/metadata count mismatch")
            except Exception as e:
                log.warning("Rebuilding embeddings for %d memories (%s)", len(self.entries), e)
                self._vectors = self.embedder.encode([m.text for m in self.entries])
        elif self.entries:
            self._vectors = self.embedder.encode([m.text for m in self.entries])
        self._rebuild_index()
        log.info("Long-term memory loaded: %d entries", len(self.entries))

    def _save(self):
        self.meta_path.write_text(
            json.dumps([asdict(e) for e in self.entries], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        np.save(self.dir / "long_term_vectors.npy", self._vectors)
        if self._faiss and self._index is not None:
            self._faiss.write_index(self._index, str(self.index_path))

    def _rebuild_index(self):
        if not self._faiss:
            self._index = None
            return
        self._index = self._faiss.IndexFlatIP(EMBED_DIM)  # inner product == cosine on unit vectors
        if len(self._vectors):
            self._index.add(self._vectors)

    # ── API ──
    def add(self, text: str, category: Optional[str] = None) -> LongTermEntry:
        text = " ".join(text.split())
        with self._lock:
            # de-dupe near-identical facts
            for existing in self.entries:
                if existing.text.lower() == text.lower():
                    log.info("Memory already stored, skipping duplicate: %r", text)
                    return existing
            entry = LongTermEntry(
                id=uuid.uuid4().hex[:8],
                text=text,
                timestamp=datetime.now().isoformat(timespec="seconds"),
                category=category or guess_category(text),
            )
            vec = self.embedder.encode([text])
            self.entries.append(entry)
            self._vectors = np.vstack([self._vectors, vec])
            if self._index is not None:
                self._index.add(vec)
            self._save()
        log.info("Long-term memory added [%s/%s]: %r", entry.id, entry.category, text)
        return entry

    def search(self, query: str, k: int = LONG_TERM_TOP_K, min_score: float = LONG_TERM_MIN_SCORE
               ) -> list[tuple[LongTermEntry, float]]:
        with self._lock:
            if not self.entries:
                return []
            q = self.embedder.encode([query])
            k = min(k, len(self.entries))
            if self._index is not None:
                scores, idxs = self._index.search(q, k)
                pairs = zip(scores[0], idxs[0])
            else:
                sims = self._vectors @ q[0]
                top = np.argsort(-sims)[:k]
                pairs = ((sims[i], i) for i in top)
            return [
                (self.entries[i], float(s))
                for s, i in pairs
                if i >= 0 and s >= min_score
            ]

    def delete(self, memory_id: str) -> bool:
        with self._lock:
            keep = [i for i, e in enumerate(self.entries) if e.id != memory_id]
            if len(keep) == len(self.entries):
                return False
            self.entries = [self.entries[i] for i in keep]
            self._vectors = self._vectors[keep] if keep else np.zeros((0, EMBED_DIM), dtype="float32")
            self._rebuild_index()
            self._save()
        log.info("Long-term memory deleted: %s", memory_id)
        return True

    def prune(self, older_than_days: int, categories: Optional[set[str]] = None) -> int:
        """Drop entries older than N days (optionally only certain categories, e.g. {'event'})."""
        cutoff = time.time() - older_than_days * 86400
        with self._lock:
            victims = [
                e.id for e in self.entries
                if e.created.timestamp() < cutoff and (categories is None or e.category in categories)
            ]
        for vid in victims:
            self.delete(vid)
        return len(victims)


# ─────────────────────────────────────────
# Short-term store
# ─────────────────────────────────────────

class ShortTermMemory:
    def __init__(self, directory: Path, window: int = SHORT_TERM_WINDOW, persist: bool = SHORT_TERM_PERSIST):
        self.path = Path(directory) / "short_term.json"
        self.window = window
        self.persist = persist
        self._lock = threading.Lock()
        self.turns: deque[dict] = deque(maxlen=window)
        if persist and self.path.exists():
            try:
                for t in json.loads(self.path.read_text(encoding="utf-8"))[-window:]:
                    self.turns.append(t)
                log.info("Short-term memory restored: %d exchanges", len(self.turns))
            except Exception as e:
                log.error("Corrupt short_term.json (%s) — starting empty", e)

    def add(self, user_text: str, reply_text: str):
        with self._lock:
            self.turns.append({
                "user": user_text,
                "assistant": reply_text,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            })
            if self.persist:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps(list(self.turns), indent=2, ensure_ascii=False), encoding="utf-8")

    def recent(self) -> list[dict]:
        with self._lock:
            return list(self.turns)

    def clear(self):
        with self._lock:
            self.turns.clear()
            if self.persist and self.path.exists():
                self.path.unlink()


# ─────────────────────────────────────────
# Explicit "remember that ..." trigger
# ─────────────────────────────────────────

_REMEMBER_RE = re.compile(
    r"^\s*(?:(?:hey|ok|okay)?\s*jarvis[,!]?\s*)?(?:please\s+)?"
    r"(?:remember(?:\s+that)?|don'?t\s+forget(?:\s+that)?|do\s+not\s+forget(?:\s+that)?|"
    r"note(?:\s+down)?(?:\s+that)?|keep\s+in\s+mind(?:\s+that)?)\s*[:,-]?\s+(?P<fact>.+?)[.!]?\s*$",
    re.IGNORECASE,
)

def extract_explicit_memory(user_text: str) -> Optional[str]:
    """Return the fact to store if the utterance is an explicit remember command, else None."""
    m = _REMEMBER_RE.match(user_text or "")
    if not m:
        return None
    fact = m.group("fact").strip()
    # normalise first-person phrasing so retrieval reads naturally later
    return fact if len(fact) > 2 else None


_FORGET_RE = re.compile(
    r"^\s*(?:(?:hey|ok|okay)?\s*jarvis[,!]?\s*)?(?:please\s+)?"
    r"(?:forget|delete|remove)\s+(?:the\s+memory\s+(?:about|that|of)\s+|(?:that|about|the fact that)\s+)?(?P<what>.+?)[.!]?\s*$",
    re.IGNORECASE,
)

def extract_forget_command(user_text: str) -> Optional[str]:
    """Return the search phrase if the utterance is an explicit forget command, else None."""
    m = _FORGET_RE.match(user_text or "")
    if not m:
        return None
    what = m.group("what").strip()
    return what if len(what) > 2 else None


# ─────────────────────────────────────────
# Facade
# ─────────────────────────────────────────

class MemoryManager:
    def __init__(self, directory: Path = MEMORY_DIR, warm: bool = True):
        self.embedder = Embedder()
        self.short = ShortTermMemory(directory)
        self.long = LongTermMemory(directory, self.embedder)
        if warm:
            self.embedder.warmup()

    # explicit facts
    def add_memory(self, text: str, category: Optional[str] = None) -> LongTermEntry:
        return self.long.add(text, category)

    def search_memory(self, query: str, k: Optional[int] = None) -> list[tuple[LongTermEntry, float]]:
        return self.long.search(query, k or LONG_TERM_TOP_K)

    def list_memories(self) -> list[LongTermEntry]:
        return list(self.long.entries)

    def delete_memory(self, memory_id: str) -> bool:
        return self.long.delete(memory_id)

    def prune(self, older_than_days: int, categories: Optional[set[str]] = None) -> int:
        return self.long.prune(older_than_days, categories)

    # conversation
    def record_exchange(self, user_text: str, reply_text: str):
        self.short.add(user_text, reply_text)

    def clear_short_term(self):
        self.short.clear()

    extract_explicit_memory = staticmethod(extract_explicit_memory)
    extract_forget_command  = staticmethod(extract_forget_command)

    def forget(self, query: str) -> Optional[LongTermEntry]:
        """Delete the single best-matching long-term memory; returns it or None."""
        hits = self.long.search(query, k=1, min_score=0.3)
        if not hits:
            return None
        entry, _ = hits[0]
        self.long.delete(entry.id)
        return entry

    # prompt injection
    def get_context(self, query: str, k: Optional[int] = None) -> str:
        """Prompt block combining recent exchanges and relevant long-term facts. Empty string if none."""
        parts = []
        recent = self.short.recent()
        if recent:
            lines = [f"User: {t['user']}\nJarvis: {t['assistant']}" for t in recent]
            parts.append("RECENT CONVERSATION (oldest first):\n" + "\n".join(lines))
        hits = self.search_memory(query, k)
        if hits:
            lines = [f"- ({e.category}, {e.timestamp[:10]}) {e.text}" for e, _ in hits]
            parts.append("KNOWN FACTS THE USER ASKED YOU TO REMEMBER:\n" + "\n".join(lines))
            log.info("Retrieved %d long-term memories for %r: %s", len(hits), query,
                     [(e.id, round(s, 2)) for e, s in hits])
        return "\n\n".join(parts)


if __name__ == "__main__":
    # Quick smoke test:  python memory.py
    mm = MemoryManager(warm=False)
    mm.add_memory("I have a meeting with the ZoomRx team at 5 pm")
    mm.add_memory("My project Jarvis uses Ollama and FAISS")
    print(mm.embedder.backend)
    for e, s in mm.search_memory("when is my meeting?"):
        print(f"{s:.2f}  [{e.category}] {e.text}")
    mm.record_exchange("open spotify", "Spotify is now open.")
    print("\n" + mm.get_context("what database does my project use"))
