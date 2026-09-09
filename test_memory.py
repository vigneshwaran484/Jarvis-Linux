"""
test_memory.py — verifies memory persistence + retrieval without touching the mic/LLM.

Run:  python test_memory.py
Uses an isolated data dir (memory_data_test/) and removes it afterwards.
"""
import os, shutil, sys
from pathlib import Path

TEST_DIR = Path("memory_data_test")
os.environ["MEMORY_DIR"] = str(TEST_DIR)
os.environ["LOG_FILE"] = str(TEST_DIR / "test.log")
shutil.rmtree(TEST_DIR, ignore_errors=True)
TEST_DIR.mkdir()

from memory import MemoryManager, extract_explicit_memory  # noqa: E402  (after env setup)

ok = True
def check(cond, msg):
    global ok
    print(("✅ " if cond else "❌ ") + msg)
    ok &= bool(cond)

# 1. explicit trigger parsing
check(extract_explicit_memory("remember that I have a meeting at 5") == "I have a meeting at 5", "trigger: 'remember that'")
check(extract_explicit_memory("Jarvis, don't forget my project X uses Postgres") == "my project X uses Postgres", "trigger: 'don't forget'")
check(extract_explicit_memory("open spotify") is None, "trigger: ignores normal commands")

# 2. store in session A
a = MemoryManager(warm=False)
a.add_memory("I have a meeting with the ZoomRx team at 5 pm")
a.add_memory("My project X uses Postgres as its database")
a.add_memory("My favourite editor is Neovim")
a.record_exchange("open spotify", "Spotify is now open.")
print("   embedder backend:", a.embedder.backend)
check(len(a.list_memories()) == 3, "3 long-term memories stored")
del a

# 3. reload in session B (simulates restarting Jarvis)
b = MemoryManager(warm=False)
check(len(b.list_memories()) == 3, "long-term memories survive restart")
check(b.short.recent() and b.short.recent()[-1]["user"] == "open spotify", "short-term window survives restart")

hits = b.search_memory("what database does project X use?", k=1)
check(hits and "Postgres" in hits[0][0].text, f"semantic retrieval → {hits[0][0].text if hits else None}")
hits = b.search_memory("when is my meeting", k=1)
check(hits and "meeting" in hits[0][0].text, f"semantic retrieval → {hits[0][0].text if hits else None}")

ctx = b.get_context("what editor do I like")
check("KNOWN FACTS" in ctx and "RECENT CONVERSATION" in ctx, "get_context() injects both stores")

# 4. delete + prune
mid = b.list_memories()[0].id
check(b.delete_memory(mid) and len(b.list_memories()) == 2, "delete_memory() removes entry + rebuilds index")
check(b.prune(older_than_days=0) == 2 and not b.list_memories(), "prune() drops old entries")

shutil.rmtree(TEST_DIR, ignore_errors=True)
print("\nALL PASSED" if ok else "\nFAILURES", file=sys.stderr)
sys.exit(0 if ok else 1)
