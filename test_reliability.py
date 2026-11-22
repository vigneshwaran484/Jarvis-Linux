"""
test_reliability.py — circuit breaker, ask_llm fail-fast, web search fallbacks,
and sleep/shutdown phrase matching. No mic, no Ollama, no network needed.

Run:  .venv/bin/python test_reliability.py
"""
import os, sys, tempfile

os.environ["LOG_FILE"] = os.path.join(tempfile.mkdtemp(), "test.log")

import requests  # noqa: E402
from circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState  # noqa: E402
import web_search  # noqa: E402
import agent_v2  # noqa: E402

ok = True
def check(cond, msg):
    global ok
    print(("✅ " if cond else "❌ ") + msg)
    ok &= bool(cond)


class FakeClock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t

def boom():
    raise ValueError("down")

def raises(exc_type, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc_type:
        return True
    except Exception:
        return False
    return False


# 1. breaker state machine
clock = FakeClock()
b = CircuitBreaker("t", failure_threshold=2, recovery_timeout=10, clock=clock)
check(b.call(lambda: 42) == 42, "breaker: passes calls through when closed")
raises(ValueError, b.call, boom)
check(b.state == CircuitState.CLOSED, "breaker: one failure keeps it closed")
raises(ValueError, b.call, boom)
check(b.state == CircuitState.OPEN, "breaker: opens after threshold failures")
calls = []
check(raises(CircuitOpenError, b.call, lambda: calls.append(1)) and not calls,
      "breaker: open circuit rejects without calling the backend")
clock.t = 10
check(b.state == CircuitState.HALF_OPEN, "breaker: half-open after cooldown")
raises(ValueError, b.call, boom)
check(b.state == CircuitState.OPEN, "breaker: failed probe re-opens")
clock.t = 20
check(b.call(lambda: "up") == "up" and b.state == CircuitState.CLOSED,
      "breaker: successful probe closes")
raises(ValueError, b.call, boom, is_failure=lambda e: False)
raises(ValueError, b.call, boom, is_failure=lambda e: False)
check(b.state == CircuitState.CLOSED, "breaker: is_failure=False errors don't count")


# 2. ask_llm fails fast once Ollama is down
real_post = requests.post
post_calls = []
def dead_post(*a, **kw):
    post_calls.append(1)
    raise requests.exceptions.ConnectionError("refused")
requests.post = dead_post
agent_v2._llm_breaker._failures = 0
agent_v2._llm_breaker._state = CircuitState.CLOSED
reasons = []
for _ in range(4):
    try:
        agent_v2.ask_llm([{"role": "user", "content": "hi"}])
    except agent_v2.LLMError as e:
        reasons.append(str(e))
requests.post = real_post
check(reasons[:2] == ["unreachable", "unreachable"], "ask_llm: first failures report 'unreachable'")
check(reasons[2:] == ["circuit_open", "circuit_open"], "ask_llm: later calls fail fast with 'circuit_open'")
check(len(post_calls) == 2, "ask_llm: Ollama not contacted while circuit is open")


# 3. web search backend fallback order
saved = web_search.BACKENDS
web_search.BACKENDS = [
    ("broken", lambda q: (_ for _ in ()).throw(RuntimeError("network"))),
    ("empty", lambda q: ""),
    ("good", lambda q: "Canberra is the capital " * 60),
]
snippet, src = web_search.lookup("capital of australia")
check(src == "good", "web: skips failing and empty backends")
check(len(snippet) <= web_search.MAX_CHARS + 1, "web: result clipped for small-model context")
web_search.BACKENDS = [("empty", lambda q: "")]
check(web_search.lookup("x") == ("", None), "web: no answer returns empty")
check(web_search.lookup("  ") == ("", None), "web: blank query returns empty")

opened = []
agent_v2.webbrowser.open = opened.append
result = agent_v2.search_web("obscure thing")
check(opened and "No direct answer" in result, "search_web: falls back to opening the browser")
web_search.BACKENDS = saved


# 4. sleep / shutdown phrases (substring matching used to misfire)
S, D = agent_v2._SLEEP_RE, agent_v2._SHUTDOWN_RE
check(not D.search("it's quite hot today"), "shutdown: 'quite' no longer means quit")
check(not D.search("exit the app"), "shutdown: 'exit' inside a command is ignored")
check(D.search("jarvis shut down"), "shutdown: 'jarvis shut down' still works")
check(S.search("thanks jarvis that's all"), "sleep: 'that's all' still works")
check(not S.search("how did i sleep"), "sleep: 'how did I sleep' is a question, not a command")


# 5. open_app refuses an empty name instead of launching a random .desktop file
check(agent_v2.open_app("") == "Which app should I open?", "open_app: empty name rejected")

print("\nALL PASSED" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
