"""
circuit_breaker.py — fail fast when Ollama is down.

Without this, every ask_llm() call against a dead or wedged Ollama waits the
full LLM_TIMEOUT before giving up, and each new voice command pays that cost
again. The breaker counts consecutive failures:

  CLOSED    → calls go through; failures are counted.
  OPEN      → after `failure_threshold` failures in a row, calls are rejected
              instantly with CircuitOpenError (Jarvis answers immediately).
  HALF_OPEN → once `recovery_timeout` seconds pass, one probe call is let
              through. Success closes the circuit, failure re-opens it.

Ported (sync version) from AIproductfactory python-backend/llm/circuit_breaker.py.
"""
import threading
import time
from enum import Enum

from jarvis_log import log


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised instead of calling a backend whose circuit is open."""

    def __init__(self, name, retry_after):
        self.name = name
        self.retry_after = retry_after
        super().__init__(f"circuit '{name}' is open; retry in {retry_after:.0f}s")


class CircuitBreaker:
    def __init__(self, name, failure_threshold=2, recovery_timeout=30.0, clock=time.monotonic):
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if recovery_timeout < 0:
            raise ValueError("recovery_timeout must not be negative")
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._clock = clock
        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._probe_in_flight = False

    @property
    def state(self):
        with self._lock:
            return self._current_state()

    def retry_after(self):
        """Seconds until an open circuit will allow a probe call."""
        with self._lock:
            if self._current_state() != CircuitState.OPEN:
                return 0.0
            return max(0.0, self._opened_at + self.recovery_timeout - self._clock())

    def call(self, func, *args, is_failure=None, **kwargs):
        """Run func(*args, **kwargs) through the breaker.

        `is_failure(exc)` decides which exceptions say the backend is unhealthy;
        by default every exception counts. Exceptions are always re-raised.
        """
        is_probe = self._acquire()
        try:
            result = func(*args, **kwargs)
        except BaseException as e:
            if isinstance(e, Exception) and (is_failure is None or is_failure(e)):
                self._record(success=False, is_probe=is_probe)
            else:
                # Not the backend's fault (e.g. KeyboardInterrupt, bad input):
                # just free the probe slot so the next call can test it.
                self._release_probe(is_probe)
            raise
        self._record(success=True, is_probe=is_probe)
        return result

    # ── internals (caller holds self._lock where noted) ──────────────────────

    def _current_state(self):
        # Lock held. OPEN turns HALF_OPEN lazily once the cooldown elapses.
        if (self._state == CircuitState.OPEN
                and self._clock() - self._opened_at >= self.recovery_timeout):
            self._state = CircuitState.HALF_OPEN
            self._probe_in_flight = False
        return self._state

    def _acquire(self):
        with self._lock:
            state = self._current_state()
            if state == CircuitState.CLOSED:
                return False
            if state == CircuitState.HALF_OPEN and not self._probe_in_flight:
                self._probe_in_flight = True
                return True
            retry_after = max(0.0, self._opened_at + self.recovery_timeout - self._clock())
        raise CircuitOpenError(self.name, retry_after)

    def _release_probe(self, is_probe):
        if not is_probe:
            return
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._probe_in_flight = False

    def _record(self, success, is_probe):
        with self._lock:
            state = self._current_state()
            if state == CircuitState.HALF_OPEN:
                if not is_probe:
                    return
                if success:
                    self._close()
                else:
                    self._trip("probe failed")
                return
            if state == CircuitState.OPEN:
                return
            if success:
                self._failures = 0
                return
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._trip(f"{self._failures} consecutive failures")

    def _trip(self, reason):
        self._state = CircuitState.OPEN
        self._opened_at = self._clock()
        self._probe_in_flight = False
        log.warning("Circuit '%s' OPEN (%s); failing fast for %.0fs",
                    self.name, reason, self.recovery_timeout)

    def _close(self):
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._probe_in_flight = False
        log.info("Circuit '%s' CLOSED: backend recovered", self.name)
