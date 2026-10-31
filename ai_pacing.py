"""
ai_pacing.py - stop the retry storm: pace AI requests instead of colliding with rate limits
===========================================================================================

The problem this solves, measured on a real run: 74 AI calls produced **65 retries** and took 21 minutes. Three
extraction threads shared one free API key, each sent requests as fast as it could, each got "429 too many requests",
and each then slept a second and tried the next model on the same exhausted key. The retries were the run time.

Nothing here talks to a network. It is a traffic light shared by every thread:

  * a token bucket per provider - requests go out at a rate the provider accepts, so they are never refused
  * a cool-down per provider - a provider that just refused us is skipped for a few seconds, and the caller moves to
    another provider immediately instead of sleeping on the exhausted one
  * the wait a provider ASKS for is obeyed ('try again in 7.5s' in its error message) rather than guessed

Rates are per minute and deliberately conservative - the published free-tier limits of each service. Raise them with
BIOSYN_RATE_<PROVIDER> (requests per minute) if you have a paid key.
"""
from __future__ import annotations

import os
import re
import threading
import time
from typing import Dict, Optional

PACING_VERSION = "1.0"

# requests per minute that the free tier of each service accepts; the bucket keeps us just under it
DEFAULT_RPM: Dict[str, float] = {
    "groq": 28.0,          # free tier is ~30/min per model family
    "gemini": 14.0,        # free tier is ~15/min
    "nvidia": 38.0,
    "openrouter": 18.0,
    "together": 55.0,
    "cerebras": 28.0,
}
_FALLBACK_RPM = 20.0
MAX_WAIT_S = 20.0          # never block a thread longer than this for pacing; let it try another provider instead


def rpm_for(provider: str) -> float:
    """The rate for a provider: the environment setting if given, otherwise its published free-tier limit."""
    env = os.getenv(f"BIOSYN_RATE_{(provider or '').upper()}")
    if env:
        try:
            v = float(env)
            if v > 0:
                return v
        except ValueError:
            pass
    return DEFAULT_RPM.get((provider or "").lower(), _FALLBACK_RPM)


class _Bucket:
    """A token bucket: `rpm` tokens refill per minute, one token per request, shared by all threads."""

    def __init__(self, rpm: float):
        self.rpm = max(1.0, float(rpm))
        self.capacity = max(1.0, self.rpm / 6.0)      # a small burst (10 seconds' worth) is allowed
        self.tokens = self.capacity
        self.updated = time.monotonic()
        self.lock = threading.Lock()

    def take(self, max_wait: float = MAX_WAIT_S) -> float:
        """Wait until a request may go out. Returns the seconds waited, or -1 if the wait would exceed max_wait."""
        with self.lock:
            now = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rpm / 60.0)
            self.updated = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return 0.0
            need = (1.0 - self.tokens) * 60.0 / self.rpm
            if need > max_wait:
                return -1.0
            self.tokens = 0.0
            self.updated = now + need                 # this request owns that slot; other threads queue after it
        time.sleep(need)
        return need


_BUCKETS: Dict[str, _Bucket] = {}
_COOL_UNTIL: Dict[str, float] = {}
_LOCK = threading.Lock()
STATS = {"paced_waits": 0, "paced_seconds": 0.0, "cooldowns": 0, "switches": 0}


def _bucket(provider: str) -> _Bucket:
    key = (provider or "").lower()
    with _LOCK:
        b = _BUCKETS.get(key)
        if b is None or b.rpm != max(1.0, rpm_for(key)):
            b = _Bucket(rpm_for(key))
            _BUCKETS[key] = b
        return b


def wait_turn(provider: str, max_wait: float = MAX_WAIT_S) -> bool:
    """Block until this provider may be called. False means 'too long - try another provider'."""
    if os.getenv("BIOSYN_NO_PACING"):
        return True
    waited = _bucket(provider).take(max_wait)
    if waited < 0:
        return False
    if waited > 0:
        with _LOCK:
            STATS["paced_waits"] += 1
            STATS["paced_seconds"] = round(STATS["paced_seconds"] + waited, 2)
    return True


def retry_after_seconds(error_text: str) -> Optional[float]:
    """The wait a provider asks for in its own error message ('Please try again in 7.5s', 'retry-after: 12')."""
    t = str(error_text or "")
    m = re.search(r"(?:try again in|retry[- ]after[\"':= ]+|wait\s+)\s*(\d+(?:\.\d+)?)\s*(ms|s|sec|seconds|m|min|minutes)?",
                  t, re.I)
    if not m:
        return None
    v = float(m.group(1))
    unit = (m.group(2) or "s").lower()
    if unit == "ms":
        v /= 1000.0
    elif unit.startswith("m") and unit != "ms":
        v *= 60.0
    return min(v, 300.0)


def cool_down(provider: str, seconds: Optional[float] = None, error_text: str = "") -> float:
    """Skip this provider for a while: it just refused us. Returns the cool-down used."""
    secs = seconds if seconds is not None else retry_after_seconds(error_text)
    if secs is None:
        secs = 8.0
    secs = max(0.5, min(float(secs), 300.0))
    with _LOCK:
        _COOL_UNTIL[(provider or "").lower()] = max(_COOL_UNTIL.get((provider or "").lower(), 0.0),
                                                     time.monotonic() + secs)
        STATS["cooldowns"] += 1
    return secs


def cooling(provider: str) -> float:
    """Seconds until this provider is worth trying again (0 = now)."""
    with _LOCK:
        return max(0.0, _COOL_UNTIL.get((provider or "").lower(), 0.0) - time.monotonic())


def note_switch() -> None:
    with _LOCK:
        STATS["switches"] += 1


def workers_for(n_providers: int, n_items: int, default_one_key: int = 3) -> int:
    """How many extraction threads are worth running.

    With several keys, one thread per key (separate rate limits). With one key the bucket decides the pace, so extra
    threads only queue - but a few are still useful, because a thread waiting on a slow reply is not using its slot.
    """
    if n_items <= 1:
        return 1
    if n_providers > 1:
        return min(max(n_providers, 2), 8, n_items)
    env = os.getenv("BIOSYN_EXTRACTION_WORKERS")
    if env and env.isdigit() and int(env) > 0:
        return min(int(env), n_items)
    return min(default_one_key, n_items)


def summary() -> Dict[str, object]:
    """What pacing did this run, for the timing table."""
    with _LOCK:
        return {**STATS, "rates_per_minute": {k: b.rpm for k, b in _BUCKETS.items()}}


def reset() -> None:
    with _LOCK:
        _BUCKETS.clear()
        _COOL_UNTIL.clear()
        STATS.update({"paced_waits": 0, "paced_seconds": 0.0, "cooldowns": 0, "switches": 0})
