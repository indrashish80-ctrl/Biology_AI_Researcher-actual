"""
provider_health.py
------------------
High-resilience health and cooldown tracker for multi-provider AI routing.

Cooldown policy:
    - RATE LIMIT (429 / TPM / RPM): Short breather (3-5 seconds). Rate limits
      are transient traffic spikes, NOT dead API keys.
    - HARD ERROR (401 / 403 / billing / invalid key): Long cooldown (5+ minutes).
      Retrying immediately is futile.
    - SOFT ERROR (500 / 502 / 503 / 504 / timeout): Escalates to cooldown only
      after consecutive threshold failures (default: 3).
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

DEFAULT_SOFT_FAILURE_THRESHOLD = 3
DEFAULT_BASE_COOLDOWN_SECONDS = 20
DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS = 3.5
DEFAULT_MAX_COOLDOWN_SECONDS = 5 * 60

# Unrecoverable account/credential errors -> immediate full cooldown
HARD_ERROR_PATTERNS = [
    r"\b401\b", r"\b403\b", r"unauthorized", r"invalid api key",
    r"insufficient_quota", r"billing", r"account.*(suspended|disabled)",
]

# Transient capacity / rate spikes -> short breather (3.5s)
RATE_LIMIT_PATTERNS = [
    r"\b429\b", r"rate limit", r"rate_limit", r"tpm", r"rpm", r"too many requests",
    r"tokens per minute", r"requests per minute",
]

# Server-side hiccups -> retryable or soft threshold
SOFT_ERROR_PATTERNS = [
    r"\b500\b", r"\b502\b", r"\b503\b", r"\b504\b",
    r"timeout", r"timed out", r"connection", r"unavailable", r"reset by peer",
    r"overloaded", r"service temporarily overloaded",
]


_KEY_RE = re.compile(r"(gsk_|AIza|nvapi-|sk-)[A-Za-z0-9_\-]{6,}")


def redact_keys(text: Any) -> str:
    """Error messages sometimes quote part of the key: never let one reach a log or a report."""
    return _KEY_RE.sub(lambda m: m.group(1) + "[redacted]", str(text or ""))


def is_daily_limit(error_text: str) -> bool:
    """A per-DAY allowance is used up (Groq 'tokens per day (TPD)' / 'requests per day', Gemini 'PerDay' quotas):
    waiting minutes cannot help - the model is skipped for the rest of the run."""
    t = (error_text or "").lower()
    return bool(re.search(r"per day|\btpd\b|\brpd\b|perday|daily", t))


def classify_error(error_text: str) -> str:
    text = (error_text or "").lower()
    # v26.2: a rate-limit message is checked FIRST. Groq's 429 text ends with 'Upgrade ... at
    # console.groq.com/settings/billing', and the word 'billing' made every rate limit a 'hard' (dead key) error:
    # 5-minute lock-outs, and '0 rate-limited' in the report while 138 limits were hit.
    if "insufficient_quota" not in text and any(re.search(p, text) for p in RATE_LIMIT_PATTERNS):
        return "rate_limit"
    for pattern in HARD_ERROR_PATTERNS:
        if re.search(pattern, text):
            return "hard"
    for pattern in RATE_LIMIT_PATTERNS:
        if re.search(pattern, text):
            return "rate_limit"
    for pattern in SOFT_ERROR_PATTERNS:
        if re.search(pattern, text):
            return "soft"
    return "unknown"


@dataclass
class ProviderStats:
    provider: str
    consecutive_failures: int = 0
    total_calls: int = 0
    total_failures: int = 0
    latencies_ms: List[float] = field(default_factory=list)
    last_error: Optional[str] = None
    last_error_class: Optional[str] = None
    cooldown_until: float = 0.0
    cooldown_strikes: int = 0
    status_override: Optional[str] = None

    def avg_latency_ms(self) -> Optional[float]:
        if not self.latencies_ms:
            return None
        recent = self.latencies_ms[-20:]
        return sum(recent) / len(recent)

    def status(self) -> str:
        if self.status_override:
            return self.status_override
        if time.time() < self.cooldown_until:
            return "COOLDOWN"
        if 1 <= self.consecutive_failures < DEFAULT_SOFT_FAILURE_THRESHOLD:
            return "DEGRADED"
        return "HEALTHY"

    def to_dict(self) -> Dict[str, Any]:
        remaining = max(0.0, self.cooldown_until - time.time())
        return {
            "provider": self.provider,
            "status": self.status(),
            "consecutive_failures": self.consecutive_failures,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "avg_latency_ms": round(self.avg_latency_ms(), 1) if self.avg_latency_ms() else None,
            "last_error": self.last_error,
            "last_error_class": self.last_error_class,
            "cooldown_seconds_remaining": round(remaining, 1),
            "cooldown_strikes": self.cooldown_strikes,
        }


class ProviderHealthTracker:
    """
    In-process provider health and cooldown manager.
    Coordinates failover routing and prevents thundering-herd retry storms.
    """

    def __init__(
        self,
        db: Optional[Any] = None,
        soft_failure_threshold: int = DEFAULT_SOFT_FAILURE_THRESHOLD,
        base_cooldown_seconds: int = DEFAULT_BASE_COOLDOWN_SECONDS,
        rate_limit_cooldown_seconds: float = DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS,
        max_cooldown_seconds: int = DEFAULT_MAX_COOLDOWN_SECONDS,
    ):
        self.db = db
        self.soft_failure_threshold = soft_failure_threshold
        self.base_cooldown_seconds = base_cooldown_seconds
        self.rate_limit_cooldown_seconds = rate_limit_cooldown_seconds
        self.max_cooldown_seconds = max_cooldown_seconds
        self._stats: Dict[str, ProviderStats] = {}
        # run instrumentation: tokens, retries and rate-limit hits (all providers) - shown per stage
        self.usage: Dict[str, int] = {"tokens_in": 0, "tokens_out": 0, "tokens_cached": 0, "retries": 0, "rate_limited": 0,
                                      "cache_hits": 0, "requests": 0, "requests_failed": 0}
        # v26.2: WHICH errors happened (provider, kind, message) - shown in the report, so a failed run explains itself
        self.errors: Dict[tuple, Dict[str, Any]] = {}
        self._err_lock = threading.Lock()

    def record_usage(self, provider: str, resp: Any) -> None:
        """Token counts from an OpenAI-style (Groq / NVIDIA) or Gemini response; silently ignored when absent."""
        try:
            u = getattr(resp, "usage", None)
            if u is not None:
                self.usage["tokens_in"] += int(getattr(u, "prompt_tokens", 0) or 0)
                self.usage["tokens_out"] += int(getattr(u, "completion_tokens", 0) or 0)
                det = getattr(u, "prompt_tokens_details", None)          # OpenAI-compatible prompt caching (Groq ...)
                cached = getattr(det, "cached_tokens", None) if det is not None else None
                if cached is None and isinstance(det, dict):
                    cached = det.get("cached_tokens")
                self.usage["tokens_cached"] = self.usage.get("tokens_cached", 0) + int(cached or 0)
                return
            m = getattr(resp, "usage_metadata", None)
            if m is not None:
                self.usage["tokens_in"] += int(getattr(m, "prompt_token_count", 0) or 0)
                self.usage["tokens_out"] += int(getattr(m, "candidates_token_count", 0) or 0)
                # Gemini implicit / explicit context caching
                self.usage["tokens_cached"] = self.usage.get("tokens_cached", 0) + \
                    int(getattr(m, "cached_content_token_count", 0) or 0)
        except Exception:
            pass

    def record_retry(self, provider: str, error_text: str = "") -> None:
        """A repeated attempt (another model, or the same model after a wait)."""
        self.bump("retries")
        if classify_error(error_text) == "rate_limit":
            self.bump("rate_limited")

    def totals(self) -> Dict[str, int]:
        """All AI calls so far (every provider): calls, failures, tokens, retries, rate-limit hits -
        for the per-stage run log."""
        return {"calls": sum(s.total_calls for s in self._stats.values()),
                "failures": sum(s.total_failures for s in self._stats.values()), **self.usage}

    def _get(self, provider: str) -> ProviderStats:
        if provider not in self._stats:
            self._stats[provider] = ProviderStats(provider=provider)
        return self._stats[provider]

    # -------------------------------------------------------------------------
    # Availability Queries
    # -------------------------------------------------------------------------

    def is_available(self, provider: str) -> bool:
        stats = self._get(provider)
        return stats.status() != "COOLDOWN"

    def status(self, provider: str) -> str:
        return self._get(provider).status()

    def all_status(self) -> Dict[str, Dict[str, Any]]:
        return {name: stats.to_dict() for name, stats in self._stats.items()}

    def rank_providers(self, providers: List[str]) -> List[str]:
        tier_rank = {"HEALTHY": 0, "DEGRADED": 1, "COOLDOWN": 2}

        def key(name: str):
            stats = self._get(name)
            latency = stats.avg_latency_ms()
            return (
                tier_rank.get(stats.status(), 3),
                latency if latency is not None else float("inf"),
            )

        return sorted(providers, key=key)

    def disable(self, provider: str) -> None:
        self._get(provider).status_override = "COOLDOWN"

    def enable(self, provider: str) -> None:
        stats = self._get(provider)
        stats.status_override = None
        stats.cooldown_until = 0.0
        stats.consecutive_failures = 0

    # -------------------------------------------------------------------------
    # Recording Outcomes
    # -------------------------------------------------------------------------

    def record_success(self, provider: str, latency_ms: float) -> None:
        stats = self._get(provider)
        stats.total_calls += 1
        stats.consecutive_failures = 0
        stats.latencies_ms.append(latency_ms)
        was_cooldown = stats.cooldown_until > time.time()
        stats.cooldown_until = 0.0

        if self.db is not None and hasattr(self.db, "log_provider_event"):
            try:
                self.db.log_provider_event(provider, "success", latency_ms=latency_ms)
                if was_cooldown:
                    self.db.log_provider_event(provider, "cooldown_cleared")
            except Exception:
                pass

    def note_error(self, provider: str, error_text: str) -> None:
        """Keep a short, key-free record of each distinct error (counted), for the report."""
        msg = redact_keys(error_text)
        kind = classify_error(error_text)
        if kind == "rate_limit" and is_daily_limit(error_text):
            kind = "daily_limit"
        if kind == "unknown":
            low = (error_text or "").lower()
            if "400" in low or "invalid" in low or "bad request" in low:
                kind = "bad_request"
            elif "404" in low or "not found" in low or "does not exist" in low:
                kind = "model_not_found"
            elif "413" in low or "too large" in low or "context" in low and "length" in low:
                kind = "too_large"
        key = (provider, kind, re.sub(r"\d{3,}", "#", msg)[:120])
        with self._err_lock:
            e = self.errors.setdefault(key, {"provider": provider, "kind": kind, "count": 0, "example": msg[:300]})
            e["count"] += 1

    def bump(self, key: str, n: int = 1) -> None:
        with self._err_lock:
            self.usage[key] = self.usage.get(key, 0) + n

    def error_summary(self, top: int = 6) -> List[Dict[str, Any]]:
        with self._err_lock:
            return sorted((dict(v) for v in self.errors.values()), key=lambda e: -e["count"])[:top]

    def record_failure(self, provider: str, error_text: str, latency_ms: Optional[float] = None) -> str:
        self.note_error(provider, error_text)
        stats = self._get(provider)
        stats.total_calls += 1
        stats.total_failures += 1
        stats.consecutive_failures += 1
        stats.last_error = error_text
        error_class = classify_error(error_text)
        stats.last_error_class = error_class
        if latency_ms is not None:
            stats.latencies_ms.append(latency_ms)

        # 1. Rate limits (429 / TPM) -> short 3.5s pause
        if error_class == "rate_limit":
            self.bump("rate_limited")
            stats.cooldown_until = time.time() + self.rate_limit_cooldown_seconds
            if self.db is not None and hasattr(self.db, "log_provider_event"):
                try:
                    self.db.log_provider_event(provider, "rate_limited", latency_ms=latency_ms, error_text=error_text)
                except Exception:
                    pass
            return "COOLDOWN"

        # 2. Hard errors (invalid key, billing dead) -> long cooldown
        if error_class == "hard":
            stats.cooldown_strikes += 1
            stats.cooldown_until = time.time() + self.max_cooldown_seconds
            if self.db is not None and hasattr(self.db, "log_provider_event"):
                try:
                    self.db.log_provider_event(provider, "hard_error", latency_ms=latency_ms, error_text=error_text)
                except Exception:
                    pass
            return "COOLDOWN"

        # 3. Soft errors (500s, 503s, timeouts) -> threshold escalation
        if stats.consecutive_failures >= self.soft_failure_threshold:
            stats.cooldown_strikes += 1
            duration = min(
                self.base_cooldown_seconds * (2 ** (stats.cooldown_strikes - 1)),
                self.max_cooldown_seconds,
            )
            stats.cooldown_until = time.time() + duration
            if self.db is not None and hasattr(self.db, "log_provider_event"):
                try:
                    self.db.log_provider_event(provider, "cooldown_entered", latency_ms=latency_ms, error_text=error_text)
                except Exception:
                    pass
            return "COOLDOWN"

        if self.db is not None and hasattr(self.db, "log_provider_event"):
            try:
                self.db.log_provider_event(provider, "error", latency_ms=latency_ms, error_text=error_text)
            except Exception:
                pass

        return "DEGRADED"