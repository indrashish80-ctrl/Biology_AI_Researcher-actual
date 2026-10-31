"""
ncbi_throttle.py
----------------
One process-wide rate limiter for every NCBI E-utilities call (PubMed, PMC, Gene,
Taxonomy, GEO). NCBI allows 3 requests/second without an API key and 10 with one.
The retrieval layer queries several sources in parallel threads, so a per-call
sleep is not enough - without a shared limiter NCBI answers HTTP 429 and records
are silently lost.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

_lock = threading.Lock()
_next_slot = 0.0


_KEY: list = [None]


def set_api_key(key: Optional[str]) -> None:
    """v27: the NCBI API key entered on the API keys page (sent with the run). With a key NCBI allows 10 requests a
    second instead of 3 - searching and downloading papers is about three times faster."""
    _KEY[0] = (key or "").strip() or None


def api_key() -> Optional[str]:
    import os
    return _KEY[0] or os.getenv("NCBI_API_KEY") or None


def ncbi_wait(has_api_key: bool = False) -> float:
    """Block until this thread may send the next NCBI request.

    Returns the moment the request may be sent (the slot this call was given). Slots are handed out under the lock and
    are always `interval` apart, so the limit holds however the operating system then schedules the threads: a thread
    delayed after its sleep sends LATER than its slot, never earlier. The returned value is the guarantee, which is
    what callers and tests should measure - a timestamp taken after the sleep also includes scheduling delay.
    """
    global _next_slot
    interval = 0.11 if (has_api_key or api_key()) else 0.36
    with _lock:
        now = time.monotonic()
        slot = max(now, _next_slot)
        _next_slot = slot + interval
    wait = slot - now
    if wait > 0:
        time.sleep(wait)
    return slot
