"""
http_client.py
--------------
Lightweight HTTP client with disk caching, exponential backoff,
and academic API compliance headers.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError as UrllibHTTPError
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


# process-wide counters for the run log ('[Time] screening: 6.2 s · 40 web requests, 12 from cache, 2 retries')
STATS = {"requests": 0, "cache_hits": 0, "retries": 0, "failures": 0}
_STATS_LOCK = threading.Lock()


HOST_TIME: Dict[str, List[float]] = {}         # host -> [seconds, requests]: which website the time went to


def _host_time(url: str, seconds: float) -> None:
    with _STATS_LOCK:
        h = HOST_TIME.setdefault(_host(url), [0.0, 0])
        h[0] += seconds
        h[1] += 1


def _count(key: str) -> None:
    with _STATS_LOCK:
        STATS[key] = STATS.get(key, 0) + 1


# ------------------------------------------------------------------ per-site circuit breaker
# A site that keeps answering 429 'too many requests' is paused for a while instead of being asked again and
# again (each refused request still costs a timeout + back-off). Every source of the run shares this table.
BREAKER_THRESHOLD = 3            # rate-limit refusals in a row ...
BREAKER_PAUSE = 300.0            # ... pause that site for 5 minutes
_BREAKER: Dict[str, Dict[str, float]] = {}


def _host(url: str) -> str:
    from urllib.parse import urlsplit
    return (urlsplit(url).hostname or "").lower()


def host_paused(url: str) -> float:
    """Seconds left on a site's pause (0 = not paused)."""
    b = _BREAKER.get(_host(url))
    return max(0.0, b["until"] - time.time()) if b else 0.0


def _rate_limited(url: str) -> None:
    with _STATS_LOCK:
        b = _BREAKER.setdefault(_host(url), {"strikes": 0, "until": 0.0})
        b["strikes"] += 1
        if b["strikes"] >= BREAKER_THRESHOLD:
            b["until"], b["strikes"] = time.time() + BREAKER_PAUSE, 0
            STATS["sites_paused"] = STATS.get("sites_paused", 0) + 1


def _ok(url: str) -> None:
    b = _BREAKER.get(_host(url))
    if b:
        b["strikes"] = 0


class HttpError(RuntimeError):
    """Raised for HTTP, network, decoding, or response-size failures."""


class HttpClient:
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        timeout: float = 25.0,
        retries: int = 2,
        user_agent: str = "BiologyAIResearcher/1.0 (https://github.com/biology-ai; mailto:researcher@biopipeline.org)",
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        self.timeout = float(timeout)
        self.retries = max(0, int(retries))
        self.user_agent = user_agent
        self.extra_headers = {k: v for k, v in (extra_headers or {}).items() if v}  # e.g. x-api-key
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            except OSError:              # a read-only working folder: run without the response cache, never crash
                self.cache_dir = None

    def get_json(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        max_bytes: int = 20 * 1024 * 1024,
    ) -> Any:
        raw = self.get_bytes(url, params=params, max_bytes=max_bytes)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HttpError(f"Invalid JSON response from {url}: {exc}") from exc

    def get_text(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        max_bytes: int = 20 * 1024 * 1024,
        encoding: str = "utf-8",
    ) -> str:
        raw = self.get_bytes(url, params=params, max_bytes=max_bytes)
        try:
            return raw.decode(encoding, errors="replace")
        except Exception as exc:
            raise HttpError(f"Could not decode response from {url}: {exc}") from exc

    def get_bytes(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        max_bytes: int = 50 * 1024 * 1024,
    ) -> bytes:
        full_url = self._build_url(url, params)
        cache_path = self._cache_path(full_url)

        if cache_path and cache_path.exists():
            try:
                data = cache_path.read_bytes()
                if len(data) <= max_bytes:
                    _count("cache_hits")
                    return data
                cache_path.unlink(missing_ok=True)
            except OSError:
                pass

        last_error: Optional[Exception] = None
        left = host_paused(full_url)
        if left:                 # this site refused us repeatedly: do not ask again until the pause is over
            _count("breaker_skips")
            raise HttpError(f"HTTP 429 (site paused for {left:.0f} s after repeated rate limits) for {full_url}")

        for attempt in range(self.retries + 1):
            _count("requests" if attempt == 0 else "retries")
            t_req = time.time()
            try:
                request = Request(
                    full_url,
                    headers={
                        "User-Agent": self.user_agent,
                        "Accept": "application/json, text/xml, */*",
                        "Accept-Language": "en-US,en;q=0.9",
                        **self.extra_headers,
                    },
                    method="GET",
                )

                with urlopen(request, timeout=self.timeout) as response:
                    status = getattr(response, "status", 200)
                    if status < 200 or status >= 300:
                        raise HttpError(f"HTTP {status} from {full_url}")

                    data = self._read_limited(response, max_bytes)

                if cache_path:
                    self._write_cache(cache_path, data)
                _ok(full_url)
                _host_time(full_url, time.time() - t_req)
                return data

            except UrllibHTTPError as exc:
                last_error = exc
                _host_time(full_url, time.time() - t_req)
                retryable = exc.code == 429 or 500 <= exc.code <= 599
                if exc.code == 429:
                    _rate_limited(full_url)
                    if host_paused(full_url):
                        retryable = False            # paused now: stop retrying this site
                    else:
                        try:                         # honour a short Retry-After (never wait more than 10 s)
                            wait = float(exc.headers.get("Retry-After") or 0)
                            if 0 < wait <= 10 and attempt < self.retries:
                                time.sleep(wait)
                        except (TypeError, ValueError, AttributeError):
                            pass
                if not retryable or attempt >= self.retries:
                    body = ""
                    try:
                        body = exc.read(300).decode("utf-8", errors="replace")
                    except Exception:
                        pass
                    _count("failures")
                    detail = f"HTTP {exc.code} from {full_url}"
                    if body:
                        detail += f": {body[:300]}"
                    raise HttpError(detail) from exc

            except (URLError, TimeoutError, OSError, HttpError) as exc:
                last_error = exc
                _host_time(full_url, time.time() - t_req)
                if isinstance(exc, HttpError) and "exceeds limit" in str(exc):
                    raise
                if attempt >= self.retries:
                    _count("failures")
                    raise HttpError(f"Request failed for {full_url}: {exc}") from exc

            time.sleep(min(1.5 ** (attempt + 1), 3.0))

        raise HttpError(f"Request failed for {full_url}: {last_error}")

    @staticmethod
    def _build_url(url: str, params: Optional[Dict[str, Any]]) -> str:
        if not params:
            return url
        clean = []
        for key, value in params.items():
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                for item in value:
                    clean.append((key, str(item)))
            else:
                clean.append((key, str(value)))
        if not clean:
            return url
        separator = "&" if "?" in url else "?"
        return url + separator + urlencode(clean)

    @staticmethod
    def _read_limited(response: Any, max_bytes: int) -> bytes:
        chunks = []
        total = 0
        chunk_size = 64 * 1024
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise HttpError(f"Response exceeds limit of {max_bytes:,} bytes.")
            chunks.append(chunk)
        return b"".join(chunks)

    def _cache_path(self, url: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.bin"

    @staticmethod
    def _write_cache(path: Path, data: bytes) -> None:
        try:
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, path)
        except OSError:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass