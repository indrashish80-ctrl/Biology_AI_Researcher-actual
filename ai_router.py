"""
ai_router.py
------------
High-throughput AI Router with universal free-tier model cascading:
    Groq (llama-3.1-8b-instant -> 70b -> gpt-oss-20b) -> NVIDIA NIM -> Gemini

Keys:
    AIRouter()                -> CLI behaviour, reads GROQ_API_KEY / GEMINI_API_KEY /
                                 NVIDIA_NIM_API_KEY from the environment / .env.
    AIRouter(keys={...})      -> web behaviour, uses ONLY the keys passed in for this
                                 request. The environment is never read or written, so
                                 one user's keys can never leak into another's run.

Deep reasoning:
    ask_reasoning(prompt)     -> free open-weight reasoning models (Groq gpt-oss-120b /
                                 Qwen, Gemini thinking, NVIDIA NIM) returning the answer
                                 plus the model's reasoning trace when exposed.
"""

from __future__ import annotations

import collections
import concurrent.futures
import logging
import contextlib
import os
import threading
import time
import warnings
import re
from typing import Any, Dict, List, Optional

# the Gemini SDK logs an advisory "automatic function calling (AFC)" notice on every call; it is not an error
logging.getLogger("google_genai").setLevel(logging.ERROR)

from dotenv import load_dotenv
from openai import OpenAI
from provider_health import ProviderHealthTracker

load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning)

# Free-tier production models on Groq (checked against console.groq.com/docs/models,
# Oct 2026). gemma2-9b-it was retired, so it is replaced by gpt-oss-20b.
GROQ_MODELS = [
    "llama-3.1-8b-instant",       # fastest, used for bulk JSON extraction
    "llama-3.3-70b-versatile",    # 70B parameter fallback
    "openai/gpt-oss-20b",         # open-weight fallback
]

# Open-weight reasoning models on Groq. DeepSeek-R1 is no longer served by Groq;
# gpt-oss-120b (reasoning_effort=high) and Qwen are the current free equivalents.
GROQ_REASONING_MODELS = [
    m.strip() for m in os.getenv(
        "GROQ_REASONING_MODELS", "openai/gpt-oss-120b,qwen/qwen3.8-27b"
    ).split(",") if m.strip()
]

# Current Gemini Flash models (gemini-2.0 / 1.5 flash have been shut down).
GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
]

NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-super-120b-a12b")
NVIDIA_REASONING_MODEL = os.getenv("NVIDIA_REASONING_MODEL", NVIDIA_MODEL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
PROVIDER_PRIORITY = ("groq", "nvidia", "gemini")
AI_TIMEOUT_S = float(os.getenv("BIOSYN_AI_TIMEOUT_S") or 120)
# v27: a request that meets a per-minute rate limit WAITS for capacity (up to this long in total) instead of failing
RATE_WAIT_MAX_S = float(os.getenv("BIOSYN_RATE_WAIT_MAX_S") or 180)
# v26.2: stop a run early (with the reason) when most AI requests fail - a report built on a third of its AI calls is
# not worth 55 minutes. Window of recent requests and the failure share that stops the run:
OUTAGE_WINDOW = int(os.getenv("BIOSYN_OUTAGE_WINDOW") or 30)
OUTAGE_SHARE = float(os.getenv("BIOSYN_OUTAGE_SHARE") or 0.7)


def _bump(health: Any, key: str) -> None:
    """Count on the tracker (thread-safe bump when it has one; a plain usage dict otherwise)."""
    f = getattr(health, "bump", None)
    if callable(f):
        f(key)
        return
    u = getattr(health, "usage", None)
    if isinstance(u, dict):
        u[key] = u.get(key, 0) + 1


def fit_max_tokens(error_text: str, max_tokens: int, floor: int = 512) -> Optional[int]:
    """v26.2: Groq's free tier counts the ANSWER allowance (max_tokens) against a per-minute token limit, so a long
    prompt with max_tokens=8192 is refused as 'Request too large ... Limit 8000, Requested 12345' every time. The
    numbers in that message give the allowance that fits; None when even `floor` answer tokens would not fit."""
    m = re.search(r"limit\D{0,3}(\d+)\D{1,20}requested\D{0,3}(\d+)", str(error_text or ""), re.I)
    if not m:
        return None
    limit, requested = int(m.group(1)), int(m.group(2))
    new = max_tokens - (requested - limit) - 64
    return new if new >= floor and new < max_tokens else None


class AIOutage(BaseException):
    """Most recent AI requests failed on every provider: the run stops and says why (BaseException, so a per-paper
    'except Exception' cannot swallow it and grind on)."""

# Which provider does which job when the user supplied several keys (each falls back to the others).
#   Groq     - fastest: bulk JSON extraction of claims / study design, screening, the agent loop
#   Gemini   - long context: planning queries, hypotheses and the written synthesis
#   NVIDIA   - large reasoning model: adversarial critique, second opinion on synthesis
TASK_ORDER = {
    "extraction": ("groq", "gemini", "nvidia"),
    "screening": ("groq", "gemini", "nvidia"),
    "agent": ("groq", "gemini", "nvidia"),
    "planning": ("gemini", "groq", "nvidia"),
    "hypotheses": ("gemini", "nvidia", "groq"),
    "synthesis": ("groq", "gemini", "nvidia"),     # Groq's gpt-oss-120b is the reasoning model of choice
    "critique": ("nvidia", "gemini", "groq"),
}
TASK_LABEL = {"extraction": "claim & study-design extraction", "screening": "screening", "agent": "agent loop",
              "planning": "search planning", "hypotheses": "hypotheses & gaps", "synthesis": "reasoned synthesis",
              "critique": "adversarial critique"}


def _is_rate_limit(error_text: str) -> bool:
    """Is this error 'you are sending too fast' rather than 'this request is wrong'?"""
    t = (error_text or "").lower()
    return ("429" in t or "rate limit" in t or "rate_limit" in t or "too many requests" in t
            or "quota" in t or "resource_exhausted" in t or "requests per" in t)


class AIRouter:
    def __init__(
        self,
        health_tracker: Optional[ProviderHealthTracker] = None,
        keys: Optional[Dict[str, Optional[str]]] = None,
    ):
        if keys is not None:
            # Request-scoped keys supplied by the user (web UI). Never touch os.environ.
            self.gemini_key = (keys.get("gemini") or "").strip() or None
            self.groq_key = (keys.get("groq") or "").strip() or None
            self.nvidia_key = (keys.get("nvidia") or "").strip() or None
        else:
            self.gemini_key = os.getenv("GEMINI_API_KEY")
            self.groq_key = os.getenv("GROQ_API_KEY")
            self.nvidia_key = os.getenv("NVIDIA_NIM_API_KEY")

        self.gemini = None
        if self.gemini_key:
            try:
                from google import genai
                from google.genai import types as _gt
                # v26.2: a request that hangs is abandoned after BIOSYN_AI_TIMEOUT_S (default 120 s) and goes to
                # the next provider; the library default could wait many minutes per request
                self.gemini = genai.Client(api_key=self.gemini_key,
                                           http_options=_gt.HttpOptions(timeout=int(AI_TIMEOUT_S * 1000)))
            except Exception:
                pass

        # v26.2: explicit timeout; ONE quiet library retry (our own fallback and pacing handle the rest - the default of
        # 2 hidden retries x 10 minutes could stall a run for half an hour on one bad request)
        self.groq = OpenAI(api_key=self.groq_key, base_url="https://api.groq.com/openai/v1", timeout=AI_TIMEOUT_S,
                           max_retries=1) if self.groq_key else None
        self.nvidia = OpenAI(api_key=self.nvidia_key, base_url="https://integrate.api.nvidia.com/v1",
                             timeout=AI_TIMEOUT_S, max_retries=1) if self.nvidia_key else None
        self.health = health_tracker or ProviderHealthTracker()
        self._local = threading.local()        # per-thread preferred provider (parallel paper extraction)

    def groq_models(self) -> List[str]:
        """v27: every chat model this Groq key can use - the known good ones first, then any other chat model the key
        lists (each Groq model has its OWN rate limits, so one free key carries several times the work). Asked once."""
        if "_groq_models" in self.__dict__:
            return self._groq_models
        found: List[str] = []
        if self.groq is not None and not os.getenv("BIOSYN_OFFLINE"):
            try:
                lst = self.groq.with_options(timeout=15, max_retries=0).models.list()
                ids = [getattr(m, "id", None) for m in getattr(lst, "data", []) or []]
                skip_re = re.compile(r"whisper|guard|orpheus|safeguard|tts|playai|distil|embed|vision|compound", re.I)
                found = [i for i in ids if i and not skip_re.search(i) and getattr(next((m for m in lst.data if m.id == i), None),
                                                                              "active", True) is not False]
            except Exception:
                found = []
        order = [m for m in GROQ_MODELS if not found or m in found]
        order += sorted(m for m in found if m not in order and m not in GROQ_REASONING_MODELS)
        order += [m for m in found if m not in order]            # reasoning models last: slower, but extra capacity
        self._groq_models = order or list(GROQ_MODELS)
        return self._groq_models

    def _sleep_cancellable(self, seconds: float, cap: float) -> bool:
        """Wait (in small steps, so a cancelled job stops at once). False when the wait would exceed `cap`."""
        if seconds > cap or os.getenv("BIOSYN_NO_PACING"):
            return False
        end = time.time() + seconds
        ev = getattr(self, "cancel_event", None)
        while time.time() < end:
            if ev is not None and ev.is_set():
                from core_utils import JobCancelled
                raise JobCancelled("cancelled by the user")
            time.sleep(min(1.0, max(0.0, end - time.time())))
        return True

    def configured_providers(self) -> List[str]:
        providers = []
        if self.groq:
            providers.append("groq")
        if self.nvidia:
            providers.append("nvidia")
        if self.gemini:
            providers.append("gemini")
        return providers

    def order_for(self, task: Optional[str] = None) -> List[str]:
        """Available providers in the order that suits the task; a thread's preferred provider goes first."""
        avail = self.get_available_providers()
        pref = TASK_ORDER.get(task or "", PROVIDER_PRIORITY)
        ordered = sorted(avail, key=lambda p: pref.index(p) if p in pref else 99)
        mine = getattr(self._local, "preferred", None)
        if mine in ordered:
            ordered.remove(mine)
            ordered.insert(0, mine)
        return ordered

    def assignments(self) -> Dict[str, str]:
        """{'claim & study-design extraction': 'groq', ...} - which provider leads each task in this run."""
        out = {}
        for task in TASK_ORDER:
            order = self.order_for(task)
            if order:
                out[TASK_LABEL[task]] = order[0]
        return out

    @contextlib.contextmanager
    def prefer(self, provider: Optional[str]):
        """Within this block the current thread asks `provider` first (others remain fallbacks)."""
        old = getattr(self._local, "preferred", None)
        self._local.preferred = provider
        try:
            yield
        finally:
            self._local.preferred = old

    def get_available_providers(self) -> List[str]:
        available = [p for p in self.configured_providers() if self.health.is_available(p)]
        return sorted(available, key=lambda p: PROVIDER_PRIORITY.index(p) if p in PROVIDER_PRIORITY else 99)

    def ask_groq(self, prompt: str, model: Optional[str] = None) -> str:
        if not self.groq or not self.health.is_available("groq"):
            raise RuntimeError("Groq unavailable.")
        
        started = time.perf_counter()
        
        # Build candidate models list (tries 8b instant first to prevent 404 access errors)
        candidates = []
        if model:
            candidates.append(model)
        for m in self.groq_models():
            if m not in candidates:
                candidates.append(m)

        # a model that refused a request because it does not exist / is not on this key (skip always) or because the
        # request is too large for its per-minute token limit (skip for requests at least that large) is not asked
        # again on every call - that was ~1 wasted call + 1 s per paper (8 retries in 9 calls in a real run)
        skip = self.__dict__.setdefault("_groq_skip", {})
        rest = self.__dict__.setdefault("_groq_rest", {})          # v26.2: model -> time it may be asked again
        now = time.time()
        why = self.__dict__.setdefault("_groq_skip_why", {})       # model -> 'daily' | 'size' | 'missing'
        usable = [m for m in candidates if not (m in skip and len(prompt) >= skip[m]) and rest.get(m, 0) <= now]
        if not usable:
            soon = [m for m in candidates if not (m in skip and len(prompt) >= skip[m])]
            if not soon:
                kinds = {why.get(m, "missing") for m in candidates}
                if kinds == {"size"} or ("size" in kinds and "daily" not in kinds):
                    msg = ("Groq: this request is larger than the free tier's per-minute token limit of every model "
                           "on the key - it is read in smaller parts or by another provider")
                else:
                    msg = ("Groq: every model on this key has used up today's token allowance - add a Gemini or "
                           "NVIDIA key, or run again tomorrow")
                getattr(self.health, "note_error", lambda *a: None)("groq", msg)
                raise RuntimeError(msg)
            # v27: every model is only RESTING (per-minute limit): wait for the first to free up, then use it - a
            # rate limit slows a run down, it never fails the request
            m0 = min(soon, key=lambda m: rest.get(m, 0))
            if not self._sleep_cancellable(max(0.0, rest.get(m0, 0) - time.time()), RATE_WAIT_MAX_S):
                raise RuntimeError(f"Groq rate limit: every model is resting - try again in "
                                   f"{max(1.0, rest.get(m0, 0) - time.time()):.0f}s")
            usable = [m0]
        candidates = usable
        last_error = None
        for _round in range(3):
            if _round:
                # v27: every model we tried is only RESTING (per-minute limit) - wait for the soonest, try again
                now = time.time()
                soon = [m for m in self.groq_models() if not (m in skip and len(prompt) >= skip[m])]
                if not soon:
                    break
                m0 = min(soon, key=lambda m: rest.get(m, 0))
                if not self._sleep_cancellable(max(0.0, rest.get(m0, 0) - now), RATE_WAIT_MAX_S):
                    break
                candidates = [m for m in soon if rest.get(m, 0) <= time.time()] or [m0]
            res = self._groq_round(prompt, candidates, skip, rest, why, started)
            if isinstance(res, str):
                return res
            last_error = res
            if not (_is_rate_limit(str(last_error)) and not __import__("provider_health").is_daily_limit(str(last_error))):
                break
        self.health.record_failure("groq", str(last_error), (time.perf_counter() - started) * 1000)
        if all(skip.get(m) == 0 and why.get(m) == "daily" for m in self.groq_models()):
            msg = ("Groq: every model on this key has used up today's token allowance - add a Gemini or NVIDIA key, "
                   "or run again tomorrow")
            getattr(self.health, "note_error", lambda *a: None)("groq", msg)
            raise RuntimeError(msg) from last_error
        raise last_error

    def _groq_round(self, prompt: str, candidates: List[str], skip: Dict[str, int], rest: Dict[str, float],
                    why: Dict[str, str], started: float):
        """One pass over the candidate models: the answer (str) or the last error."""
        last_error = None
        for target_model in candidates:
            try:
                resp = self.groq.chat.completions.create(
                    model=target_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=2048,
                )
                self.health.record_success("groq", (time.perf_counter() - started) * 1000)
                self.health.record_usage("groq", resp)
                return resp.choices[0].message.content or ""
            except Exception as error:
                last_error = error
                err_str = str(error).lower()
                from provider_health import is_daily_limit
                if any(k in err_str for k in ("404", "model_not_found", "does not exist", "decommission")) or \
                        ("access" in err_str and "403" in err_str):
                    skip[target_model] = 0                                    # never on this key
                    why[target_model] = "missing"
                    self.health.record_retry("groq", err_str)
                    continue
                if "413" in err_str or "too large" in err_str or "request_too_large" in err_str:
                    fit = fit_max_tokens(err_str, 2048)
                    if fit:                     # shorter answer allowance that fits the per-minute token limit
                        try:
                            resp = self.groq.chat.completions.create(
                                model=target_model, messages=[{"role": "user", "content": prompt}],
                                temperature=0.2, max_tokens=fit)
                            self.health.record_success("groq", (time.perf_counter() - started) * 1000)
                            self.health.record_usage("groq", resp)
                            return resp.choices[0].message.content or ""
                        except Exception as error2:
                            last_error = error2
                    skip[target_model] = min(skip.get(target_model, 10 ** 9), len(prompt))
                    why[target_model] = "size"
                    self.health.record_retry("groq", err_str)
                    continue
                if _is_rate_limit(err_str):
                    # Groq's limits are PER MODEL: this model rests, the next model is tried at once. A per-DAY
                    # allowance that is used up skips the model for the whole run (waiting minutes cannot help).
                    if is_daily_limit(err_str):
                        skip[target_model] = 0
                        why[target_model] = "daily"
                        getattr(self.health, "note_error", lambda *a: None)("groq", f"{target_model}: daily allowance used up - {error}")
                    else:
                        from ai_pacing import retry_after_seconds
                        rest[target_model] = time.time() + (retry_after_seconds(err_str) or 20.0)
                    self.health.record_retry("groq", err_str)
                    continue
                break
        return last_error

    def ask_nvidia(self, prompt: str, model: str = NVIDIA_MODEL) -> str:
        if not self.nvidia or not self.health.is_available("nvidia"):
            raise RuntimeError("NVIDIA NIM unavailable.")
        started = time.perf_counter()
        for attempt in range(2):
            try:
                resp = self.nvidia.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=2048,
                )
                self.health.record_success("nvidia", (time.perf_counter() - started) * 1000)
                self.health.record_usage("nvidia", resp)
                return resp.choices[0].message.content or ""
            except Exception as error:
                err_msg = str(error).lower()
                if ("503" in err_msg or "overloaded" in err_msg) and attempt == 0:
                    self.health.record_retry("nvidia", err_msg)
                    time.sleep(2.0)
                    continue
                self.health.record_failure("nvidia", str(error), (time.perf_counter() - started) * 1000)
                raise

    def ask_gemini(self, prompt: str, model: Optional[str] = None) -> str:
        if not self.gemini or not self.health.is_available("gemini"):
            raise RuntimeError("Gemini unavailable.")
        
        started = time.perf_counter()
        gskip = self.__dict__.setdefault("_gemini_skip", set())      # v26.2: models with today's quota used up
        candidates = [model] if model else [m for m in GEMINI_MODELS if m not in gskip] or GEMINI_MODELS[-1:]

        last_error = None
        for target_model in candidates:
            try:
                from google.genai import types
                config = types.GenerateContentConfig(temperature=0.2)
                resp = self.gemini.models.generate_content(model=target_model, contents=prompt, config=config)
                self.health.record_success("gemini", (time.perf_counter() - started) * 1000)
                self.health.record_usage("gemini", resp)
                return resp.text or ""
            except Exception as error:
                last_error = error
                if "404" in str(error) or "not found" in str(error).lower():
                    self.health.record_retry("gemini", str(error))
                    continue
                from provider_health import is_daily_limit
                if _is_rate_limit(str(error)):
                    if is_daily_limit(str(error)):
                        gskip.add(target_model)          # this model's daily quota is used up: the next model
                    self.health.record_retry("gemini", str(error))
                    continue                             # Gemini quotas are per model too
                if re.search(r"\b50[034]\b|deadline|unavailable|overloaded|timed? ?out", str(error), re.I):
                    self.health.record_retry("gemini", str(error))
                    continue                             # v27: a slow / overloaded model -> the next (lighter) one
                break

        self.health.record_failure("gemini", str(last_error), (time.perf_counter() - started) * 1000)
        raise last_error

    # -------------------------------------------------------------------------
    # Deep reasoning (free open-weight reasoning models)
    # -------------------------------------------------------------------------

    def _reason_groq(self, prompt: str) -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        skip = self.__dict__.setdefault("_groq_skip", {})
        for model in GROQ_REASONING_MODELS:
            if skip.get(model) == 0:
                continue                          # v26.2: not on this key / today's allowance used up
            started = time.perf_counter()
            try:
                kwargs: Dict[str, Any] = dict(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=8192,
                )
                if model.startswith("openai/gpt-oss"):
                    kwargs["reasoning_effort"] = "high"
                else:
                    kwargs["extra_body"] = {"reasoning_format": "parsed"}
                try:
                    resp = self.groq.chat.completions.create(**kwargs)
                except Exception as e413:
                    fit = fit_max_tokens(str(e413), kwargs["max_tokens"], floor=1024)
                    if not fit:
                        raise
                    kwargs["max_tokens"] = fit      # v26.2: an answer allowance that fits the free tier's limit
                    resp = self.groq.chat.completions.create(**kwargs)
                msg = resp.choices[0].message
                reasoning = getattr(msg, "reasoning", None) or (getattr(msg, "model_extra", None) or {}).get("reasoning")
                content, think = _split_think(msg.content or "")
                self.health.record_success("groq", (time.perf_counter() - started) * 1000)
                self.health.record_usage("groq", resp)
                return {"provider": "groq", "model": model, "content": content,
                        "reasoning": reasoning or think}
            except Exception as error:
                last_error = error
                from provider_health import is_daily_limit
                es = str(error).lower()
                if is_daily_limit(es) or "404" in es or "model_not_found" in es or "decommission" in es:
                    skip[model] = 0
                continue
        if last_error is not None:
            self.health.record_failure("groq", str(last_error))
        raise last_error or RuntimeError("No Groq reasoning model available.")

    def _reason_gemini(self, prompt: str) -> Dict[str, Any]:
        from google.genai import types
        last_error: Optional[Exception] = None
        for model in GEMINI_MODELS:
            started = time.perf_counter()
            try:
                config = types.GenerateContentConfig(
                    temperature=0.3,
                    thinking_config=types.ThinkingConfig(include_thoughts=True),
                )
                try:
                    resp = self.gemini.models.generate_content(model=model, contents=prompt, config=config)
                except Exception as exc:
                    if "400" not in str(exc) and "thinking" not in str(exc).lower():
                        raise
                    # model without thinking support: ask again without thought summaries
                    resp = self.gemini.models.generate_content(
                        model=model, contents=prompt, config=types.GenerateContentConfig(temperature=0.3))
                thoughts, answer = [], []
                for cand in (resp.candidates or [])[:1]:
                    for part in (cand.content.parts if cand.content else []) or []:
                        if not getattr(part, "text", None):
                            continue
                        (thoughts if getattr(part, "thought", False) else answer).append(part.text)
                self.health.record_success("gemini", (time.perf_counter() - started) * 1000)
                self.health.record_usage("gemini", resp)
                return {"provider": "gemini", "model": model,
                        "content": "".join(answer) or (resp.text or ""),
                        "reasoning": "\n".join(thoughts) or None}
            except Exception as error:
                last_error = error
                if "404" in str(error) or "not found" in str(error).lower():
                    continue
                break
        self.health.record_failure("gemini", str(last_error))
        raise last_error or RuntimeError("Gemini reasoning unavailable.")

    def _reason_nvidia(self, prompt: str) -> Dict[str, Any]:
        started = time.perf_counter()
        try:
            resp = self.nvidia.chat.completions.create(
                model=NVIDIA_REASONING_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=8192,
            )
            msg = resp.choices[0].message
            content, think = _split_think(msg.content or "")
            reasoning = getattr(msg, "reasoning_content", None) or think
            self.health.record_success("nvidia", (time.perf_counter() - started) * 1000)
            self.health.record_usage("nvidia", resp)
            return {"provider": "nvidia", "model": NVIDIA_REASONING_MODEL, "content": content, "reasoning": reasoning}
        except Exception as error:
            self.health.record_failure("nvidia", str(error), (time.perf_counter() - started) * 1000)
            raise

    def ask_reasoning(self, prompt: str) -> Dict[str, Any]:
        """
        Multi-step deduction with a free reasoning model.
        Order: Groq reasoning models -> Gemini (thinking) -> NVIDIA NIM -> ordinary ask().
        Returns {"provider", "model", "content", "reasoning"}.
        """
        errors = []
        table = {"groq": (self._reason_groq, self.groq), "gemini": (self._reason_gemini, self.gemini),
                 "nvidia": (self._reason_nvidia, self.nvidia)}
        order = [p for p in TASK_ORDER["synthesis"] if p in table]
        for name in order:
            fn, client = table[name]
            if not client or not self.health.is_available(name):
                continue
            try:
                res = fn(prompt)
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                continue
            if not (res.get("content") or "").strip():
                # an empty answer is a failure too: try the next provider instead of returning nothing
                errors.append(f"{name}: empty response")
                continue
            return res
        # Last resort: a normal (non-reasoning) completion through the standard cascade.
        content = self.ask(prompt)
        return {"provider": "fallback", "model": "standard", "content": content,
                "reasoning": None, "errors": errors}

    def _ask_specific(self, prompt: str, provider: str) -> str:
        if provider == "groq":
            return self.ask_groq(prompt)
        if provider == "nvidia":
            return self.ask_nvidia(prompt)
        if provider == "gemini":
            return self.ask_gemini(prompt)
        raise ValueError(f"Unknown provider '{provider}'")

    def ask(self, prompt: str, provider: Optional[str] = None, fallback: bool = True, task: Optional[str] = None) -> str:
        ev = getattr(self, "cancel_event", None)              # v26: a cancelled job stops before its next AI call
        if ev is not None and ev.is_set():
            from core_utils import JobCancelled
            raise JobCancelled("cancelled by the user")
        real = any(getattr(self, a, None) is not None for a in ("groq", "gemini", "nvidia"))   # only real AI clients are cached
        cached = ai_cache_get(prompt, task) if real else None
        if cached is not None:              # the same reading prompt was answered before (re-run, repair round, quick->full)
            self.health.usage["cache_hits"] = self.health.usage.get("cache_hits", 0) + 1
            return cached
        _bump(self.health, "requests")
        try:
            out = self._ask_uncached(prompt, provider, fallback, task)
        except Exception as exc:
            _bump(self.health, "requests_failed")
            self._outcome(False, exc)
            raise
        self._outcome(True)
        if real:
            ai_cache_put(prompt, task, out)
        return out

    def _outcome(self, ok: bool, exc: Optional[BaseException] = None) -> None:
        """Sliding window of request outcomes; stop the run when most recent requests failed everywhere."""
        win = self.__dict__.setdefault("_recent", collections.deque(maxlen=OUTAGE_WINDOW))
        if not ok and exc is not None:
            from provider_health import classify_error, is_daily_limit
            t = str(exc)
            if (classify_error(t) == "rate_limit" and not is_daily_limit(t)) or "every model is resting" in t:
                return          # v27: a per-minute rate limit slows the run down; it is never a reason to stop it
        win.append(ok)
        if len(win) >= OUTAGE_WINDOW and win.count(False) >= OUTAGE_SHARE * len(win) and not os.getenv("BIOSYN_NO_OUTAGE_STOP") \
                and not os.getenv("BIOSYN_OFFLINE"):            # offline tests run on purpose without working AI
            top = self.health.error_summary(3) if hasattr(self.health, "error_summary") else []
            why = "; ".join(f"{e['provider']} {e['kind'].replace('_', ' ')} x{e['count']}: {e['example'][:140]}" for e in top)
            raise AIOutage(f"stopped early: {win.count(False)} of the last {len(win)} AI requests failed on every "
                           f"provider. Most common errors - {why or 'see the log'}. Check the keys on the API keys "
                           f"page (Test keys), then run again.")

    def preflight(self, timeout_s: float = 25.0) -> Dict[str, str]:
        """v26.2: one tiny request per configured provider BEFORE the run. A key that is rejected, or a provider
        that cannot answer, is switched off for this run with the reason, instead of failing on every paper.
        Returns {provider: 'ok' | reason}. Raises RuntimeError when no provider works."""
        res: Dict[str, str] = {}
        probe = "Reply with the single word OK."
        for name in self.configured_providers():
            try:
                if name == "groq":
                    c = self.groq.with_options(timeout=timeout_s, max_retries=0)
                    last = None
                    from provider_health import is_daily_limit
                    gskip = self.__dict__.setdefault("_groq_skip", {})
                    for m in GROQ_MODELS:
                        try:
                            c.chat.completions.create(model=m, messages=[{"role": "user", "content": probe}], max_tokens=5)
                            last = None
                            break
                        except Exception as e:  # a model this key cannot use: try the next
                            last = e
                            if _is_rate_limit(str(e)) and is_daily_limit(str(e)):
                                gskip[m] = 0    # today's allowance on this model is used up: skipped all run
                                continue
                            if _is_rate_limit(str(e)):
                                last = None     # rate-limited (per minute) means the key WORKS
                                break
                            if any(k in str(e).lower() for k in ("404", "model_not_found", "decommission")):
                                gskip[m] = 0
                                continue
                            break
                    if last is not None:
                        raise last
                elif name == "nvidia":
                    self.nvidia.with_options(timeout=timeout_s, max_retries=0).chat.completions.create(
                        model=NVIDIA_MODEL, messages=[{"role": "user", "content": probe}], max_tokens=5)
                elif name == "gemini":
                    last = None
                    for m in GEMINI_MODELS:
                        try:
                            self.gemini.models.generate_content(model=m, contents=probe)
                            last = None
                            break
                        except Exception as e:
                            last = e
                            if _is_rate_limit(str(e)):
                                last = None
                                break
                    if last is not None:
                        raise last
                res[name] = "ok"
            except Exception as exc:
                from provider_health import redact_keys
                msg = redact_keys(f"{type(exc).__name__}: {exc}")[:240]
                res[name] = msg
                getattr(self.health, "note_error", lambda *a: None)(name, msg)
                self.health.disable(name)
        if res and not any(v == "ok" for v in res.values()):
            raise RuntimeError("No AI provider answered a test request, so the run was not started: "
                               + " | ".join(f"{k}: {v}" for k, v in res.items()))
        return res

    def _ask_uncached(self, prompt: str, provider: Optional[str] = None, fallback: bool = True,
                      task: Optional[str] = None) -> str:
        candidates = [provider] if provider else self.order_for(task)
        if fallback and provider:
            candidates.extend(p for p in self.order_for(task) if p != provider)

        if not candidates:
            self.health.enable("groq")
            candidates = ["groq"]

        from ai_pacing import cool_down, cooling, note_switch, retry_after_seconds, wait_turn
        errors = []
        usable = [c for c in candidates if self.health.is_available(c)]
        # a provider that just refused us is tried LAST, not skipped for ever: order by how long it is still cooling
        usable.sort(key=lambda c: cooling(c))
        for i, candidate in enumerate(usable):
            if cooling(candidate) > 0 and i < len(usable) - 1:
                note_switch()
                continue                  # another provider is ready now - go there instead of waiting
            if not wait_turn(candidate):
                note_switch()
                continue                  # its queue is long; another provider will be quicker
            try:
                return self._ask_specific(prompt, candidate)
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")
                if _is_rate_limit(str(exc)):
                    # do not sleep here and retry the same exhausted key: stand it down and use the next provider
                    # (counted once, by the provider call that failed)
                    cool_down(candidate, retry_after_seconds(str(exc)), str(exc))
                    _bump(self.health, "retries")
                if provider and not fallback:
                    raise

        # every provider is cooling / rate-limited: keep waiting for the first to free up (v27: up to
        # BIOSYN_RATE_WAIT_MAX_S in total) - a rate limit slows the run, it does not fail the request
        deadline = time.time() + RATE_WAIT_MAX_S
        from provider_health import classify_error, is_daily_limit
        def transient(e: str) -> bool:
            if (classify_error(e) == "rate_limit" and not is_daily_limit(e)) or "resting" in e:
                return True
            prov = e.split(":", 1)[0].split(" ")[0]
            st = getattr(self.health, "_stats", {}).get(prov)
            return "unavailable" in e.lower() and st is not None and st.last_error_class == "rate_limit"
        while usable and not os.getenv("BIOSYN_NO_PACING") and errors and all(transient(e) for e in errors[-len(usable):]) \
                and time.time() < deadline:
            soonest = min(usable, key=lambda c: cooling(c))
            wait = min(max(cooling(soonest), 2.0), 30.0, max(0.0, deadline - time.time()))
            if not self._sleep_cancellable(wait, RATE_WAIT_MAX_S):
                break
            try:
                return self._ask_specific(prompt, soonest)
            except Exception as exc:
                errors.append(f"{soonest} (after waiting {wait:.0f}s): {exc}")
                if transient(str(exc)):
                    cool_down(soonest, retry_after_seconds(str(exc)), str(exc))
        raise RuntimeError(f"All AI providers failed: {' | '.join(errors)}")

    def ask_parallel(
        self,
        prompts: List[str],
        max_workers: int = 3,
        provider: Optional[str] = None,
    ) -> List[Optional[str]]:
        results: List[Optional[str]] = [None] * len(prompts)
        # several keys: spread the prompts round-robin over the providers (separate rate limits = faster),
        # every prompt still falls back to the other providers
        spread = [provider] if provider else (self.order_for("extraction") or [None])
        if not provider and len(spread) > 1:
            max_workers = max(max_workers, min(len(prompts), 3 * len(spread)))

        def _worker(idx: int, p: str):
            try:
                return idx, self.ask(p, provider=spread[idx % len(spread)], fallback=True, task="extraction")
            except Exception:
                return idx, None

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            futures = [executor.submit(_worker, i, p) for i, p in enumerate(prompts)]
            for future in concurrent.futures.as_completed(futures):
                idx, res = future.result()
                results[idx] = res

        return results

# ------------------------------------------------------------------ AI answer cache (reading tasks only)
# Claim extraction and screening prompts contain the paper text and the question; the same prompt asked again (the
# same question re-run, a repair round, quick scan then full dossier) gets the stored answer instead of a new AI call.
# Synthesis, hypotheses and critique are never cached (they should be re-thought). Answers are kept 7 days.
AI_CACHE_TASKS = {"extraction", "screening", "extraction_check"}
AI_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "ai_answers")


def _ai_cache_dir() -> str:
    """v27: follows BIOSYN_CACHE_DIR like every other cache (tests and moved installs never write into the program
    folder)."""
    d = os.getenv("BIOSYN_CACHE_DIR")
    return os.path.join(d, "ai_answers") if d else AI_CACHE_DIR
AI_CACHE_MAX_AGE = 7 * 24 * 3600


def _ai_cache_path(prompt: str, task: Optional[str]) -> Optional[str]:
    if task not in AI_CACHE_TASKS or os.getenv("BIOSYN_OFFLINE") or os.getenv("BIOSYN_NO_AI_CACHE"):
        return None
    import hashlib
    return os.path.join(_ai_cache_dir(), hashlib.sha256(f"{task}\n{prompt}".encode("utf-8")).hexdigest() + ".txt")


def ai_cache_get(prompt: str, task: Optional[str]) -> Optional[str]:
    path = _ai_cache_path(prompt, task)
    try:
        if path and os.path.exists(path) and time.time() - os.path.getmtime(path) <= AI_CACHE_MAX_AGE:
            with open(path, encoding="utf-8") as fh:
                txt = fh.read()
            return txt if txt.strip() else None
    except OSError:
        pass
    return None


def ai_cache_put(prompt: str, task: Optional[str], answer: Optional[str]) -> None:
    path = _ai_cache_path(prompt, task)
    if not path or not (answer or "").strip():
        return
    try:
        os.makedirs(_ai_cache_dir(), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(answer)
        os.replace(tmp, path)
    except OSError:
        pass                                   # a read-only folder: run without the cache


def _split_think(text: str):
    """Separate inline <think>...</think> blocks (DeepSeek-R1 style) from the answer."""
    thoughts = _THINK_RE.findall(text or "")
    answer = _THINK_RE.sub("", text or "").strip()
    return answer, ("\n".join(t.strip() for t in thoughts) or None)
