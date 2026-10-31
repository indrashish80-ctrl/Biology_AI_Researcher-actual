"""
llm_json.py
-----------
Small, shared JSON-output adapter for the research pipeline.

The scientific extraction modules use this module so that:
- the AI router remains responsible for provider selection/fallback;
- models are asked for JSON only;
- markdown fences / surrounding text are stripped safely;
- a caller-supplied validator can reject malformed structures;
- provider attempts can be logged without making AI output evidence.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, List, Optional


class LLMJSONError(RuntimeError):
    """Raised when the LLM response cannot be converted into valid/accepted JSON."""


def _extract_json_text(text: str) -> str:
    """Extract the most likely JSON object/array from an LLM response."""
    if not isinstance(text, str) or not text.strip():
        raise LLMJSONError("LLM returned an empty response.")

    s = text.strip()

    # Remove common markdown fences.
    s = re.sub(r"^\s*```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s).strip()

    # First try the whole response.
    try:
        json.loads(s)
        return s
    except json.JSONDecodeError:
        pass

    # Models sometimes prepend a short sentence. Find the first balanced
    # JSON object/array while respecting quoted strings.
    starts = []
    for i, ch in enumerate(s):
        if ch in "{[":
            starts.append(i)

    for start in starts:
        opener = s[start]
        closer = "}" if opener == "{" else "]"
        depth = 0
        in_string = False
        escaped = False

        for i in range(start, len(s)):
            ch = s[i]

            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    candidate = s[start:i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        break

    raise LLMJSONError("Could not find a valid JSON object/array in the LLM response.")


def parse_json_response(text: str) -> Any:
    """Parse an LLM response as JSON."""
    try:
        return json.loads(_extract_json_text(text))
    except LLMJSONError:
        raise
    except Exception as exc:
        raise LLMJSONError(f"JSON parsing failed: {exc}") from exc


def ask_json(
    router: Any,
    prompt: str,
    validate: Optional[Callable[[Any], Any]] = None,
    providers: Optional[List[str]] = None,
    log: Optional[List[dict]] = None,
    max_attempts: Optional[int] = None,
) -> Any:
    """
    Ask the shared AIRouter for JSON.

    Parameters
    ----------
    router:
        AIRouter instance.
    prompt:
        Prompt requesting JSON.
    validate:
        Optional validator. It may return a normalized value or raise.
    providers:
        Optional ordered provider list. If supplied, providers are tried in
        that order. Otherwise AIRouter automatic routing/fallback is used.
    log:
        Optional list receiving attempt metadata.

    Returns
    -------
    Any
        Parsed JSON, or the validator's normalized result.

    Raises
    ------
    LLMJSONError
        If all attempts fail or validation fails.
    """
    if router is None:
        raise LLMJSONError("No AI router was supplied.")

    attempts = list(providers) if providers else [None]
    bounded = max_attempts is not None
    if bounded:
        # bounded: each listed provider is asked ONCE (no router-internal fall-through), at most max_attempts calls,
        # then the caller's validated fallback takes over
        attempts = attempts[:max(1, max_attempts)]
    errors = []

    for provider in attempts:
        try:
            if provider is None:
                raw = router.ask(prompt)
                used_provider = "automatic"
            else:
                raw = router.ask(prompt, provider=provider, fallback=not bounded)
                used_provider = provider

            parsed = parse_json_response(raw)

            if validate is not None:
                parsed = validate(parsed)

            if log is not None:
                log.append({
                    "provider": used_provider,
                    "status": "success",
                })

            return parsed

        except Exception as exc:
            msg = str(exc)
            errors.append(msg)

            if log is not None:
                log.append({
                    "provider": provider or "automatic",
                    "status": "failed",
                    "error": msg,
                })

    joined = " | ".join(errors[-5:])
    raise LLMJSONError(f"All JSON LLM attempts failed: {joined}")


__all__ = [
    "LLMJSONError",
    "ask_json",
    "parse_json_response",
]
