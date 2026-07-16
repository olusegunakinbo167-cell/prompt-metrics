# src/prompt_metrics/evaluators/text.py
"""
Text-based evaluators for prompt responses.

Includes exact matching, keyword overlap, regex matching, and substring checks.
"""

from __future__ import annotations

import re
from typing import Any


class ExactMatchEvaluator:
    """
    1.0 if expected_text matches the response exactly (whitespace-trimmed,
    case-insensitive), else 0.0. Returns None when expected_text is missing.
    """

    name = "exact_match"

    def evaluate(
        self,
        prompt: str,
        response: str,
        *,
        expected_text: str | None = None,
        keywords: list[str] | None = None,
        case_id: str | None = None,
    ) -> dict[str, Any]:
        if expected_text is None:
            return {"score": None, "passed": None, "reason": "no expected_text"}
        passed = response.strip().lower() == expected_text.strip().lower()
        return {"score": 1.0 if passed else 0.0, "passed": passed}


class KeywordEvaluator:
    """
    Fraction of supplied keywords found in the response (case-insensitive).
    Returns score=None when no keywords are provided.
    """

    name = "keyword"

    def evaluate(
        self,
        prompt: str,
        response: str,
        *,
        expected_text: str | None = None,
        keywords: list[str] | None = None,
        case_id: str | None = None,
    ) -> dict[str, Any]:
        keywords = keywords or []
        if not keywords:
            return {"score": None, "matched": [], "missing": []}

        response_lower = response.lower()
        matched = [kw for kw in keywords if kw.lower() in response_lower]
        missing = [kw for kw in keywords if kw.lower() not in response_lower]

        return {
            "score": round(len(matched) / len(keywords), 4),
            "matched_count": len(matched),
            "total_count": len(keywords),
            "matched": matched,
            "missing": missing,
        }


class RegexMatchEvaluator:
    """
    Returns 1.0 if a regex pattern matches the response, 0.0 if not.

    The pattern can be supplied in one of three ways (in order of precedence):
      1. expected_text — if it is a valid regex pattern
      2. regex_pattern in the case metadata (via case_id lookup — not yet wired)
      3. pattern passed to __init__

    For CLI usage, the pattern is typically passed via expected_text.
    For programmatic usage, pass pattern=... to the constructor.

    Matching is case-insensitive by default (re.IGNORECASE). Set
    case_sensitive=True to disable.
    """

    name = "regex_match"

    def __init__(
        self,
        pattern: str | None = None,
        *,
        case_sensitive: bool = False,
        multiline: bool = False,
        dotall: bool = False,
    ):
        """
        Args:
            pattern: Default regex pattern to match against. If None,
                the evaluator will try to use expected_text as the pattern.
            case_sensitive: If False (default), matching is case-insensitive.
            multiline: If True, ^ and $ match start/end of each line (re.MULTILINE).
            dotall: If True, '.' matches newlines (re.DOTALL).
        """
        self._default_pattern = pattern
        self._case_sensitive = case_sensitive
        self._multiline = multiline
        self._dotall = dotall

        flags = 0
        if not case_sensitive:
            flags |= re.IGNORECASE
        if multiline:
            flags |= re.MULTILINE
        if dotall:
            flags |= re.DOTALL
        self._flags = flags

    def evaluate(
        self,
        prompt: str,
        response: str,
        *,
        expected_text: str | None = None,
        keywords: list[str] | None = None,
        case_id: str | None = None,
    ) -> dict[str, Any]:
        pattern = self._default_pattern or expected_text
        if pattern is None:
            return {
                "score": None,
                "passed": None,
                "reason": "no pattern (expected_text or constructor pattern)",
            }

        try:
            compiled = re.compile(pattern, self._flags)
        except re.error as e:
            return {
                "score": None,
                "passed": None,
                "reason": f"invalid regex pattern: {e}",
                "pattern": pattern,
            }

        match = compiled.search(response)
        passed = match is not None

        result: dict[str, Any] = {
            "score": 1.0 if passed else 0.0,
            "passed": passed,
            "pattern": pattern,
        }
        if passed and match:
            result["match"] = match.group(0)
            if match.groups():
                result["groups"] = match.groups()

        return result


# ---------------------------------------------------------------------------
# Additional text evaluators (retained for backwards compatibility)
# ---------------------------------------------------------------------------


class ContainsEvaluator:
    """
    1.0 if expected_text is a substring of the response (case-insensitive),
    else 0.0. Useful as a looser alternative to exact_match.
    """

    name = "contains"

    def evaluate(
        self,
        prompt: str,
        response: str,
        *,
        expected_text: str | None = None,
        keywords: list[str] | None = None,
        case_id: str | None = None,
    ) -> dict[str, Any]:
        if expected_text is None:
            return {"score": None, "passed": None, "reason": "no expected_text"}
        passed = expected_text.lower() in response.lower()
        return {"score": 1.0 if passed else 0.0, "passed": passed}
