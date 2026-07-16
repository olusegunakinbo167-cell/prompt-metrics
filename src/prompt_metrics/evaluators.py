"""Core evaluation engine for scoring LLM responses."""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional


class BaseEvaluator(abc.ABC):
    """Abstract base for all response evaluators.

    Concrete evaluators must implement :meth:`evaluate`, which returns a
    consistent result dictionary.
    """

    name: str = "base"

    @abc.abstractmethod
    def evaluate(
        self,
        response_text: str,
        expected_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Score a model response against an expected value.

        Args:
            response_text: The LLM-generated response to evaluate.
            expected_text: The ground-truth / reference text. May be None
                for evaluators that don't require a reference.

        Returns:
            A dictionary with at minimum:
                {
                    "score_raw": <float | int | bool>,
                    "score_norm": <float>,       # 0.0 – 1.0
                    "metadata": <dict>,
                }
        """
        raise NotImplementedError

    @staticmethod
    def _norm_result(
        score_raw: float,
        score_norm: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Helper to build a consistently shaped result dict."""
        # Clamp normalized score to [0, 1]
        score_norm = max(0.0, min(1.0, float(score_norm)))
        return {
            "score_raw": score_raw,
            "score_norm": score_norm,
            "metadata": metadata or {},
        }


# ---------------------------------------------------------------------------
# Concrete evaluators
# ---------------------------------------------------------------------------

class ExactMatchEvaluator(BaseEvaluator):
    """Case-insensitive exact string match evaluator.

    Whitespace at the start/end of both strings is stripped before comparison.
    """

    name = "exact_match"

    def __init__(self, case_sensitive: bool = False) -> None:
        self.case_sensitive = case_sensitive

    def evaluate(
        self,
        response_text: str,
        expected_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        if expected_text is None:
            raise ValueError("ExactMatchEvaluator requires expected_text.")

        resp = response_text.strip()
        exp = expected_text.strip()

        if not self.case_sensitive:
            resp = resp.lower()
            exp = exp.lower()

        matched = resp == exp
        score_raw = 1 if matched else 0

        return self._norm_result(
            score_raw=score_raw,
            score_norm=float(score_raw),
            metadata={
                "evaluator": self.name,
                "case_sensitive": self.case_sensitive,
                "match": matched,
            },
        )


class KeywordEvaluator(BaseEvaluator):
    """Scores responses by the fraction of required keywords present.

    Each keyword is matched case-insensitively as a substring.
    The raw score is the count of keywords found; the normalized score
    is found / total.
    """

    name = "keyword"

    def __init__(
        self,
        keywords: List[str],
        case_sensitive: bool = False,
    ) -> None:
        if not keywords:
            raise ValueError("KeywordEvaluator requires at least one keyword.")
        self.keywords = keywords
        self.case_sensitive = case_sensitive

    def evaluate(
        self,
        response_text: str,
        expected_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        # expected_text is ignored – keywords are fixed at construction
        haystack = response_text if self.case_sensitive else response_text.lower()

        found: List[str] = []
        missing: List[str] = []

        for kw in self.keywords:
            needle = kw if self.case_sensitive else kw.lower()
            if needle in haystack:
                found.append(kw)
            else:
                missing.append(kw)

        total = len(self.keywords)
        found_count = len(found)
        score_norm = found_count / total if total else 0.0

        return self._norm_result(
            score_raw=found_count,
            score_norm=score_norm,
            metadata={
                "evaluator": self.name,
                "keywords_total": total,
                "keywords_found": found,
                "keywords_missing": missing,
                "case_sensitive": self.case_sensitive,
            },
        )
