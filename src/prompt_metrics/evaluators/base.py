# src/prompt_metrics/evaluators/base.py
"""
Base evaluator protocol and adapter utilities.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class Evaluator(Protocol):
    """
    Protocol that all evaluators must satisfy.

    Evaluators must expose a `name` attribute and an `evaluate` method.
    """

    name: str

    def evaluate(
        self,
        prompt: str,
        response: str,
        *,
        expected_text: str | None = None,
        keywords: list[str] | None = None,
        case_id: str | None = None,
    ) -> dict[str, Any]: ...


class EvaluatorAdapter:
    """
    Wraps evaluators with non-standard interfaces so they conform to the
    Evaluator protocol.
    """

    def __init__(
        self,
        evaluator: Any,
        name: str | None = None,
        score_fn: Callable[[Any, str, str], Any] | None = None,
    ):
        self._evaluator = evaluator
        self.name = (
            name
            or getattr(evaluator, "rubric_id", None)
            or evaluator.__class__.__name__
        )

        # Auto-detect scoring method
        if score_fn is not None:
            self._score_fn = score_fn
        elif hasattr(evaluator, "evaluate"):
            self._score_fn = lambda ev, p, r: ev.evaluate(p, r)
        elif hasattr(evaluator, "score"):
            # RubricEvaluator-compatible
            self._score_fn = lambda ev, p, r: ev.score(prompt=p, response=r)
        else:
            raise TypeError(
                f"Evaluator {evaluator!r} has neither .evaluate() nor .score() — "
                "pass an explicit score_fn"
            )

    def evaluate(
        self,
        prompt: str,
        response: str,
        *,
        expected_text: str | None = None,
        keywords: list[str] | None = None,
        case_id: str | None = None,
    ) -> dict[str, Any]:
        """Call the wrapped evaluator and normalise the result to a dict."""
        result = self._score_fn(self._evaluator, prompt, response)

        # Normalise ScoreResult / dataclass / raw dict → dict
        if hasattr(result, "to_dict"):
            result = result.to_dict()
        elif hasattr(result, "__dataclass_fields__"):
            result = asdict(result)

        if isinstance(result, dict):
            return result
        return {"score": result, "raw": result}
