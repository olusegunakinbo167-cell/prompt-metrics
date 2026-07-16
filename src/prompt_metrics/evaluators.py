"""Core evaluation engine for scoring LLM responses."""

from __future__ import annotations

import abc
import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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


# ---------------------------------------------------------------------------
# LLM-as-a-judge evaluators
# ---------------------------------------------------------------------------

class _JudgeScore(BaseModel):
    """Structured output schema for the LLM judge.

    Internal – not exported from the module public API.
    """

    score: int = Field(
        ...,
        ge=1,
        le=5,
        description="Overall score from 1 (worst) to 5 (best)",
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation justifying the score",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Judge's confidence in the score, 0.0–1.0",
    )


class JSONLLMEvaluator(BaseEvaluator):
    """LLM-as-a-judge evaluator using OpenAI Structured Outputs.

    Forces the judge model to return a Pydantic-validated JSON object with
    a 1–5 integer score, reasoning text, and confidence level. The raw score
    is the integer 1–5; the normalized score is (score - 1) / 4.

    Example:
        >>> judge = JSONLLMEvaluator(
        ...     rubric="Rate the helpfulness and accuracy on a 1-5 scale.",
        ...     model="gpt-4o-mini",
        ... )
        >>> result = judge.evaluate(
        ...     response_text="Paris is the capital of France.",
        ...     expected_text="The capital of France is Paris.",
        ... )
        >>> result["score_norm"]  # 0.0 – 1.0
        1.0
    """

    name = "json_llm"

    def __init__(
        self,
        rubric: str,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 2,
        system_prompt: Optional[str] = None,
    ) -> None:
        """Create a JSON LLM judge evaluator.

        Args:
            rubric: Freeform scoring criteria shown to the judge model.
                Be specific – e.g. "Score factual accuracy and tone on a
                1-5 scale. 1 = completely wrong / hostile, 5 = perfect."
            model: OpenAI model ID (must support Structured Outputs).
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env.
            temperature: Sampling temperature for the judge (default 0.0
                for deterministic scoring).
            max_retries: Number of retries on parse/API failure.
            system_prompt: Optional override for the system prompt.
                If None, a default LLM-as-a-judge prompt is used.
        """
        self.rubric = rubric
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_retries = max_retries
        self._custom_system_prompt = system_prompt
        self._client = None  # lazy-init

    @property
    def _openai_client(self):
        """Lazy-load the OpenAI client so import doesn't fail without openai."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "JSONLLMEvaluator requires the 'openai' package. "
                    "Install with: pip install openai"
                ) from exc
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def _build_judge_messages(
        self, response_text: str, expected_text: Optional[str]
    ) -> List[Dict[str, str]]:
        system = self._custom_system_prompt or (
            "You are an expert LLM response evaluator. "
            "Score the given response strictly according to the rubric. "
            "Return a structured JSON object with score, reasoning, and confidence. "
            "Be objective and consistent."
        )

        user_parts = [
            f"Rubric:\n{rubric}\n" if (rubric := self.rubric) else "",
            f"Response to evaluate:\n---\n{response_text}\n---",
        ]
        if expected_text:
            user_parts.append(
                f"Reference / expected output:\n---\n{expected_text}\n---"
            )
        user_parts.append(
            "\nProvide your score (1-5), reasoning, and confidence level "
            "as a structured JSON object."
        )

        # filter empty strings
        user_parts = [p for p in user_parts if p]

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]

    def evaluate(
        self,
        response_text: str,
        expected_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        client = self._openai_client
        messages = self._build_judge_messages(response_text, expected_text)

        # Structured Outputs – Pydantic model is enforced by the API
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = client.beta.chat.completions.parse(
                    model=self.model,
                    messages=messages,  # type: ignore[arg-type]
                    response_format=_JudgeScore,
                    temperature=self.temperature,
                )
                parsed: Optional[_JudgeScore] = resp.choices[0].message.parsed
                if parsed is None:
                    raise ValueError("Judge returned no parsed output.")
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == self.max_retries:
                    raise RuntimeError(
                        f"JSONLLMEvaluator failed after {self.max_retries + 1} "
                        f"attempt(s): {exc}"
                    ) from exc
                continue
        else:
            # Should be unreachable – loop either breaks or raises
            raise RuntimeError("Unreachable")

        # Normalize 1–5 → 0.0–1.0
        score_raw = int(parsed.score)
        score_norm = (score_raw - 1) / 4.0

        return self._norm_result(
            score_raw=score_raw,
            score_norm=score_norm,
            metadata={
                "evaluator": self.name,
                "reasoning": parsed.reasoning,
                "confidence": parsed.confidence,
                "judge_model": self.model,
                "rubric": self.rubric,
                "expected_text_provided": expected_text is not None,
            },
        )
