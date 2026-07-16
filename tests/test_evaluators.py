"""Unit tests for the core evaluation engine."""

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.prompt_metrics.evaluators import (
    ExactMatchEvaluator,
    KeywordEvaluator,
    JSONLLMEvaluator,
    _JudgeScore,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_judge_eval(monkeypatch, score: int, reasoning: str, confidence: float):
    """Patch JSONLLMEvaluator._openai_client to return a canned _JudgeScore."""
    parsed = _JudgeScore(score=score, reasoning=reasoning, confidence=confidence)
    mock_resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))]
    )
    mock_client = MagicMock()
    mock_client.beta.chat.completions.parse.return_value = mock_resp
    monkeypatch.setattr(
        JSONLLMEvaluator,
        "_openai_client",
        property(lambda self: mock_client),
    )


# ---------------------------------------------------------------------------
# ExactMatchEvaluator
# ---------------------------------------------------------------------------

def test_exact_match_perfect():
    """Exact match succeeds on identical strings."""
    ev = ExactMatchEvaluator()
    result = ev.evaluate("hello world", "hello world")
    assert result["score_raw"] == 1
    assert result["score_norm"] == 1.0
    assert result["metadata"]["match"] is True


def test_exact_match_whitespace_stripped():
    """Leading/trailing whitespace is stripped before comparison."""
    ev = ExactMatchEvaluator()
    result = ev.evaluate("  foo bar  \n", "\tfoo bar\n")
    assert result["score_raw"] == 1
    assert result["score_norm"] == 1.0
    assert result["metadata"]["match"] is True


def test_exact_match_case_insensitive_default():
    """Case-insensitive matching is the default."""
    ev = ExactMatchEvaluator()
    result = ev.evaluate("Hello WORLD", "hello world")
    assert result["score_raw"] == 1
    assert result["score_norm"] == 1.0
    assert result["metadata"]["case_sensitive"] is False


def test_exact_match_case_sensitive_flag():
    """Case-sensitive flag causes casing mismatches to fail."""
    ev = ExactMatchEvaluator(case_sensitive=True)
    result = ev.evaluate("Hello World", "hello world")
    assert result["score_raw"] == 0
    assert result["score_norm"] == 0.0
    assert result["metadata"]["match"] is False
    assert result["metadata"]["case_sensitive"] is True


def test_exact_match_requires_expected_text():
    """ExactMatchEvaluator raises if expected_text is missing."""
    ev = ExactMatchEvaluator()
    with pytest.raises(ValueError, match="requires expected_text"):
        ev.evaluate("anything")


def test_exact_match_mismatch():
    """Non-matching strings score 0."""
    ev = ExactMatchEvaluator()
    result = ev.evaluate("foo", "bar")
    assert result["score_raw"] == 0
    assert result["score_norm"] == 0.0
    assert result["metadata"]["match"] is False


# ---------------------------------------------------------------------------
# KeywordEvaluator
# ---------------------------------------------------------------------------

def test_keyword_all_found():
    """All keywords present → 1.0 score."""
    ev = KeywordEvaluator(keywords=["alpha", "beta", "gamma"])
    result = ev.evaluate("Alpha and BETA with gamma ray")
    assert result["score_raw"] == 3
    assert result["score_norm"] == pytest.approx(1.0)
    assert result["metadata"]["keywords_missing"] == []


def test_keyword_partial_fractional_score():
    """Partial keyword hits produce a correct fractional score."""
    ev = KeywordEvaluator(keywords=["Paris", "Eiffel Tower", "France", "Louvre"])
    result = ev.evaluate("Paris, France — home of the Eiffel Tower")
    # 3 / 4 found, Louvre missing
    assert result["score_raw"] == 3
    assert result["score_norm"] == pytest.approx(0.75)
    assert set(result["metadata"]["keywords_found"]) == {
        "Paris", "Eiffel Tower", "France"
    }
    assert result["metadata"]["keywords_missing"] == ["Louvre"]


def test_keyword_none_found():
    """Zero keywords found → 0.0 score."""
    ev = KeywordEvaluator(keywords=["foo", "bar"])
    result = ev.evaluate("nothing in here")
    assert result["score_raw"] == 0
    assert result["score_norm"] == 0.0
    assert set(result["metadata"]["keywords_missing"]) == {"foo", "bar"}


def test_keyword_case_sensitive_flag():
    """Case-sensitive mode only matches exact casing."""
    ev = KeywordEvaluator(keywords=["Paris"], case_sensitive=True)
    # wrong case → miss
    result = ev.evaluate("paris")
    assert result["score_raw"] == 0
    assert result["score_norm"] == 0.0
    # correct case → hit
    result = ev.evaluate("Paris")
    assert result["score_raw"] == 1
    assert result["score_norm"] == 1.0


def test_keyword_empty_list_raises():
    """Instantiating with an empty keyword list raises ValueError."""
    with pytest.raises(ValueError, match="at least one keyword"):
        KeywordEvaluator(keywords=[])


def test_keyword_substring_matching():
    """Keywords match as substrings (not whole-word)."""
    ev = KeywordEvaluator(keywords=["test"])
    result = ev.evaluate("this is a testing ground")
    assert result["score_raw"] == 1
    assert result["score_norm"] == 1.0


# ---------------------------------------------------------------------------
# JSONLLMEvaluator
# ---------------------------------------------------------------------------

def test_json_llm_success_max_score(monkeypatch):
    """Mocked judge returning score 5 normalizes to 1.0 with full metadata."""
    _mock_judge_eval(
        monkeypatch,
        score=5,
        reasoning="Perfect factual accuracy and excellent tone.",
        confidence=0.95,
    )

    rubric = "Rate accuracy and tone, 1-5."
    ev = JSONLLMEvaluator(rubric=rubric, model="gpt-4o-mini")

    result = ev.evaluate(
        response_text="Paris is the capital of France.",
        expected_text="The capital of France is Paris.",
    )

    # Score normalization: (5 - 1) / 4 = 1.0
    assert result["score_raw"] == 5
    assert result["score_norm"] == pytest.approx(1.0)

    # Metadata mapping
    md = result["metadata"]
    assert md["evaluator"] == "json_llm"
    assert md["reasoning"] == "Perfect factual accuracy and excellent tone."
    assert md["confidence"] == pytest.approx(0.95)
    assert md["judge_model"] == "gpt-4o-mini"
    assert md["rubric"] == rubric
    assert md["expected_text_provided"] is True


def test_json_llm_low_score_normalization(monkeypatch):
    """Mocked judge returning score 1 normalizes to 0.0 without errors."""
    _mock_judge_eval(
        monkeypatch,
        score=1,
        reasoning="Completely incorrect and unhelpful.",
        confidence=0.92,
    )

    ev = JSONLLMEvaluator(
        rubric="Rate helpfulness, 1 = worst, 5 = best.",
        model="gpt-4o-mini",
    )

    result = ev.evaluate(response_text="The moon is made of cheese.")

    # Score normalization: (1 - 1) / 4 = 0.0
    assert result["score_raw"] == 1
    assert result["score_norm"] == pytest.approx(0.0)
    # Clamping safety – never negative
    assert result["score_norm"] >= 0.0

    md = result["metadata"]
    assert md["confidence"] == pytest.approx(0.92)
    assert md["reasoning"] == "Completely incorrect and unhelpful."
    assert md["expected_text_provided"] is False
