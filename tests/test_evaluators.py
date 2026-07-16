"""Unit tests for the core evaluation engine."""

import pytest

from src.prompt_metrics.evaluators import ExactMatchEvaluator, KeywordEvaluator


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
