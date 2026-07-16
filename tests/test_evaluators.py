# tests/test_evaluators.py
"""
Tests for prompt_metrics.evaluators text evaluators.
"""

from prompt_metrics.evaluators import (
    ExactMatchEvaluator,
    KeywordEvaluator,
    RegexMatchEvaluator,
    ContainsEvaluator,
)


def test_exact_match_case_insensitive():
    ev = ExactMatchEvaluator()
    r = ev.evaluate("", "Hello World", expected_text="hello world")
    assert r["score"] == 1.0
    assert r["passed"] is True

    r = ev.evaluate("", "foo", expected_text="bar")
    assert r["score"] == 0.0
    assert r["passed"] is False


def test_keyword_overlap():
    ev = KeywordEvaluator()
    r = ev.evaluate("", "The quick brown fox", keywords=["quick", "fox", "cat"])
    assert r["score"] == 0.6667
    assert set(r["matched"]) == {"quick", "fox"}
    assert r["missing"] == ["cat"]


def test_regex_match():
    ev = RegexMatchEvaluator()
    r = ev.evaluate("", "Order #12345 confirmed", expected_text=r"#\d+")
    assert r["score"] == 1.0
    assert r["passed"] is True
    assert "#12345" in r["match"]

    r = ev.evaluate("", "no numbers here", expected_text=r"\d+")
    assert r["score"] == 0.0
    assert r["passed"] is False


def test_contains_evaluator():
    ev = ContainsEvaluator()
    r = ev.evaluate("", "The quick brown fox", expected_text="BROWN")
    assert r["score"] == 1.0
    assert r["passed"] is True
