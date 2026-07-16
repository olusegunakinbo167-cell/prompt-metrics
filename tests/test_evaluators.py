# tests/test_evaluators.py
"""
Tests for prompt_metrics.evaluators.

Covers:
  - Text evaluators: ExactMatch, Keyword, Regex, Contains
  - Model-based evaluators: QAEvaluator, CritiqueEvaluator (with mocked LLM)
"""

from unittest.mock import Mock

import pytest

from prompt_metrics.evaluators import (
    ContainsEvaluator,
    CritiqueEvaluator,
    ExactMatchEvaluator,
    KeywordEvaluator,
    QAEvaluator,
    RegexMatchEvaluator,
)


# ---------------------------------------------------------------------------
# Text evaluators
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Model-based evaluators (with mocked LLM)
# ---------------------------------------------------------------------------

def test_qa_evaluator_parses_json_score():
    """QAEvaluator should parse JSON judge output and return 1-5 score."""
    # Mock judge that returns a clean JSON response
    mock_judge = Mock(return_value='{"score": 4, "reasoning": "Mostly correct, minor omission."}')

    ev = QAEvaluator(model_client=mock_judge)
    result = ev.evaluate(
        prompt="What is the capital of France?",
        response="Paris is the capital city of France.",
        expected_text="Paris",
    )

    # Judge was called once with a prompt containing the question/response/reference
    mock_judge.assert_called_once()
    judge_prompt = mock_judge.call_args[0][0]
    assert "What is the capital of France?" in judge_prompt
    assert "Paris is the capital city of France." in judge_prompt
    assert "Paris" in judge_prompt

    # Score is parsed correctly
    assert result["score"] == 4.0
    assert result["score_norm"] == pytest.approx(0.75)  # (4-1)/4
    assert "Mostly correct" in result["reasoning"]
    assert result["raw"] == '{"score": 4, "reasoning": "Mostly correct, minor omission."}'


def test_qa_evaluator_parses_embedded_json():
    """QAEvaluator should find JSON embedded in free-form judge text."""
    mock_judge = Mock(
        return_value='Here is my evaluation:\n{"score": 2, "reasoning": "The answer is wrong."}\nHope that helps!'
    )
    ev = QAEvaluator(model_client=mock_judge)
    result = ev.evaluate(
        prompt="2 + 2 = ?",
        response="5",
        expected_text="4",
    )
    assert result["score"] == 2.0
    assert result["score_norm"] == pytest.approx(0.25)
    assert "wrong" in result["reasoning"].lower()


def test_qa_evaluator_fallback_regex_parsing():
    """QAEvaluator should fall back to regex when judge doesn't return JSON."""
    # Judge returns unstructured text with "score: 5"
    mock_judge = Mock(
        return_value="score: 5\nreasoning: Perfect answer, fully correct."
    )
    ev = QAEvaluator(model_client=mock_judge)
    result = ev.evaluate(
        prompt="Q?", response="A", expected_text="A"
    )
    assert result["score"] == 5.0
    assert result["score_norm"] == pytest.approx(1.0)
    assert "Perfect answer" in result["reasoning"]


def test_qa_evaluator_clamps_score_range():
    """Scores outside 1-5 should be clamped."""
    mock_judge = Mock(return_value='{"score": 99, "reasoning": "way too high"}')
    ev = QAEvaluator(model_client=mock_judge)
    result = ev.evaluate(prompt="Q", response="A", expected_text="A")
    assert result["score"] == 5.0  # clamped
    assert result["score_norm"] == 1.0


def test_qa_evaluator_missing_reference():
    """QAEvaluator returns score=None when expected_text is missing."""
    mock_judge = Mock()
    ev = QAEvaluator(model_client=mock_judge)
    result = ev.evaluate(prompt="Q?", response="A", expected_text=None)
    assert result["score"] is None
    assert "no reference answer" in result["reasoning"].lower()
    mock_judge.assert_not_called()  # judge should never be invoked


def test_qa_evaluator_handles_model_error():
    """Model client exceptions should be caught and reported."""
    def failing_judge(prompt: str) -> str:
        raise RuntimeError("API rate limit exceeded")

    ev = QAEvaluator(model_client=failing_judge)
    result = ev.evaluate(prompt="Q", response="A", expected_text="A")
    assert result["score"] is None
    assert "model_client error" in result["reasoning"]
    assert "rate limit" in result["reasoning"]


def test_critique_evaluator_pass_fail_json():
    """CritiqueEvaluator should parse pass/fail JSON output."""
    mock_judge = Mock(
        return_value='{"pass": true, "reasoning": "Response is clear and helpful."}'
    )
    ev = CritiqueEvaluator(model_client=mock_judge, rubric="helpfulness")
    result = ev.evaluate(
        prompt="Explain photosynthesis",
        response="Photosynthesis converts light into chemical energy in plants.",
    )

    mock_judge.assert_called_once()
    judge_prompt = mock_judge.call_args[0][0]
    assert "helpfulness" in judge_prompt.lower()
    assert "Explain photosynthesis" in judge_prompt

    assert result["score"] == 1.0
    assert result["passed"] is True
    assert "helpful" in result["reasoning"].lower()
    assert result["rubric"] == "helpfulness"


def test_critique_evaluator_fail_case():
    """CritiqueEvaluator should correctly parse a FAIL verdict."""
    mock_judge = Mock(
        return_value='{"pass": false, "reasoning": "Response is off-topic and confusing."}'
    )
    ev = CritiqueEvaluator(model_client=mock_judge, rubric="helpfulness")
    result = ev.evaluate(prompt="Q", response="nonsense")
    assert result["score"] == 0.0
    assert result["passed"] is False


def test_critique_evaluator_builtin_rubrics():
    """All built-in rubrics should be resolvable and inject into the judge prompt."""
    from prompt_metrics.evaluators.model import RUBRIC_TEMPLATES

    for rubric_name in RUBRIC_TEMPLATES.keys():
        mock_judge = Mock(return_value='{"pass": true, "reasoning": "ok"}')
        ev = CritiqueEvaluator(model_client=mock_judge, rubric=rubric_name)
        ev.evaluate(prompt="Q", response="A")
        # Verify the rubric text made it into the judge prompt
        judge_prompt = mock_judge.call_args[0][0]
        assert rubric_name in judge_prompt.lower()
        # And the rubric template text is present
        assert RUBRIC_TEMPLATES[rubric_name][:30] in judge_prompt


def test_critique_evaluator_custom_rubric():
    """CritiqueEvaluator should accept arbitrary custom rubric strings."""
    custom_rubric = "Is the response written in iambic pentameter?"
    mock_judge = Mock(return_value='{"pass": false, "reasoning": "Nope, free verse."}')
    ev = CritiqueEvaluator(model_client=mock_judge, rubric=custom_rubric)
    result = ev.evaluate(prompt="Q", response="A")
    assert result["rubric"] == custom_rubric
    # Custom rubric text should appear in the judge prompt
    judge_prompt = mock_judge.call_args[0][0]
    assert "iambic pentameter" in judge_prompt
    assert result["passed"] is False


def test_critique_evaluator_parses_freeform_pass_fail():
    """CritiqueEvaluator should handle non-JSON pass/fail responses."""
    # Judge returns plain text instead of JSON
    mock_judge = Mock(return_value="Verdict: PASS\nReasoning: Good job, clear and concise.")
    ev = CritiqueEvaluator(model_client=mock_judge)
    result = ev.evaluate(prompt="Q", response="A")
    assert result["score"] == 1.0
    assert result["passed"] is True


def test_model_evaluator_requires_model_client():
    """ModelEvaluator subclasses must be given a model_client."""
    with pytest.raises(ValueError, match="requires a model_client"):
        QAEvaluator(model_client=None)  # type: ignore

    with pytest.raises(TypeError, match="model_client must be callable"):
        CritiqueEvaluator(model_client="not a callable")  # type: ignore
