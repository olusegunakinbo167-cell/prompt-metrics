# tests/test_evaluators.py
"""
Tests for prompt_metrics.evaluators.

Covers:
  - Text evaluators: ExactMatch, Keyword, Regex, Contains
  - Semantic evaluator: SemanticSimilarity (TF-IDF + mocked embeddings)
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
    SemanticSimilarityEvaluator,
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
# Semantic similarity evaluator
# ---------------------------------------------------------------------------

def test_semantic_similarity_tfidf_identical():
    """TF-IDF fallback: identical texts should score ~1.0."""
    ev = SemanticSimilarityEvaluator()
    # Force TF-IDF backend even if sentence-transformers is installed
    ev._st_backend = None  # type: ignore
    ev._backend_name = "tfidf"

    r = ev.evaluate(
        prompt="",
        response="The quick brown fox jumps over the lazy dog",
        expected_text="The quick brown fox jumps over the lazy dog",
    )
    assert r["score"] is not None
    assert r["score"] > 0.99
    assert r["backend"] == "tfidf"


def test_semantic_similarity_tfidf_paraphrase():
    """TF-IDF fallback: paraphrased text should have moderate similarity."""
    ev = SemanticSimilarityEvaluator()
    ev._st_backend = None  # type: ignore
    ev._backend_name = "tfidf"

    r = ev.evaluate(
        prompt="",
        response="Machine learning is a subset of artificial intelligence",
        expected_text="ML is a branch of AI",
    )
    # TF-IDF is a bag-of-words model, so with no overlapping tokens
    # after tokenization, similarity will be 0. These are quite different
    # lexically. Let's use texts with actual overlap:
    assert r["score"] is not None
    assert 0.0 <= r["score"] <= 1.0

    # Now test with overlapping vocabulary
    r2 = ev.evaluate(
        prompt="",
        response="machine learning is a subset of artificial intelligence",
        expected_text="machine learning is a branch of artificial intelligence",
    )
    # "machine", "learning", "is", "a", "of", "artificial", "intelligence" overlap
    # should give a decent score
    assert r2["score"] is not None
    assert r2["score"] > 0.3
    assert r2["backend"] == "tfidf"


def test_semantic_similarity_tfidf_unrelated():
    """TF-IDF fallback: unrelated texts should score near 0."""
    ev = SemanticSimilarityEvaluator()
    ev._st_backend = None  # type: ignore
    ev._backend_name = "tfidf"

    r = ev.evaluate(
        prompt="",
        response="purple elephants dance on mars",
        expected_text="quantum physics and thermodynamics",
    )
    assert r["score"] is not None
    assert r["score"] < 0.1


def test_semantic_similarity_missing_reference():
    """SemanticSimilarityEvaluator returns score=None when expected_text is missing."""
    ev = SemanticSimilarityEvaluator()
    ev._st_backend = None  # type: ignore
    ev._backend_name = "tfidf"

    r = ev.evaluate(prompt="Q?", response="A", expected_text=None)
    assert r["score"] is None
    assert "no expected_text" in r["reason"].lower()


def test_semantic_similarity_with_embedding_client():
    """SemanticSimilarityEvaluator with a mocked embedding API client."""
    # Mock embedding client: returns a 3D vector based on simple heuristics
    # so we can test cosine similarity logic end-to-end
    def fake_embed(text: str) -> list[float]:
        # Very simple: count occurrences of "cat", "dog", "bird"
        t = text.lower()
        return [
            float(t.count("cat")),
            float(t.count("dog")),
            float(t.count("bird")),
        ]

    ev = SemanticSimilarityEvaluator(
        embedding_client=fake_embed,
        normalize=True,
    )

    # Identical semantic content → similarity = 1.0
    r = ev.evaluate(
        prompt="",
        response="cat cat dog",
        expected_text="cat cat dog",
    )
    assert r["score"] == pytest.approx(1.0, abs=1e-5)
    assert r["backend"] == "custom"
    assert r["embedding_dim"] == 3

    # Orthogonal vectors → similarity ≈ 0
    r = ev.evaluate(
        prompt="",
        response="cat cat cat",
        expected_text="dog dog dog",
    )
    assert r["score"] == pytest.approx(0.0, abs=1e-5)

    # Partial overlap
    r = ev.evaluate(
        prompt="",
        response="cat dog",
        expected_text="cat bird",
    )
    # cat·cat = 1*1, dog·bird = 0 → cos = 1 / (sqrt(2) * sqrt(2)) = 0.5
    assert r["score"] == pytest.approx(0.5, abs=1e-5)


def test_semantic_similarity_embedding_client_error():
    """Embedding client exceptions should be caught gracefully."""
    def failing_embed(text: str) -> list[float]:
        raise RuntimeError("embedding API quota exceeded")

    ev = SemanticSimilarityEvaluator(embedding_client=failing_embed)
    r = ev.evaluate(
        prompt="",
        response="hello",
        expected_text="world",
    )
    assert r["score"] is None
    assert "embedding failed" in r["error"].lower()
    assert "quota" in r["error"].lower()


def test_semantic_similarity_dimension_mismatch():
    """Mismatched embedding dimensions should be reported cleanly."""
    call_count = [0]

    def bad_embed(text: str) -> list[float]:
        # Return different dimensions on first vs second call
        call_count[0] += 1
        if call_count[0] == 1:
            return [0.1, 0.2, 0.3]
        return [0.1, 0.2]  # wrong dim!

    ev = SemanticSimilarityEvaluator(embedding_client=bad_embed)
    r = ev.evaluate(
        prompt="",
        response="a",
        expected_text="b",
    )
    assert r["score"] is None
    assert "dimension mismatch" in r["error"].lower()


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
