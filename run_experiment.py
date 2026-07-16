"""Quick ad-hoc runner for testing evaluators against sample data."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.prompt_metrics.evaluators import (
    ExactMatchEvaluator,
    KeywordEvaluator,
    JSONLLMEvaluator,
    _JudgeScore,
)


def _mock_openai_client(score: int = 4, reasoning: str = "", confidence: float = 0.88):
    """Build a mocked OpenAI client that returns a canned _JudgeScore.

    Simulates: client.beta.chat.completions.parse(...)
    """
    if not reasoning:
        reasoning = (
            "The response correctly identifies Paris as the capital of France "
            "and mentions the Eiffel Tower, matching the reference. "
            "It includes additional relevant context, so not a verbatim match, "
            "but factually accurate and well-phrased."
        )

    parsed_score = _JudgeScore(
        score=score,
        reasoning=reasoning,
        confidence=confidence,
    )

    mock_resp = SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(parsed=parsed_score))
        ]
    )

    mock_client = MagicMock()
    mock_client.beta.chat.completions.parse.return_value = mock_resp
    return mock_client


def main() -> None:
    # --- sample test case ---
    response_text = (
        "The capital of France is Paris. "
        "It is known for the Eiffel Tower and its rich history."
    )
    expected_text = "The capital of France is Paris."

    keywords = ["Paris", "Eiffel Tower", "France", "Louvre"]
    rubric = (
        "Rate factual accuracy and completeness on a 1-5 scale. "
        "1 = completely inaccurate, 3 = mostly accurate with minor issues, "
        "5 = perfectly accurate and complete."
    )

    # --- run evaluators ---
    exact_eval = ExactMatchEvaluator(case_sensitive=False)
    exact_result = exact_eval.evaluate(response_text, expected_text)

    keyword_eval = KeywordEvaluator(keywords=keywords, case_sensitive=False)
    keyword_result = keyword_eval.evaluate(response_text)

    # JSONLLMEvaluator with mocked OpenAI client (no API key needed)
    json_llm_eval = JSONLLMEvaluator(rubric=rubric, model="gpt-4o-mini")
    # Inject mock client directly to avoid needing openai installed / API key
    json_llm_eval._client = _mock_openai_client(
        score=4,
        confidence=0.88,
        reasoning=(
            "The response correctly identifies Paris as the capital of France "
            "and mentions the Eiffel Tower, matching the reference. "
            "It includes additional relevant context, so not a verbatim match, "
            "but factually accurate and well-phrased."
        ),
    )
    json_llm_result = json_llm_eval.evaluate(response_text, expected_text)

    # --- print formatted results ---
    print("=" * 60)
    print("Response: ", response_text)
    print("Expected: ", expected_text)
    print("Keywords:  ", ", ".join(keywords))
    print(f"Rubric:    {rubric}")
    print("=" * 60)

    print("\n--- ExactMatchEvaluator ---")
    print(json.dumps(exact_result, indent=2))

    print("\n--- KeywordEvaluator ---")
    print(json.dumps(keyword_result, indent=2))

    print("\n--- JSONLLMEvaluator (mocked) ---")
    print(json.dumps(json_llm_result, indent=2))


if __name__ == "__main__":
    main()
