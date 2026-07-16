"""Quick ad-hoc runner for testing evaluators against sample data."""

import json

from src.prompt_metrics.evaluators import ExactMatchEvaluator, KeywordEvaluator


def main() -> None:
    # --- sample test case ---
    response_text = (
        "The capital of France is Paris. "
        "It is known for the Eiffel Tower and its rich history."
    )
    expected_text = "The capital of France is Paris."

    keywords = ["Paris", "Eiffel Tower", "France", "Louvre"]

    # --- run evaluators ---
    exact_eval = ExactMatchEvaluator(case_sensitive=False)
    exact_result = exact_eval.evaluate(response_text, expected_text)

    keyword_eval = KeywordEvaluator(keywords=keywords, case_sensitive=False)
    keyword_result = keyword_eval.evaluate(response_text)

    # --- print formatted results ---
    print("=" * 60)
    print("Response:", response_text)
    print("Expected:", expected_text)
    print("Keywords:", ", ".join(keywords))
    print("=" * 60)

    print("\n--- ExactMatchEvaluator ---")
    print(json.dumps(exact_result, indent=2))

    print("\n--- KeywordEvaluator ---")
    print(json.dumps(keyword_result, indent=2))


if __name__ == "__main__":
    main()
