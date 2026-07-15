"""
score_response.py

A simple starter script for scoring an LLM prompt/response pair
against a basic rubric. Intended as a starting point for
prompt-metrics — extend the rubric and scoring logic as needed.

Usage:
    python score_response.py
"""

RUBRIC = {
    "relevance": "Does the response directly address the prompt?",
    "accuracy": "Is the information factually correct?",
    "clarity": "Is the response easy to understand?",
    "completeness": "Does it fully answer the question, without missing key parts?",
}


def score_response(prompt: str, response: str, scores: dict) -> dict:
    """
    Combine a prompt/response pair with manual rubric scores (1-5 each)
    and return a summary including the average score.

    Args:
        prompt: The input prompt given to the LLM.
        response: The LLM's response to be scored.
        scores: Dict mapping each rubric key to an integer score (1-5).

    Returns:
        A dict containing the original data plus the computed average.
    """
    missing = [key for key in RUBRIC if key not in scores]
    if missing:
        raise ValueError(f"Missing scores for: {', '.join(missing)}")

    average = sum(scores.values()) / len(scores)

    return {
        "prompt": prompt,
        "response": response,
        "scores": scores,
        "average": round(average, 2),
    }


if __name__ == "__main__":
    example_prompt = "Explain what a hallucination is in the context of LLMs."
    example_response = (
        "A hallucination is when an LLM generates information that "
        "sounds plausible but is factually incorrect or not grounded "
        "in the source data."
    )

    example_scores = {
        "relevance": 5,
        "accuracy": 5,
        "clarity": 4,
        "completeness": 4,
    }

    result = score_response(example_prompt, example_response, example_scores)

    print("Prompt:", result["prompt"])
    print("Response:", result["response"])
    print("Scores:", result["scores"])
    print("Average score:", result["average"])
