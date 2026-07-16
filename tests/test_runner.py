# tests/test_runner.py
"""
Tests for prompt_metrics.ExperimentRunner and export_results.

Covers:
  - End-to-end suite execution with a mock dataset and generator
  - JSON and CSV export with nested key flattening
  - Result aggregation correctness
"""

from __future__ import annotations

import csv
import json
from typing import Any

import pytest

from src.prompt_metrics import (
    ExperimentRunner,
    CaseResult,
    SuiteResult,
    TestCase,
    export_results,
)


# ---------------------------------------------------------------------------
# Mock evaluators
# ---------------------------------------------------------------------------

class ExactMatchEvaluator:
    """Trivial evaluator: 1.0 if expected_text matches response exactly, else 0.0."""
    name = "exact_match"

    def evaluate(
        self,
        prompt: str,
        response: str,
        *,
        expected_text: str | None = None,
        keywords: list[str] | None = None,
        case_id: str | None = None,
    ) -> dict[str, Any]:
        if expected_text is None:
            return {"score": None, "passed": None, "reason": "no expected_text"}
        passed = response.strip() == expected_text.strip()
        return {
            "score": 1.0 if passed else 0.0,
            "passed": passed,
            "expected": expected_text,
            "actual": response,
        }


class KeywordEvaluator:
    """
    Scores the fraction of expected keywords found in the response.
    Returns nested structure to test CSV flattening of lists/dicts.
    """
    name = "keyword_match"

    def evaluate(
        self,
        prompt: str,
        response: str,
        *,
        expected_text: str | None = None,
        keywords: list[str] | None = None,
        case_id: str | None = None,
    ) -> dict[str, Any]:
        keywords = keywords or []
        if not keywords:
            return {"score": None, "matched": [], "missing": []}

        response_lower = response.lower()
        matched = [kw for kw in keywords if kw.lower() in response_lower]
        missing = [kw for kw in keywords if kw.lower() not in response_lower]
        score = len(matched) / len(keywords)

        # Nested structure + list of dicts → tests deep flattening
        return {
            "score": round(score, 4),
            "counts": {"matched": len(matched), "total": len(keywords)},
            "matched_keywords": matched,
            "missing_keywords": missing,
            "details": [
                {"keyword": kw, "found": kw.lower() in response_lower}
                for kw in keywords
            ],
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_dataset() -> list[dict[str, Any]]:
    """Small 3-case dataset covering exact matches, keyword hits/misses."""
    return [
        {
            "id": "case_001",
            "input_prompt": "What is 2 + 2?",
            "expected_text": "4",
            "keywords": ["four", "4"],
        },
        {
            "id": "case_002",
            "input_prompt": "Name a red fruit.",
            "expected_text": "apple",
            "keywords": ["apple", "strawberry", "cherry"],
        },
        {
            "id": "case_003",
            "input_prompt": "Say hello.",
            "expected_text": None,  # no ground truth
            "keywords": ["hello", "hi"],
        },
    ]


@pytest.fixture
def mock_generator() -> Any:
    """
    Deterministic mock generator: returns canned responses by prompt content.
    """
    canned = {
        "2 + 2": "4",                                    # exact match → pass
        "red fruit": "A strawberry is red and tasty.",  # keyword hit
        "hello": "Greetings!",                           # keyword miss
    }

    def generator(prompt: str) -> str:
        prompt_lower = prompt.lower()
        for key, response in canned.items():
            if key in prompt_lower:
                return response
        return "I don't know."

    return generator


# ---------------------------------------------------------------------------
# Test 1: ExperimentRunner end-to-end pipeline
# ---------------------------------------------------------------------------

def test_runner_executes_full_suite(
    mock_dataset: list[dict[str, Any]],
    mock_generator: Any,
) -> None:
    """
    Verify ExperimentRunner:
      - iterates the full dataset
      - calls generator_fn for each case
      - runs all evaluators on each output
      - returns a valid SuiteResult with correct aggregations
    """
    runner = ExperimentRunner([
        ExactMatchEvaluator(),
        KeywordEvaluator(),
    ])

    suite = runner.run_suite(
        mock_dataset,
        mock_generator,
        continue_on_error=False,
        verbose=False,
    )

    # --- Suite-level assertions ---
    assert isinstance(suite, SuiteResult)
    assert suite.total_cases == 3
    assert suite.successful_cases == 3
    assert suite.failed_cases == 0
    assert suite.evaluator_names == ["exact_match", "keyword_match"]
    assert suite.total_runtime_s >= 0
    assert len(suite.results) == 3

    # --- Per-case assertions ---
    results_by_id = {r.case_id: r for r in suite.results}

    # case_001: "4" — exact match, keyword "4" found
    r1 = results_by_id["case_001"]
    assert isinstance(r1, CaseResult)
    assert r1.generated_response == "4"
    assert r1.error is None
    assert "exact_match" in r1.evaluator_results
    assert "keyword_match" in r1.evaluator_results
    assert r1.evaluator_results["exact_match"]["passed"] is True
    assert r1.evaluator_results["exact_match"]["score"] == 1.0
    assert r1.evaluator_results["keyword_match"]["score"] == 0.5  # "4" hit, "four" miss
    assert r1.latency_ms >= 0

    # case_002: "A strawberry is red and tasty."
    # expected_text="apple" → no exact match; keyword "strawberry" hit
    r2 = results_by_id["case_002"]
    assert r2.evaluator_results["exact_match"]["passed"] is False
    assert r2.evaluator_results["exact_match"]["score"] == 0.0
    assert abs(r2.evaluator_results["keyword_match"]["score"] - 1/3) < 1e-4
    assert "strawberry" in r2.evaluator_results["keyword_match"]["matched_keywords"]

    # case_003: "Greetings!" — no expected_text, no keyword hits
    r3 = results_by_id["case_003"]
    assert r3.evaluator_results["exact_match"]["score"] is None
    assert r3.evaluator_results["keyword_match"]["score"] == 0.0
    assert r3.evaluator_results["keyword_match"]["counts"]["matched"] == 0


def test_runner_loads_dataset_from_dict_list(
    mock_generator: Any,
) -> None:
    """Sanity check: runner accepts raw list[dict], not just TestCase objects."""
    runner = ExperimentRunner([ExactMatchEvaluator()])
    dataset_dicts = [
        {"id": "x1", "input_prompt": "test", "expected_text": "foo"}
    ]

    def gen(_: str) -> str:
        return "foo"

    suite = runner.run_suite(dataset_dicts, gen)
    assert suite.total_cases == 1
    assert suite.results[0].case_id == "x1"


def test_runner_records_generator_errors() -> None:
    """Failed generator calls should be recorded in CaseResult.error, not crash the suite."""

    def failing_generator(prompt: str) -> str:
        if "boom" in prompt.lower():
            raise RuntimeError("simulated generator crash")
        return "ok"

    runner = ExperimentRunner([ExactMatchEvaluator()])
    dataset = [
        {"id": "good", "input_prompt": "normal prompt"},
        {"id": "bad", "input_prompt": "this will boom"},
    ]

    suite = runner.run_suite(dataset, failing_generator, continue_on_error=True)

    assert suite.total_cases == 2
    assert suite.successful_cases == 1
    assert suite.failed_cases == 1

    bad_result = next(r for r in suite.results if r.case_id == "bad")
    assert bad_result.error is not None
    assert "generator_fn failed" in bad_result.error
    assert "simulated generator crash" in bad_result.error


# ---------------------------------------------------------------------------
# Test 2 & 3: export_results — JSON + flattened CSV
# ---------------------------------------------------------------------------

def test_export_json_and_csv(
    mock_dataset: list[dict[str, Any]],
    mock_generator: Any,
    tmp_path: Any,
) -> None:
    """
    Verify export_results:
      - Accepts a SuiteResult directly
      - Writes valid JSON with all case fields intact
      - Writes a flattened CSV where nested evaluator keys become columns
    """
    runner = ExperimentRunner([ExactMatchEvaluator(), KeywordEvaluator()])
    suite = runner.run_suite(mock_dataset, mock_generator)

    json_path = tmp_path / "results.json"
    csv_path = tmp_path / "results.csv"

    # Export both formats
    out_json = export_results(suite, str(json_path), format="json")
    out_csv = export_results(suite, str(csv_path), format="csv")

    assert out_json == str(json_path)
    assert out_csv == str(csv_path)
    assert json_path.exists()
    assert csv_path.exists()

    # --- JSON validation ---
    with open(json_path, encoding="utf-8") as f:
        json_data = json.load(f)

    assert isinstance(json_data, list)
    assert len(json_data) == 3
    # Nested structure should be preserved in JSON
    assert "evaluator_results" in json_data[0]
    assert "exact_match" in json_data[0]["evaluator_results"]
    assert "keyword_match" in json_data[0]["evaluator_results"]
    assert json_data[0]["evaluator_results"]["exact_match"]["score"] in (0.0, 1.0, None)

    # --- CSV validation: nested keys are flattened ---
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        csv_rows = list(reader)
        csv_fieldnames = reader.fieldnames or []

    assert len(csv_rows) == 3

    # Core case metadata columns exist
    for col in ["case_id", "input_prompt", "generated_response", "latency_ms"]:
        assert col in csv_fieldnames, f"Missing core column: {col}"

    # Nested evaluator results are flattened into separate columns
    # exact_match.score → evaluator_results_exact_match_score
    assert "evaluator_results_exact_match_score" in csv_fieldnames
    assert "evaluator_results_exact_match_passed" in csv_fieldnames

    # keyword_match nested dict: counts.matched → ...counts_matched
    assert "evaluator_results_keyword_match_score" in csv_fieldnames
    assert "evaluator_results_keyword_match_counts_matched" in csv_fieldnames
    assert "evaluator_results_keyword_match_counts_total" in csv_fieldnames

    # keyword_match list fields are indexed
    # matched_keywords[0], missing_keywords[0], details[0].keyword, etc.
    keyword_cols = [c for c in csv_fieldnames if "keyword_match" in c]
    assert len(keyword_cols) >= 4, f"Expected flattened keyword_match columns, got: {keyword_cols}"

    # List-of-dicts flattening: details[0].keyword → ..._details_0_keyword
    details_cols = [c for c in csv_fieldnames if "details_0_keyword" in c]
    assert len(details_cols) > 0, (
        "List dimensions (details[0].keyword) were not flattened. "
        f"Available columns: {csv_fieldnames}"
    )

    # Spot-check actual values in the CSV
    case_001_row = next(r for r in csv_rows if r["case_id"] == "case_001")
    assert case_001_row["evaluator_results_exact_match_score"] == "1.0"
    assert case_001_row["evaluator_results_exact_match_passed"] == "True"
    assert case_001_row["evaluator_results_keyword_match_counts_total"] == "2"


def test_export_accepts_various_result_types(tmp_path: Any) -> None:
    """export_results should normalise CaseResult, list[CaseResult], list[dict], etc."""
    case = CaseResult(
        case_id="solo",
        input_prompt="test",
        generated_response="ok",
        evaluator_results={"dummy": {"score": 0.99}},
    )

    # Single CaseResult
    p1 = tmp_path / "single.json"
    export_results(case, str(p1), format="json")
    with open(p1) as f:
        data = json.load(f)
    assert isinstance(data, list) and len(data) == 1

    # list[dict]
    p2 = tmp_path / "raw.csv"
    export_results([{"case_id": "x", "foo": {"bar": 1}}], str(p2), format="csv")
    with open(p2) as f:
        assert "foo_bar" in f.read()


def test_export_csv_is_pandas_compatible(
    mock_dataset: list[dict[str, Any]],
    mock_generator: Any,
    tmp_path: Any,
) -> None:
    """
    Sanity check: the flattened CSV can be read by pandas without errors,
    and evaluator score columns are parseable as numeric.
    """
    pd = pytest.importorskip("pandas")

    runner = ExperimentRunner([ExactMatchEvaluator(), KeywordEvaluator()])
    suite = runner.run_suite(mock_dataset, mock_generator)

    csv_path = tmp_path / "pandas_test.csv"
    export_results(suite, str(csv_path), format="csv")

    df = pd.read_csv(csv_path)
    assert len(df) == 3
    assert "case_id" in df.columns
    assert "evaluator_results_exact_match_score" in df.columns

    # Score column should be numeric (allowing NaN for None)
    scores = pd.to_numeric(df["evaluator_results_exact_match_score"], errors="coerce")
    assert scores.notna().sum() >= 2  # at least 2 cases had a real score
