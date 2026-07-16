# tests/test_monitoring.py
"""
Tests for prompt_metrics.monitoring — regression and drift detection.

Covers:
  - compare_suites() metric delta calculations
  - Score drop detection with configurable thresholds
  - Latency regression detection
  - Case alignment (new/missing cases)
  - Baseline JSON loading (both export_results and SuiteResult.save_json formats)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from src.prompt_metrics import (
    CaseResult,
    ExperimentRunner,
    SuiteResult,
    export_results,
)
from src.prompt_metrics.monitoring import compare_suites, load_suite_result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class MockEvaluator:
    """Simple evaluator that returns a configurable score."""

    def __init__(self, name: str = "mock_score", score_map: dict[str, float] | None = None):
        self.name = name
        self.score_map = score_map or {}

    def evaluate(
        self,
        prompt: str,
        response: str,
        *,
        expected_text: str | None = None,
        keywords: list[str] | None = None,
        case_id: str | None = None,
    ) -> dict[str, Any]:
        # Look up score by case_id, default 0.5
        score = self.score_map.get(case_id or "", 0.5)
        return {"score": score}


def _make_suite(
    case_scores: dict[str, float],
    latencies_ms: dict[str, float] | None = None,
    evaluator_name: str = "test_eval",
) -> SuiteResult:
    """Build a SuiteResult with given per-case scores and latencies."""
    latencies_ms = latencies_ms or {}
    results = []
    for case_id, score in case_scores.items():
        results.append(
            CaseResult(
                case_id=case_id,
                input_prompt=f"prompt for {case_id}",
                generated_response=f"response for {case_id}",
                expected_text=None,
                keywords=None,
                evaluator_results={evaluator_name: {"score": score}},
                latency_ms=latencies_ms.get(case_id, 100.0),
                error=None,
                metadata={},
            )
        )
    return SuiteResult(
        results=results,
        evaluator_names=[evaluator_name],
        total_cases=len(results),
        successful_cases=len(results),
        failed_cases=0,
        total_runtime_s=1.0,
    )


# ---------------------------------------------------------------------------
# Test: basic metric drift calculation
# ---------------------------------------------------------------------------

def test_compare_suites_calculates_deltas():
    """compare_suites should compute correct mean/median/min/max deltas."""
    baseline = _make_suite(
        {"case_a": 0.8, "case_b": 0.6, "case_c": 0.9},
        latencies_ms={"case_a": 100, "case_b": 200, "case_c": 150},
    )
    current = _make_suite(
        # Deltas: a: +0.1, b: -0.2, c: 0.0
        {"case_a": 0.9, "case_b": 0.4, "case_c": 0.9},
        latencies_ms={"case_a": 110, "case_b": 210, "case_c": 140},
    )

    result = compare_suites(current, baseline)

    # Summary
    assert result["summary"]["current_cases"] == 3
    assert result["summary"]["baseline_cases"] == 3
    assert result["summary"]["common_cases"] == 3
    assert result["summary"]["new_cases"] == []
    assert result["summary"]["missing_cases"] == []

    # Metrics drift
    drift = result["metrics_drift"]["test_eval"]
    # Deltas: [0.1, -0.2, 0.0] → mean = -0.0333, median = 0.0
    assert drift["mean_delta"] == pytest.approx(-0.0333, abs=1e-4)
    assert drift["median_delta"] == pytest.approx(0.0, abs=1e-4)
    assert drift["min_delta"] == pytest.approx(-0.2)
    assert drift["max_delta"] == pytest.approx(0.1)
    assert drift["count"] == 3
    assert drift["improved"] == 1  # case_a
    assert drift["regressed"] == 1  # case_b
    assert drift["unchanged"] == 1  # case_c

    # Per-case deltas
    per_case = result["per_case"]
    assert per_case["case_a"]["evaluators"]["test_eval"]["delta"] == pytest.approx(0.1)
    assert per_case["case_b"]["evaluators"]["test_eval"]["delta"] == pytest.approx(-0.2)
    assert per_case["case_c"]["evaluators"]["test_eval"]["delta"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test: score drop detection
# ---------------------------------------------------------------------------

def test_compare_suites_detects_score_drops():
    """Cases with score drops exceeding threshold should be flagged."""
    baseline = _make_suite({"c1": 0.9, "c2": 0.8, "c3": 0.5})
    current = _make_suite({"c1": 0.9, "c2": 0.6, "c3": 0.3})
    # Deltas: c1: 0.0, c2: -0.2, c3: -0.2

    # Default threshold = 0.15, so c2 and c3 should be flagged
    result = compare_suites(current, baseline, score_drop_threshold=0.15)
    score_drops = result["score_drops"]
    assert len(score_drops) == 2
    dropped_ids = {d["case_id"] for d in score_drops}
    assert dropped_ids == {"c2", "c3"}

    # Verify drop details
    c2_drop = next(d for d in score_drops if d["case_id"] == "c2")
    assert c2_drop["baseline_score"] == pytest.approx(0.8)
    assert c2_drop["current_score"] == pytest.approx(0.6)
    assert c2_drop["delta"] == pytest.approx(-0.2)

    # Higher threshold: only drops > 0.25 should be flagged (none)
    result2 = compare_suites(current, baseline, score_drop_threshold=0.25)
    assert len(result2["score_drops"]) == 0

    # Lower threshold: drops > 0.05 should flag c2 and c3
    result3 = compare_suites(current, baseline, score_drop_threshold=0.05)
    assert len(result3["score_drops"]) == 2


# ---------------------------------------------------------------------------
# Test: latency regression detection
# ---------------------------------------------------------------------------

def test_compare_suites_detects_latency_regressions():
    """Cases with latency increases exceeding threshold should be flagged."""
    baseline = _make_suite(
        {"fast": 0.8, "slow": 0.8, "same": 0.8},
        latencies_ms={"fast": 100.0, "slow": 100.0, "same": 100.0},
    )
    current = _make_suite(
        {"fast": 0.8, "slow": 0.8, "same": 0.8},
        # fast: 90ms (improvement, 0.9x), slow: 150ms (1.5x regression),
        # same: 105ms (1.05x, below default 1.2 threshold)
        latencies_ms={"fast": 90.0, "slow": 150.0, "same": 105.0},
    )

    # Default threshold = 1.2 (20% slower), only "slow" should be flagged
    result = compare_suites(current, baseline, latency_regression_threshold=1.2)
    regressions = result["latency_regressions"]
    assert len(regressions) == 1
    assert regressions[0]["case_id"] == "slow"
    assert regressions[0]["baseline_ms"] == pytest.approx(100.0)
    assert regressions[0]["current_ms"] == pytest.approx(150.0)
    assert regressions[0]["factor"] == pytest.approx(1.5)
    assert regressions[0]["delta_ms"] == pytest.approx(50.0)

    # Lower threshold = 1.04, "same" (1.05x) and "slow" (1.5x) flagged
    result2 = compare_suites(current, baseline, latency_regression_threshold=1.04)
    assert len(result2["latency_regressions"]) == 2
    # Sorted by severity (factor desc), so slow first
    assert result2["latency_regressions"][0]["case_id"] == "slow"


# ---------------------------------------------------------------------------
# Test: case alignment (new / missing cases)
# ---------------------------------------------------------------------------

def test_compare_suites_handles_new_and_missing_cases():
    """New cases in current, and missing cases from baseline, should be tracked."""
    baseline = _make_suite({"keep": 0.8, "gone": 0.7, "stable": 0.9})
    current = _make_suite({"keep": 0.85, "new": 0.6, "stable": 0.9})
    # baseline has "gone", current has "new"

    result = compare_suites(current, baseline)

    assert result["summary"]["baseline_cases"] == 3
    assert result["summary"]["current_cases"] == 3
    assert result["summary"]["common_cases"] == 2
    assert result["summary"]["new_cases"] == ["new"]
    assert result["summary"]["missing_cases"] == ["gone"]

    # Per-case metadata
    per_case = result["per_case"]
    assert per_case["new"]["is_new"] is True
    assert per_case["new"]["is_missing"] is False
    assert per_case["gone"]["is_new"] is False
    assert per_case["gone"]["is_missing"] is True
    assert per_case["keep"]["is_new"] is False
    assert per_case["keep"]["is_missing"] is False

    # Drift stats should only include common cases (keep, stable)
    # keep: +0.05, stable: 0.0 → mean = 0.025
    drift = result["metrics_drift"]["test_eval"]
    assert drift["count"] == 2  # only common cases with both scores
    assert drift["mean_delta"] == pytest.approx(0.025, abs=1e-4)


# ---------------------------------------------------------------------------
# Test: baseline JSON loading
# ---------------------------------------------------------------------------

def test_load_suite_result_handles_export_results_format(tmp_path: Path):
    """load_suite_result should handle export_results JSON (flat list)."""
    # Create a fake export_results JSON file
    results = [
        {
            "case_id": "x1",
            "input_prompt": "test",
            "generated_response": "foo",
            "evaluator_results": {"exact_match": {"score": 1.0}},
            "latency_ms": 42.5,
        },
        {
            "case_id": "x2",
            "input_prompt": "test2",
            "generated_response": "bar",
            "evaluator_results": {"exact_match": {"score": 0.0}},
            "latency_ms": 55.0,
        },
    ]
    json_path = tmp_path / "export_results.json"
    json_path.write_text(json.dumps(results), encoding="utf-8")

    suite = load_suite_result(str(json_path))
    assert "results" in suite
    assert set(suite["results"].keys()) == {"x1", "x2"}
    assert suite["results"]["x1"]["latency_ms"] == pytest.approx(42.5)


def test_load_suite_result_handles_suite_envelope_format(tmp_path: Path):
    """load_suite_result should handle SuiteResult.save_json() format."""
    envelope = {
        "summary": {"total_cases": 1, "successful_cases": 1},
        "results": [
            {
                "case_id": "y1",
                "input_prompt": "q?",
                "generated_response": "a",
                "evaluator_results": {"keyword": {"score": 0.75}},
                "latency_ms": 123.0,
            }
        ],
    }
    json_path = tmp_path / "suite_envelope.json"
    json_path.write_text(json.dumps(envelope), encoding="utf-8")

    suite = load_suite_result(str(json_path))
    assert set(suite["results"].keys()) == {"y1"}
    assert suite["summary"]["total_cases"] == 1


# ---------------------------------------------------------------------------
# Test: multi-evaluator drift
# ---------------------------------------------------------------------------

def test_compare_suites_multi_evaluator():
    """Drift should be calculated independently per evaluator."""
    # Build suites manually with 2 evaluators per case
    def make_case(case_id: str, eval_a_score: float, eval_b_score: float, latency: float = 100.0):
        return CaseResult(
            case_id=case_id,
            input_prompt="p",
            generated_response="r",
            evaluator_results={
                "eval_a": {"score": eval_a_score},
                "eval_b": {"score": eval_b_score},
            },
            latency_ms=latency,
        )

    baseline = SuiteResult(
        results=[
            make_case("c1", 0.8, 0.5),
            make_case("c2", 0.6, 0.7),
        ],
        evaluator_names=["eval_a", "eval_b"],
        total_cases=2,
        successful_cases=2,
        failed_cases=0,
        total_runtime_s=1.0,
    )
    current = SuiteResult(
        results=[
            # eval_a improves on average, eval_b regresses
            make_case("c1", 0.9, 0.3),  # a: +0.1, b: -0.2
            make_case("c2", 0.7, 0.6),  # a: +0.1, b: -0.1
        ],
        evaluator_names=["eval_a", "eval_b"],
        total_cases=2,
        successful_cases=2,
        failed_cases=0,
        total_runtime_s=1.0,
    )

    result = compare_suites(current, baseline, score_drop_threshold=0.15)

    # eval_a: mean_delta = +0.1 (improvement)
    assert result["metrics_drift"]["eval_a"]["mean_delta"] == pytest.approx(0.1)
    assert result["metrics_drift"]["eval_a"]["improved"] == 2

    # eval_b: mean_delta = -0.15 (regression)
    assert result["metrics_drift"]["eval_b"]["mean_delta"] == pytest.approx(-0.15)
    assert result["metrics_drift"]["eval_b"]["regressed"] == 2

    # Score drops: eval_b on c1 dropped by 0.2 (> 0.15 threshold)
    score_drops = result["score_drops"]
    assert len(score_drops) == 1
    assert score_drops[0]["case_id"] == "c1"
    assert score_drops[0]["evaluator"] == "eval_b"


# ---------------------------------------------------------------------------
# Test: comparison report generation
# ---------------------------------------------------------------------------

def test_generate_comparison_report(tmp_path: Path):
    """generate_comparison_report should produce a valid markdown file."""
    from src.prompt_metrics.reports import generate_comparison_report

    baseline = _make_suite(
        {"case_x": 0.9, "case_y": 0.5},
        latencies_ms={"case_x": 100, "case_y": 200},
    )
    current = _make_suite(
        {"case_x": 0.6, "case_y": 0.5},  # case_x regresses
        latencies_ms={"case_x": 150, "case_y": 205},  # case_x: 1.5x latency
    )

    comparison = compare_suites(
        current, baseline,
        score_drop_threshold=0.15,
        latency_regression_threshold=1.2,
    )

    report_path = tmp_path / "comparison.md"
    out = generate_comparison_report(
        comparison,
        str(report_path),
        title="Test Regression Report",
    )

    assert out == str(report_path)
    assert report_path.exists()

    md = report_path.read_text(encoding="utf-8")
    # Report should contain key sections
    assert "Regression Report" in md
    assert "Summary" in md
    assert "Metrics Drift" in md
    assert "Score Drops" in md
    assert "Latency Regressions" in md
    assert "Per-Case Comparison" in md
    # Should flag the regression
    assert "case_x" in md
    # Score dropped from 0.9 → 0.6
    assert "0.900" in md or "0.9" in md
