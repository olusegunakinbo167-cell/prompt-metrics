# src/prompt_metrics/monitoring.py
"""
Regression and metric drift monitoring.

Compares a current SuiteResult against a baseline run to detect:
  - Score drops / metric drift per evaluator
  - Latency regressions
  - New failures, missing cases, etc.

Use this to catch performance regressions between commits, model versions,
or releases.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from typing import Any

from .runner import CaseResult, SuiteResult


# ---------------------------------------------------------------------------
# Score extraction (shared with reports.py)
# ---------------------------------------------------------------------------

def _extract_numeric_score(result: Any) -> float | None:
    """
    Extract a numeric score from an evaluator result.

    Tries common score keys: score, overall_score, accuracy, f1, etc.
    Returns None if no numeric score is found.
    """
    if result is None:
        return None
    if isinstance(result, (int, float)):
        return float(result)
    if not isinstance(result, dict):
        return None

    for key in ("score", "overall_score", "accuracy", "f1", "similarity"):
        if key in result and isinstance(result[key], (int, float)):
            return float(result[key])

    # Percentage → normalise to 0-1
    if "percentage" in result and isinstance(result["percentage"], (int, float)):
        return float(result["percentage"]) / 100.0

    # One-level recursive search
    for v in result.values():
        if isinstance(v, dict):
            nested = _extract_numeric_score(v)
            if nested is not None:
                return nested

    return None


# ---------------------------------------------------------------------------
# Suite normalization
# ---------------------------------------------------------------------------

def _normalize_suite(suite: Any) -> dict[str, Any]:
    """
    Normalize a SuiteResult (object, dict, or list) into a standard dict:
    {
      "results": {case_id: CaseResult-like dict, ...},
      "evaluator_names": [...],
      "summary": {...}  # optional
    }

    Accepts:
      - SuiteResult object
      - SuiteResult.to_dict() envelope {"summary": {...}, "results": [...]}
      - list[CaseResult] / list[dict]  (flat case records, e.g. from export_results json)
      - dict with "results" key
    """
    # SuiteResult object
    if isinstance(suite, SuiteResult):
        results = {r.case_id: r.to_dict() for r in suite.results}
        return {
            "results": results,
            "evaluator_names": suite.evaluator_names,
            "summary": {
                "total_cases": suite.total_cases,
                "successful_cases": suite.successful_cases,
                "failed_cases": suite.failed_cases,
                "total_runtime_s": suite.total_runtime_s,
            },
        }

    # Dict input
    if isinstance(suite, dict):
        # SuiteResult.to_dict() envelope
        if "results" in suite and isinstance(suite["results"], list):
            results_list = suite["results"]
            summary = suite.get("summary", {})
        else:
            # Single case dict – wrap it
            results_list = [suite]
            summary = {}

        results = {}
        for r in results_list:
            if isinstance(r, CaseResult):
                r = r.to_dict()
            case_id = r.get("case_id")
            if case_id:
                results[case_id] = r

        # Infer evaluator names from the first case
        evaluator_names: list[str] = []
        if results:
            first = next(iter(results.values()))
            eval_results = first.get("evaluator_results", {})
            if isinstance(eval_results, dict):
                evaluator_names = sorted(eval_results.keys())

        return {
            "results": results,
            "evaluator_names": evaluator_names,
            "summary": summary,
        }

    # List input: list[CaseResult] or list[dict]
    if isinstance(suite, list):
        results: dict[str, Any] = {}
        for r in suite:
            if isinstance(r, CaseResult):
                r = r.to_dict()
            if isinstance(r, dict) and "case_id" in r:
                results[r["case_id"]] = r

        evaluator_names = []
        if results:
            first = next(iter(results.values()))
            eval_results = first.get("evaluator_results", {})
            if isinstance(eval_results, dict):
                evaluator_names = sorted(eval_results.keys())

        return {
            "results": results,
            "evaluator_names": evaluator_names,
            "summary": {},
        }

    raise TypeError(
        f"compare_suites: unsupported suite type {type(suite)!r}. "
        "Expected SuiteResult, list[CaseResult], list[dict], "
        "or SuiteResult.to_dict() envelope."
    )


def load_suite_result(path: str) -> dict[str, Any]:
    """
    Load a SuiteResult from a JSON file.

    Handles both formats:
      - Full SuiteResult envelope: {"summary": {...}, "results": [...]}
      - Flat case list: [{"case_id": "...", ...}, ...]  (export_results format)

    Returns a normalized suite dict (see _normalize_suite).
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Detect format and normalize
    if isinstance(raw, list):
        # export_results JSON: flat list of case records
        return _normalize_suite(raw)
    elif isinstance(raw, dict):
        # Could be SuiteResult.to_dict() envelope, or single case
        return _normalize_suite(raw)
    else:
        raise ValueError(f"Unexpected JSON structure in {path}: {type(raw)}")


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_suites(
    current_suite: Any,
    baseline_suite: Any,
    *,
    latency_regression_threshold: float = 1.2,
    score_drop_threshold: float = 0.15,
) -> dict[str, Any]:
    """
    Compare a current SuiteResult against a baseline.

    Aligns test cases by case_id, calculates metric deltas and latency changes,
    and flags regressions.

    Args:
        current_suite: Current run. Accepts SuiteResult, list[CaseResult],
            list[dict], or SuiteResult.to_dict() envelope.
        baseline_suite: Baseline run to compare against. Same accepted types.
        latency_regression_threshold: Latency increase factor that counts as
            a regression. Default 1.2 = 20% slower.
            A case is flagged if current_latency / baseline_latency > threshold.
        score_drop_threshold: Score decrease that counts as a regression.
            Default 0.15 = 0.15 point drop (on 0-1 scale).
            A case is flagged if baseline_score - current_score > threshold.

    Returns:
        {
          "summary": {
            "current_cases": int,
            "baseline_cases": int,
            "common_cases": int,
            "new_cases": [case_id, ...],
            "missing_cases": [case_id, ...],
            "latency_regression_threshold": float,
            "score_drop_threshold": float,
          },
          "metrics_drift": {
            evaluator_name: {
              "mean_delta": float,      # current - baseline
              "median_delta": float,
              "min_delta": float,
              "max_delta": float,
              "stdev_delta": float,
              "count": int,             # number of cases with both scores
              "improved": int,          # count where delta > +0.01
              "regressed": int,         # count where delta < -0.01
              "unchanged": int,
            }, ...
          },
          "latency_regressions": [
            {
              "case_id": str,
              "baseline_ms": float,
              "current_ms": float,
              "factor": float,          # current / baseline
              "delta_ms": float,
            }, ...
          ],
          "score_drops": [
            {
              "case_id": str,
              "evaluator": str,
              "baseline_score": float,
              "current_score": float,
              "delta": float,           # current - baseline (negative)
            }, ...
          ],
          "per_case": {
            case_id: {
              "latency_baseline_ms": float | None,
              "latency_current_ms": float | None,
              "latency_factor": float | None,
              "latency_regression": bool,
              "evaluators": {
                evaluator_name: {
                  "baseline": float | None,
                  "current": float | None,
                  "delta": float | None,
                  "score_drop": bool,
                }, ...
              }
            }, ...
          }
        }
    """
    current = _normalize_suite(current_suite)
    baseline = _normalize_suite(baseline_suite)

    current_results = current["results"]
    baseline_results = baseline["results"]

    current_ids = set(current_results.keys())
    baseline_ids = set(baseline_results.keys())

    common_ids = sorted(current_ids & baseline_ids)
    new_ids = sorted(current_ids - baseline_ids)
    missing_ids = sorted(baseline_ids - current_ids)

    # Per-evaluator deltas
    per_eval_deltas: dict[str, list[float]] = defaultdict(list)
    score_drops: list[dict[str, Any]] = []
    latency_regressions: list[dict[str, Any]] = []
    per_case: dict[str, Any] = {}

    for case_id in sorted(current_ids | baseline_ids):
        curr_case = current_results.get(case_id)
        base_case = baseline_results.get(case_id)

        # Latency comparison
        curr_latency = None
        base_latency = None
        latency_factor = None
        latency_regression = False

        if curr_case:
            curr_latency = float(curr_case.get("latency_ms", 0) or 0)
        if base_case:
            base_latency = float(base_case.get("latency_ms", 0) or 0)

        if (
            curr_latency is not None
            and base_latency is not None
            and base_latency > 0
        ):
            latency_factor = curr_latency / base_latency
            latency_regression = latency_factor > latency_regression_threshold
            if latency_regression:
                latency_regressions.append(
                    {
                        "case_id": case_id,
                        "baseline_ms": round(base_latency, 2),
                        "current_ms": round(curr_latency, 2),
                        "factor": round(latency_factor, 3),
                        "delta_ms": round(curr_latency - base_latency, 2),
                    }
                )

        # Evaluator score comparison
        curr_evals = (
            curr_case.get("evaluator_results", {}) if curr_case else {}
        )
        base_evals = (
            base_case.get("evaluator_results", {}) if base_case else {}
        )

        all_evaluators = set(curr_evals.keys()) | set(base_evals.keys())
        case_eval_details: dict[str, Any] = {}

        for eval_name in sorted(all_evaluators):
            curr_score = _extract_numeric_score(curr_evals.get(eval_name))
            base_score = _extract_numeric_score(base_evals.get(eval_name))

            delta = None
            score_drop = False
            if curr_score is not None and base_score is not None:
                delta = curr_score - base_score
                per_eval_deltas[eval_name].append(delta)
                score_drop = delta < -score_drop_threshold
                if score_drop:
                    score_drops.append(
                        {
                            "case_id": case_id,
                            "evaluator": eval_name,
                            "baseline_score": round(base_score, 4),
                            "current_score": round(curr_score, 4),
                            "delta": round(delta, 4),
                        }
                    )

            case_eval_details[eval_name] = {
                "baseline": round(base_score, 4) if base_score is not None else None,
                "current": round(curr_score, 4) if curr_score is not None else None,
                "delta": round(delta, 4) if delta is not None else None,
                "score_drop": score_drop,
            }

        per_case[case_id] = {
            "latency_baseline_ms": round(base_latency, 2)
            if base_latency is not None
            else None,
            "latency_current_ms": round(curr_latency, 2)
            if curr_latency is not None
            else None,
            "latency_factor": round(latency_factor, 3)
            if latency_factor is not None
            else None,
            "latency_regression": latency_regression,
            "evaluators": case_eval_details,
            "is_new": case_id in new_ids,
            "is_missing": case_id in missing_ids,
        }

    # Aggregate metrics_drift
    metrics_drift: dict[str, Any] = {}
    for eval_name, deltas in per_eval_deltas.items():
        if not deltas:
            continue
        improved = sum(1 for d in deltas if d > 0.01)
        regressed = sum(1 for d in deltas if d < -0.01)
        unchanged = len(deltas) - improved - regressed
        metrics_drift[eval_name] = {
            "mean_delta": round(statistics.mean(deltas), 4),
            "median_delta": round(statistics.median(deltas), 4),
            "min_delta": round(min(deltas), 4),
            "max_delta": round(max(deltas), 4),
            "stdev_delta": round(statistics.stdev(deltas), 4)
            if len(deltas) > 1
            else 0.0,
            "count": len(deltas),
            "improved": improved,
            "regressed": regressed,
            "unchanged": unchanged,
        }

    # Sort regressions by severity
    latency_regressions.sort(key=lambda x: x["factor"], reverse=True)
    score_drops.sort(key=lambda x: x["delta"])  # most negative first

    return {
        "summary": {
            "current_cases": len(current_ids),
            "baseline_cases": len(baseline_ids),
            "common_cases": len(common_ids),
            "new_cases": new_ids,
            "missing_cases": missing_ids,
            "latency_regression_threshold": latency_regression_threshold,
            "score_drop_threshold": score_drop_threshold,
            "total_latency_regressions": len(latency_regressions),
            "total_score_drops": len(score_drops),
        },
        "metrics_drift": metrics_drift,
        "latency_regressions": latency_regressions,
        "score_drops": score_drops,
        "per_case": per_case,
    }


__all__ = ["compare_suites", "load_suite_result"]
