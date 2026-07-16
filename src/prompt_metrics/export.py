# src/prompt_metrics/export.py
"""
Experiment result export utilities.

Supports JSON and flattened CSV output for downstream analysis
(e.g. pandas DataFrames, notebook ingestion).
"""

from __future__ import annotations

import csv
import json
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Flattening
# ---------------------------------------------------------------------------

def flatten_dict(
    obj: Any,
    parent_key: str = "",
    sep: str = "_",
) -> dict[str, Any]:
    """
    Recursively flatten a nested dict/list structure into a flat dict.

    Example:
        {"evaluator_a": {"score": 0.8, "passed": true}}
        → {"evaluator_a_score": 0.8, "evaluator_a_passed": true}

    Lists are indexed: {"flags": [{"level": "SEV1"}]}
        → {"flags_0_level": "SEV1"}
    """
    items: dict[str, Any] = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, (dict, list)):
                items.update(flatten_dict(v, new_key, sep=sep))
            else:
                items[new_key] = v

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_key = f"{parent_key}{sep}{i}"
            if isinstance(v, (dict, list)):
                items.update(flatten_dict(v, new_key, sep=sep))
            else:
                items[new_key] = v
    else:
        # Scalar value at root — shouldn't normally happen, but handle it
        items[parent_key or "value"] = obj

    return items


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _normalise_results(results: Any) -> list[dict[str, Any]]:
    """
    Accept CaseResult objects, SuiteResult objects, or raw dicts/lists,
    and normalise to a flat list[dict] of case records.
    """
    # SuiteResult-like
    if hasattr(results, "results") and isinstance(results.results, list):
        results = results.results

    # Single CaseResult
    if hasattr(results, "to_dict") and callable(results.to_dict):
        results = [results.to_dict()]

    # List of CaseResult objects
    if isinstance(results, list) and results and hasattr(results[0], "to_dict"):
        results = [r.to_dict() for r in results]

    # Already a list of dicts
    if isinstance(results, list):
        return results

    # Single dict
    if isinstance(results, dict):
        # If it looks like a SuiteResult.to_dict() envelope, unwrap it
        if "results" in results and isinstance(results["results"], list):
            return results["results"]
        return [results]

    raise TypeError(
        f"export_results: unsupported results type {type(results)!r}. "
        "Expected SuiteResult, CaseResult, list[CaseResult], or list[dict]."
    )


def _write_json(records: list[dict[str, Any]], output_path: str) -> None:
    """Write records as pretty-printed JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def _write_csv(records: list[dict[str, Any]], output_path: str) -> None:
    """Write records as a flattened CSV."""
    if not records:
        # Write an empty file with no header — caller can decide what to do
        open(output_path, "w", encoding="utf-8").close()
        return

    # Flatten every row and collect the full column universe
    flat_rows: list[dict[str, Any]] = []
    all_columns: set[str] = set()

    for rec in records:
        flat = flatten_dict(rec)
        flat_rows.append(flat)
        all_columns.update(flat.keys())

    # Stable column ordering:
    # 1. Core case metadata first, 2. evaluator_results_*, 3. everything else
    core_prefixes = (
        "case_id",
        "input_prompt",
        "generated_response",
        "expected_text",
        "keywords",
        "latency_ms",
        "error",
    )

    def column_sort_key(col: str) -> tuple[int, str]:
        for i, prefix in enumerate(core_prefixes):
            if col == prefix or col.startswith(prefix + "_"):
                return (i, col)
        if col.startswith("evaluator_results_"):
            return (len(core_prefixes), col)
        if col.startswith("metadata_"):
            return (len(core_prefixes) + 2, col)
        return (len(core_prefixes) + 1, col)

    fieldnames = sorted(all_columns, key=column_sort_key)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in flat_rows:
            # Scalar-serialise complex leftovers (shouldn't happen post-flatten,
            # but be defensive)
            serialised = {
                k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                for k, v in row.items()
            }
            writer.writerow(serialised)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_results(
    results: Any,
    output_path: str,
    format: str = "json",
) -> str:
    """
    Export experiment results to disk.

    Args:
        results: One of:
            - SuiteResult
            - CaseResult
            - list[CaseResult]
            - list[dict]  (case records)
            - dict        (single case record, or SuiteResult.to_dict() envelope)
        output_path: Destination file path.
        format: "json" | "csv"
            - json: Flat array of case record dicts, pretty-printed.
            - csv:  Fully flattened table. Nested keys in evaluator_results
                    become separate columns:
                    evaluator_results.exact_match.score
                    → evaluator_results_exact_match_score
                    Lists are indexed:
                    severity_flags[0].level
                    → evaluator_results_..._severity_flags_0_level
                    Ideal for pd.read_csv().

    Returns:
        The output_path that was written.

    Raises:
        ValueError: If format is not "json" or "csv".
        TypeError:  If results is not a recognised type.
    """
    fmt = format.lower().strip()
    if fmt not in {"json", "csv"}:
        raise ValueError(f'format must be "json" or "csv", got {format!r}')

    records = _normalise_results(results)

    if fmt == "json":
        _write_json(records, output_path)
    else:  # csv
        _write_csv(records, output_path)

    return output_path


__all__ = ["export_results", "flatten_dict"]
