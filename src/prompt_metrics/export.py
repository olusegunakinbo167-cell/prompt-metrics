from __future__ import annotations
import json
import csv
from typing import Any, Dict, List, Union

def flatten_dict(d: Any, parent_key: str = "", sep: str = "_") -> Dict[str, Any]:
    """
    Recursively flatten nested dicts/lists into a single-level dict.

    - dict keys are joined with `sep`
    - lists are indexed numerically: key_0, key_1, ...
    - lists containing only primitives also produce a joined string
      at the base key (semicolon-separated) to preserve a convenient
      flat column (e.g. `keywords` -> "key1; key2"), while still
      emitting indexed entries (keywords_0, keywords_1, ...).
    """
    items: Dict[str, Any] = {}

    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.update(flatten_dict(v, new_key, sep=sep))
        return items

    if isinstance(d, list):
        # If list is primitive-only, store a joined convenience copy
        if d and all(not isinstance(x, (dict, list)) for x in d):
            items[parent_key] = "; ".join("" if x is None else str(x) for x in d)
        elif not d and parent_key:
            # empty list -> explicit empty value at base key
            items[parent_key] = ""
        for i, v in enumerate(d):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.update(flatten_dict(v, new_key, sep=sep))
        return items

    # primitive
    if parent_key:
        items[parent_key] = d
    return items


def _normalize_cases(results: Any) -> tuple[List[Dict[str, Any]], Dict[str, Any] | None]:
    """
    Accepts:
      - SuiteResult instance
      - CaseResult instance
      - list[CaseResult]
      - list[dict]
      - dict with {"results": [...], "metadata": {...}}
      - single dict case
    Returns: (cases: list[dict], suite_meta: dict | None)
    """
    # SuiteResult duck-type: has .to_dict() and .results attribute (list)
    if hasattr(results, "to_dict") and hasattr(results, "results") and isinstance(getattr(results, "results"), list):
        suite_dict = results.to_dict()
        return suite_dict.get("results", []), suite_dict.get("metadata")

    # CaseResult duck-type
    if hasattr(results, "to_dict") and not hasattr(results, "results"):
        return [results.to_dict()], None

    if isinstance(results, list):
        if results and hasattr(results[0], "to_dict"):
            return [r.to_dict() for r in results], None
        # assume list[dict]
        return results, None

    if isinstance(results, dict):
        if "results" in results and isinstance(results["results"], list):
            return results["results"], results.get("metadata")
        # single case dict
        return [results], None

    raise TypeError(f"export_results: unsupported results type {type(results)!r}")


def export_results(
    results: Any,
    output_path: str,
    format: str = "json"
) -> str:
    """
    Export suite/case results to JSON or CSV.

    Parameters
    ----------
    results: SuiteResult | CaseResult | list[CaseResult] | list[dict] | dict
        Results object(s) to export. Automatically normalized.
    output_path: str
        Destination file path.
    format: str, default "json"
        "json" or "csv".

    Returns
    -------
    str
        The output_path written.

    CSV flattening:
      - Nested dicts are exploded: evaluator_results_exact_match_score, ...
      - Lists are indexed: evaluator_results_..._dimension_scores_0_raw_score
      - Primitive lists also produce a joined base column (e.g. keywords)
      - Column order: core metadata first, then evaluator_results_*, metadata_*, then remaining
    """
    fmt = format.lower()
    if fmt not in {"json", "csv"}:
        raise ValueError(f"format must be 'json' or 'csv', got {format!r}")

    cases, meta = _normalize_cases(results)

    if fmt == "json":
        payload: Any
        if meta is not None:
            payload = {"metadata": meta, "results": cases}
        else:
            # Preserve suite shape even without metadata
            payload = {"results": cases}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return output_path

    # CSV path
    flat_rows: List[Dict[str, Any]] = [flatten_dict(case) for case in cases]

    # Inject suite-level metadata as metadata_* columns (if present)
    if meta:
        flat_meta = flatten_dict(meta, parent_key="metadata")
        for row in flat_rows:
            row.update(flat_meta)

    # Collect all keys
    all_keys: set[str] = set()
    for row in flat_rows:
        all_keys.update(row.keys())

    # Stable column ordering
    # Core fields (support both singular and plural variants from different runner versions)
    core_priority = [
        "case_id",
        "input_prompt",
        "generated_response",
        "output_text",  # legacy alias
        "expected_text",
        "keywords",
        "latency_ms",
        "latencies_ms",
        "error",
        "errors",
    ]
    # Expand core list with indexed keyword columns keywords_0, keywords_1 ... if present,
    # keeping them immediately after `keywords`
    ordered: List[str] = []
    def push_if_present(key: str):
        if key in all_keys and key not in ordered:
            ordered.append(key)

    for key in core_priority:
        push_if_present(key)

    # Push keywords_N, latency_N, etc., right after their base key
    for base in ["keywords", "latencies_ms", "errors"]:
        indexed = sorted([k for k in all_keys if k.startswith(f"{base}_")])
        for k in indexed:
            push_if_present(k)

    # Next: evaluator_results_*
    eval_keys = sorted([k for k in all_keys if k.startswith("evaluator_results_") or k.startswith("evaluator_results")])
    for k in eval_keys:
        push_if_present(k)

    # Next: scores_* (legacy runner field name)
    score_keys = sorted([k for k in all_keys if k.startswith("scores_")])
    for k in score_keys:
        push_if_present(k)

    # Next: metadata_*
    meta_keys = sorted([k for k in all_keys if k.startswith("metadata_")])
    for k in meta_keys:
        push_if_present(k)

    # Finally: everything else, alphabetically
    remaining = sorted(all_keys - set(ordered))
    ordered.extend(remaining)

    # Write CSV
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        for row in flat_rows:
            # Normalize None -> ""
            writer.writerow({k: ("" if row.get(k) is None else row.get(k, "")) for k in ordered})

    return output_path
