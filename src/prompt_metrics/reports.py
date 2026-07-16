from __future__ import annotations
import json
from typing import Any, Dict, List, Tuple

def _flatten_dict(d: Any, parent_key: str = "", sep: str = "_") -> Dict[str, Any]:
    """Local flatten helper to avoid import cycle with export.py"""
    items: Dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.update(_flatten_dict(v, new_key, sep=sep))
        return items
    if isinstance(d, list):
        for i, v in enumerate(d):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.update(_flatten_dict(v, new_key, sep=sep))
        return items
    if parent_key:
        items[parent_key] = d
    return items


def _normalize_suite(suite_result: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return (cases: list[dict], meta: dict)"""
    if hasattr(suite_result, "to_dict"):
        suite_dict = suite_result.to_dict()
        return suite_dict.get("results", []), suite_dict.get("metadata", {})
    if isinstance(suite_result, dict):
        if "results" in suite_result:
            return suite_result.get("results", []), suite_result.get("metadata", {})
        return [suite_result], {}
    if isinstance(suite_result, list):
        return suite_result, {}
    raise TypeError(f"generate_markdown_report: unsupported suite_result type {type(suite_result)!r}")


def _get_case_latency_ms(case: Dict[str, Any]) -> int:
    # Prefer summed latencies_ms dict
    latencies = case.get("latencies_ms")
    if isinstance(latencies, dict):
        try:
            return int(sum(v for v in latencies.values() if isinstance(v, (int, float))))
        except Exception:
            pass
    # Fall back to singular latency_ms
    v = case.get("latency_ms")
    if isinstance(v, (int, float)):
        return int(v)
    return 0


def _get_case_errors(case: Dict[str, Any]) -> List[Any]:
    errors = case.get("errors", [])
    if isinstance(errors, list):
        return errors
    # legacy singular error field
    err = case.get("error")
    return [err] if err else []


def _score_badge(mean_score: float | None, has_errors: bool) -> str:
    if mean_score is None:
        return "🔴" if has_errors else "🟢"
    if mean_score >= 0.8:
        return "🟢"
    if mean_score >= 0.5:
        return "🟡"
    return "🔴"


def generate_markdown_report(
    suite_result: Any,
    output_path: str,
    title: str = "Experiment Report"
) -> str:
    """
    Generate a GitHub-flavored Markdown report from a SuiteResult.

    Parameters
    ----------
    suite_result : SuiteResult | dict
        SuiteResult instance or dict with {"results": [...], "metadata": {...}}
    output_path : str
        Destination .md file
    title : str, default "Experiment Report"
        Report title

    Returns
    -------
    str
        output_path written
    """
    cases, meta = _normalize_suite(suite_result)
    total_cases = len(cases)

    # Success / fail counts
    success_count = meta.get("success_count") if isinstance(meta, dict) else None
    fail_count = meta.get("fail_count") if isinstance(meta, dict) else None

    case_latencies = [_get_case_latency_ms(c) for c in cases]
    avg_latency = sum(case_latencies) / total_cases if total_cases else 0

    # Compute per-case error status if counts not provided
    case_error_flags = [bool(_get_case_errors(c)) for c in cases]
    if success_count is None:
        success_count = sum(1 for e in case_error_flags if not e)
    if fail_count is None:
        fail_count = sum(1 for e in case_error_flags if e)

    # Evaluator breakdown – discover all numeric metrics under evaluator_results
    evaluator_names = meta.get("evaluator_names", []) if isinstance(meta, dict) else []
    metric_values: Dict[str, List[float]] = {}
    for case in cases:
        ev_results = case.get("evaluator_results", case.get("scores", {}))
        if not isinstance(ev_results, dict):
            continue
        flat = _flatten_dict(ev_results)
        for k, v in flat.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                metric_values.setdefault(k, []).append(float(v))

    metric_stats = {}
    for k, vals in metric_values.items():
        if vals:
            metric_stats[k] = {
                "mean": sum(vals) / len(vals),
                "min": min(vals),
                "max": max(vals),
                "n": len(vals),
            }

    # Build markdown
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total cases:** {total_cases}")
    lines.append(f"- **Success:** {success_count}")
    lines.append(f"- **Fail:** {fail_count}")
    lines.append(f"- **Average latency:** {avg_latency:.1f} ms")
    if evaluator_names:
        lines.append(f"- **Evaluators:** {', '.join(evaluator_names)}")
    total_runtime_ms = meta.get("total_runtime_ms") if isinstance(meta, dict) else None
    if total_runtime_ms is not None:
        lines.append(f"- **Total runtime:** {total_runtime_ms} ms")
    lines.append("")

    # Evaluator breakdown
    lines.append("## Evaluator Breakdown")
    lines.append("")
    if metric_stats:
        lines.append("| Metric | Mean | Min | Max | N |")
        lines.append("|---|---|---|---|---|")
        for metric_key in sorted(metric_stats.keys()):
            st = metric_stats[metric_key]
            lines.append(
                f"| `{metric_key}` | {st['mean']:.4f} | {st['min']:.4f} | {st['max']:.4f} | {st['n']} |"
            )
    else:
        lines.append("_No numeric evaluator metrics found._")
    lines.append("")

    # Detailed test cases
    lines.append("## Test Cases")
    lines.append("")
    for idx, case in enumerate(cases, 1):
        case_id = case.get("case_id") or f"case_{idx:03d}"
        ev_results = case.get("evaluator_results", case.get("scores", {}))
        flat_scores = _flatten_dict(ev_results) if isinstance(ev_results, dict) else {}
        # collect score-like metrics for badge
        score_vals = [
            float(v) for k, v in flat_scores.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
            and (k == "score" or k.endswith("_score"))
        ]
        mean_score = sum(score_vals) / len(score_vals) if score_vals else None
        has_errors = bool(_get_case_errors(case))
        badge = _score_badge(mean_score, has_errors)

        lines.append(f"### {case_id} {badge}")
        lines.append("")

        # Prompt
        prompt = case.get("input_prompt", "")
        lines.append("**Prompt:**")
        lines.append("```")
        lines.append(str(prompt))
        lines.append("```")
        lines.append("")

        # Response
        response = case.get("generated_response", case.get("output_text", ""))
        lines.append("**Response:**")
        lines.append("```")
        lines.append(str(response))
        lines.append("```")
        lines.append("")

        # Expected / Keywords
        expected = case.get("expected_text")
        if expected:
            lines.append(f"**Expected:** `{expected}`")
            lines.append("")
        keywords = case.get("keywords")
        if keywords:
            if isinstance(keywords, list):
                kw_str = ", ".join(str(k) for k in keywords)
            else:
                kw_str = str(keywords)
            if kw_str:
                lines.append(f"**Keywords:** {kw_str}")
                lines.append("")

        # Latency
        latency_ms = _get_case_latency_ms(case)
        lines.append(f"**Latency:** {latency_ms} ms")
        lines.append("")

        # Scores
        if flat_scores:
            lines.append("**Scores:**")
            for sk in sorted(flat_scores.keys()):
                sv = flat_scores[sk]
                if isinstance(sv, (int, float)) and not isinstance(sv, bool):
                    lines.append(f"- `{sk}`: {sv:.4f}")
                else:
                    # truncate long non-numeric values
                    sv_str = json.dumps(sv, ensure_ascii=False)
                    if len(sv_str) > 120:
                        sv_str = sv_str[:117] + "..."
                    lines.append(f"- `{sk}`: {sv_str}")
            lines.append("")

        # Errors
        errors = _get_case_errors(case)
        if errors:
            lines.append("**Errors:**")
            for e in errors:
                if isinstance(e, dict):
                    stage = e.get("stage", "unknown")
                    err_msg = e.get("error", str(e))
                    lines.append(f"- `{stage}`: {err_msg}")
                else:
                    lines.append(f"- {e}")
            lines.append("")

        lines.append("---")
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return output_path
