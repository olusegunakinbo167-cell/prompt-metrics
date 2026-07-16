# src/prompt_metrics/reports.py
"""
Markdown report generation for experiment suite results.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any


# ---------------------------------------------------------------------------
# Score extraction helpers
# ---------------------------------------------------------------------------

def _extract_numeric_score(result: Any) -> float | None:
    """
    Extract a normalised [0, 1] score from an evaluator result dict.

    Tries common score keys in order:
      score, overall_score, percentage (/100→normalised), accuracy, f1
    Recursively searches one level deep for nested score fields.
    Returns None if no numeric score is found.
    """
    if result is None:
        return None
    if isinstance(result, (int, float)):
        return float(result)

    if not isinstance(result, dict):
        return None

    # Direct keys — try most specific first
    score_keys = [
        "score",
        "overall_score",
        "accuracy",
        "f1",
        "exact_match",
    ]
    for key in score_keys:
        if key in result and isinstance(result[key], (int, float)):
            return float(result[key])

    # Percentage → normalise to 0-1
    if "percentage" in result and isinstance(result["percentage"], (int, float)):
        return float(result["percentage"]) / 100.0

    # One-level recursive search (for wrapped/legacy formats)
    for v in result.values():
        if isinstance(v, dict):
            nested = _extract_numeric_score(v)
            if nested is not None:
                return nested

    return None


def _format_score_badge(score: float | None, width: int = 12) -> str:
    """Render a score as a visual badge: `🟢 0.850` / `🟡 0.620` / `🔴 0.310` / `—`."""
    if score is None:
        return "` — `" 
    if score >= 0.8:
        icon = "🟢"
    elif score >= 0.5:
        icon = "🟡"
    else:
        icon = "🔴"
    return f"`{icon} {score:.3f}`"


def _truncate(text: str, max_len: int = 300) -> str:
    """Truncate long text with an ellipsis, preserving readability."""
    text = text.replace("\r", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _md_escape(text: str) -> str:
    """Escape pipe characters so text is safe inside Markdown tables."""
    return text.replace("|", r"\|").replace("\n", " ").strip()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate_evaluator_stats(suite_result: Any) -> dict[str, dict[str, Any]]:
    """
    Compute per-evaluator aggregate statistics across all cases.

    Returns:
        {evaluator_name: {
            "count": n,
            "scores": [float, ...],
            "mean": float,
            "median": float,
            "stdev": float,
            "min": float,
            "max": float,
        }}
    """
    # Normalise SuiteResult → list of case dicts
    if hasattr(suite_result, "results"):
        cases = suite_result.results
    else:
        cases = suite_result

    per_eval_scores: dict[str, list[float]] = defaultdict(list)

    for case in cases:
        # CaseResult object or dict
        if hasattr(case, "evaluator_results"):
            eval_results = case.evaluator_results
        elif isinstance(case, dict):
            eval_results = case.get("evaluator_results", {})
        else:
            continue

        for eval_name, eval_output in eval_results.items():
            score = _extract_numeric_score(eval_output)
            if score is not None:
                per_eval_scores[eval_name].append(score)

    stats: dict[str, dict[str, Any]] = {}
    for eval_name, scores in per_eval_scores.items():
        if not scores:
            stats[eval_name] = {"count": 0}
            continue
        stats[eval_name] = {
            "count": len(scores),
            "scores": scores,
            "mean": statistics.mean(scores),
            "median": statistics.median(scores),
            "stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "min": min(scores),
            "max": max(scores),
        }

    return stats


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def generate_markdown_report(
    suite_result: Any,
    output_path: str,
    *,
    title: str = "Experiment Report",
    truncate_response: int = 400,
    truncate_prompt: int = 300,
) -> str:
    """
    Generate a human-readable Markdown report from a SuiteResult.

    Args:
        suite_result: SuiteResult object, or a dict/list compatible with one.
        output_path: Destination .md file path.
        title: Report title (H1).
        truncate_response: Max chars to show per generated_response in detail table.
        truncate_prompt: Max chars to show per input_prompt in detail table.

    Returns:
        The output_path that was written.

    Report sections:
        1. Summary — total cases, success/fail, runtime, avg latency
        2. Evaluator Breakdown — mean/median/min/max per evaluator
        3. Test Cases — prompt, response, and per-evaluator score badges
    """
    # ---- Normalise SuiteResult ----
    if hasattr(suite_result, "results"):
        results = suite_result.results
        total_cases = getattr(suite_result, "total_cases", len(results))
        successful_cases = getattr(suite_result, "successful_cases", 0)
        failed_cases = getattr(suite_result, "failed_cases", 0)
        total_runtime_s = getattr(suite_result, "total_runtime_s", 0.0)
        evaluator_names = getattr(suite_result, "evaluator_names", [])
    else:
        # Raw list/dict fallback
        results = suite_result if isinstance(suite_result, list) else [suite_result]
        total_cases = len(results)
        successful_cases = total_cases
        failed_cases = 0
        total_runtime_s = 0.0
        evaluator_names = []

    # Helper to safely extract fields from CaseResult objects or dicts
    def get_field(case: Any, name: str, default: Any = None) -> Any:
        if hasattr(case, name):
            return getattr(case, name)
        if isinstance(case, dict):
            return case.get(name, default)
        return default

    # ---- Compute aggregates ----
    latencies = [
        get_field(c, "latency_ms", 0.0) or 0.0
        for c in results
    ]
    avg_latency = statistics.mean(latencies) if latencies else 0.0

    eval_stats = _aggregate_evaluator_stats(results)

    if not evaluator_names and eval_stats:
        evaluator_names = sorted(eval_stats.keys())

    # ---- Build Markdown ----
    lines: list[str] = []

    # 1. Summary
    lines.append(f"# {title}")
    lines.append("")
    success_rate = (successful_cases / total_cases * 100) if total_cases else 0.0
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Total test cases | {total_cases} |")
    lines.append(f"| Successful | {successful_cases} |")
    lines.append(f"| Failed | {failed_cases} |")
    lines.append(f"| Success rate | {success_rate:.1f}% |")
    lines.append(f"| Total runtime | {total_runtime_s:.2f}s |")
    lines.append(f"| Avg latency | {avg_latency:.1f}ms |")
    lines.append("")

    # 2. Evaluator Breakdown
    lines.append("## Evaluator Breakdown")
    lines.append("")
    if eval_stats:
        lines.append("| Evaluator | Mean | Median | Min | Max | Stdev | N |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for eval_name in sorted(eval_stats.keys()):
            s = eval_stats[eval_name]
            if s.get("count", 0) == 0:
                lines.append(f"| `{_md_escape(eval_name)}` | — | — | — | — | — | 0 |")
                continue
            lines.append(
                f"| `{_md_escape(eval_name)}` "
                f"| {s['mean']:.3f} "
                f"| {s['median']:.3f} "
                f"| {s['min']:.3f} "
                f"| {s['max']:.3f} "
                f"| {s['stdev']:.3f} "
                f"| {s['count']} |"
            )
    else:
        lines.append("_No numeric evaluator scores found._")
    lines.append("")

    # 3. Test Cases (detailed)
    lines.append("## Test Cases")
    lines.append("")

    # Per-case detailed blocks — far more readable than a giant table
    # for long prompts/responses
    for i, case in enumerate(results, 1):
        case_id = get_field(case, "case_id", f"case_{i:03d}")
        input_prompt = get_field(case, "input_prompt", "")
        generated_response = get_field(case, "generated_response", "")
        expected_text = get_field(case, "expected_text")
        keywords = get_field(case, "keywords")
        error = get_field(case, "error")

        eval_results = get_field(case, "evaluator_results", {})
        if not isinstance(eval_results, dict):
            eval_results = {}

        lines.append(f"### {i}. `{case_id}`")
        lines.append("")

        if error:
            lines.append(f"> ⚠️ **Error:** `{_md_escape(error.splitlines()[0])}`")
            lines.append("")

        # Evaluator score badges — inline, scannable
        if eval_results:
            badges = []
            for eval_name in sorted(eval_results.keys()):
                score = _extract_numeric_score(eval_results[eval_name])
                badge = _format_score_badge(score)
                badges.append(f"**{eval_name}**: {badge}")
            lines.append(" | ".join(badges))
            lines.append("")

        # Prompt
        lines.append("**Prompt:**")
        lines.append("```")
        lines.append(_truncate(str(input_prompt), truncate_prompt))
        lines.append("```")
        lines.append("")

        # Generated response
        lines.append("**Response:**")
        lines.append("```")
        lines.append(_truncate(str(generated_response), truncate_response))
        lines.append("```")

        # Expected / keywords (compact, only if present)
        meta_bits: list[str] = []
        if expected_text:
            meta_bits.append(
                f"**Expected:** `{_md_escape(_truncate(str(expected_text), 120))}`"
            )
        if keywords:
            kw_str = ", ".join(f"`{_md_escape(str(k))}`" for k in keywords[:8])
            if len(keywords) > 8:
                kw_str += f" … +{len(keywords) - 8} more"
            meta_bits.append(f"**Keywords:** {kw_str}")
        if meta_bits:
            lines.append("")
            lines.append(" | ".join(meta_bits))

        lines.append("")
        lines.append("---")
        lines.append("")

    # ---- Write file ----
    markdown = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    return output_path


# ---------------------------------------------------------------------------
# Comparison / regression reporting
# ---------------------------------------------------------------------------

def _fmt_delta(delta: float | None, *, signed: bool = True) -> str:
    """Format a score delta with sign and color emoji."""
    if delta is None:
        return "—"
    sign = "+" if (signed and delta >= 0) else ""
    if delta >= 0.05:
        icon = "🟢"
    elif delta >= -0.01:
        icon = "⚪"
    elif delta >= -0.15:
        icon = "🟡"
    else:
        icon = "🔴"
    return f"{icon} {sign}{delta:.3f}"


def _fmt_score(score: float | None) -> str:
    """Format a score value."""
    if score is None:
        return "—"
    return f"{score:.3f}"


def generate_comparison_report(
    comparison_result: dict[str, Any],
    output_path: str,
    *,
    title: str = "Regression Report — Baseline vs Current",
    baseline_label: str = "Baseline",
    current_label: str = "Current",
) -> str:
    """
    Generate a Markdown comparison report from a compare_suites() result.

    Args:
        comparison_result: Output dict from compare_suites().
        output_path: Destination .md file path.
        title: Report title (H1).
        baseline_label: Label for the baseline run in tables.
        current_label: Label for the current run in tables.

    Returns:
        The output_path that was written.
    """
    summary = comparison_result.get("summary", {})
    metrics_drift = comparison_result.get("metrics_drift", {})
    score_drops = comparison_result.get("score_drops", [])
    latency_regressions = comparison_result.get("latency_regressions", [])
    per_case = comparison_result.get("per_case", {})

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")

    current_cases = summary.get("current_cases", 0)
    baseline_cases = summary.get("baseline_cases", 0)
    common_cases = summary.get("common_cases", 0)
    new_cases = summary.get("new_cases", [])
    missing_cases = summary.get("missing_cases", [])
    total_score_drops = summary.get("total_score_drops", len(score_drops))
    total_latency_regressions = summary.get("total_latency_regressions", len(latency_regressions))
    score_threshold = summary.get("score_drop_threshold", 0.15)
    latency_threshold = summary.get("latency_regression_threshold", 1.2)

    has_regressions = total_score_drops > 0 or total_latency_regressions > 0
    if has_regressions:
        lines.append(f"> ⚠️ **Regressions detected:** {total_score_drops} score drop(s), {total_latency_regressions} latency regression(s)")
    else:
        lines.append("> ✅ **No regressions detected** — all metrics stable or improved")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| {baseline_label} cases | {baseline_cases} |")
    lines.append(f"| {current_label} cases | {current_cases} |")
    lines.append(f"| Common cases | {common_cases} |")
    lines.append(f"| New cases | {len(new_cases)} |")
    lines.append(f"| Missing cases | {len(missing_cases)} |")
    lines.append(f"| Score drops (>{score_threshold}) | {total_score_drops} |")
    lines.append(f"| Latency regressions (>{(latency_threshold-1)*100:.0f}%) | {total_latency_regressions} |")
    lines.append("")

    if new_cases:
        lines.append(f"**New cases:** " + ", ".join(f"`{c}`" for c in new_cases[:10]) + (f" … +{len(new_cases)-10} more" if len(new_cases) > 10 else ""))
        lines.append("")
    if missing_cases:
        lines.append(f"**Missing cases:** " + ", ".join(f"`{c}`" for c in missing_cases[:10]) + (f" … +{len(missing_cases)-10} more" if len(missing_cases) > 10 else ""))
        lines.append("")

    # Metrics Drift
    lines.append("## Metrics Drift")
    lines.append("")
    if metrics_drift:
        lines.append("| Evaluator | Mean Δ | Median Δ | Min Δ | Max Δ | Stdev | Improved | Regressed | Unchanged | N |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for eval_name in sorted(metrics_drift.keys()):
            s = metrics_drift[eval_name]
            lines.append(
                f"| `{eval_name}` | {_fmt_delta(s.get('mean_delta', 0.0))} | {_fmt_delta(s.get('median_delta', 0.0))} | "
                f"{_fmt_delta(s.get('min_delta', 0.0))} | {_fmt_delta(s.get('max_delta', 0.0))} | "
                f"{s.get('stdev_delta', 0.0):.3f} | {s.get('improved', 0)} | {s.get('regressed', 0)} | "
                f"{s.get('unchanged', 0)} | {s.get('count', 0)} |"
            )
    else:
        lines.append("_No comparable evaluator scores found._")
    lines.append("")

    # Score Drops
    lines.append("## Score Drops")
    lines.append("")
    if score_drops:
        lines.append(f"_Cases where score dropped by more than {score_threshold}:_")
        lines.append("")
        lines.append(f"| Case | Evaluator | {baseline_label} | {current_label} | Δ |")
        lines.append("|---|---|---:|---:|---:|")
        for drop in score_drops[:50]:
            lines.append(
                f"| `{drop['case_id']}` | `{drop['evaluator']}` | {_fmt_score(drop['baseline_score'])} | "
                f"{_fmt_score(drop['current_score'])} | {_fmt_delta(drop['delta'], signed=True)} |"
            )
        if len(score_drops) > 50:
            lines.append(f"| … | … | … | … | _{len(score_drops) - 50} more_ |")
    else:
        lines.append(f"_No score drops exceeding the {score_threshold} threshold._ ✅")
    lines.append("")

    # Latency Regressions
    lines.append("## Latency Regressions")
    lines.append("")
    if latency_regressions:
        pct_thresh = (latency_threshold - 1.0) * 100
        lines.append(f"_Cases where latency increased by more than {pct_thresh:.0f}%:_")
        lines.append("")
        lines.append(f"| Case | {baseline_label} (ms) | {current_label} (ms) | Δ (ms) | Factor |")
        lines.append("|---|---:|---:|---:|---:|")
        for reg in latency_regressions[:50]:
            factor = reg["factor"]
            icon = "🔴" if factor >= 2.0 else "🟡" if factor >= 1.5 else "⚪"
            lines.append(
                f"| `{reg['case_id']}` | {reg['baseline_ms']:.1f} | {reg['current_ms']:.1f} | "
                f"{reg['delta_ms']:+.1f} | {icon} {factor:.2f}× |"
            )
        if len(latency_regressions) > 50:
            lines.append(f"| … | … | … | … | _{len(latency_regressions) - 50} more_ |")
    else:
        pct_thresh = (latency_threshold - 1.0) * 100
        lines.append(f"_No latency regressions exceeding the {pct_thresh:.0f}% threshold._ ✅")
    lines.append("")

    # Per-Case Comparison
    lines.append("## Per-Case Comparison")
    lines.append("")
    all_evals: set[str] = set()
    for case_data in per_case.values():
        all_evals.update(case_data.get("evaluators", {}).keys())
    all_evals = sorted(all_evals)

    if not per_case:
        lines.append("_No cases to compare._")
    elif all_evals:
        eval_headers = " | ".join(f"{e}" for e in all_evals)
        lines.append(f"| Case | Latency | {eval_headers} |")
        lines.append("|---|---:|" + "|".join([":---:"] * len(all_evals)) + "|")
        for case_id in sorted(per_case.keys()):
            cd = per_case[case_id]
            lat_factor = cd.get("latency_factor")
            if lat_factor is None:
                lat_cell = "—"
            elif cd.get("latency_regression"):
                lat_cell = f"🔴 {lat_factor:.2f}×"
            elif lat_factor > 1.05:
                lat_cell = f"🟡 {lat_factor:.2f}×"
            else:
                lat_cell = f"{lat_factor:.2f}×"
            eval_cells = []
            for ev in all_evals:
                ed = cd.get("evaluators", {}).get(ev, {})
                base = ed.get("baseline")
                curr = ed.get("current")
                delta = ed.get("delta")
                score_drop = ed.get("score_drop", False)
                if base is None and curr is None:
                    eval_cells.append("—")
                elif base is None:
                    eval_cells.append(f"— → {_fmt_score(curr)}")
                elif curr is None:
                    eval_cells.append(f"{_fmt_score(base)} → —")
                else:
                    d_str = f" ({delta:+.3f})" if delta is not None else ""
                    cell = f"{base:.3f} → {curr:.3f}{d_str}"
                    if score_drop:
                        cell = "🔴 " + cell
                    elif delta is not None and delta > 0.05:
                        cell = "🟢 " + cell
                    eval_cells.append(cell)
            case_label = f"`{case_id}`"
            if cd.get("is_new"):
                case_label += " 🆕"
            if cd.get("is_missing"):
                case_label += " ⚠️ missing"
            lines.append(f"| {case_label} | {lat_cell} | " + " | ".join(eval_cells) + " |")
    lines.append("")
    markdown = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    return output_path


__all__ = ["generate_markdown_report", "generate_comparison_report"]
