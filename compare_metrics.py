#!/usr/bin/env python3
"""
compare_metrics.py — Compare prompt evaluation metrics against baselines.

Two modes:

1. Single-file mode (backward compatible):
   python compare_metrics.py --latest latest_metrics.json --baseline baseline.json \
       --output comparison.md

   Compares one current run against one baseline, writes a full regression report.

2. Matrix aggregate / leaderboard mode:
   python compare_metrics.py --aggregate \
       --glob "latest_metrics_*.json" \
       --baseline-glob "baseline_*.json" \
       --output leaderboard.md

   Finds all matching current metric files, pairs each with its baseline
   (by variant name), and produces a single Markdown leaderboard table:

   | Variant | Current Score | Baseline Score | Delta | Estimated Cost | Status |
   |---|---|---:|---:|---:|---|
   | v1_concise | 0.842 | 0.830 | +0.012 | $0.00042 | 🟢 improved |
   | v2_detailed | 0.801 | 0.830 | -0.029 | $0.00410 (+850%) | 🔴 regression |

The leaderboard is suitable for embedding directly into a sticky PR comment.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

# Try to import from the local prompt_metrics package for consistent scoring
try:
    from prompt_metrics.monitoring import compare_suites, load_suite_result  # type: ignore
    from prompt_metrics.reports import generate_comparison_report  # type: ignore
    HAS_PM = True
except Exception:
    HAS_PM = False


# ---------------------------------------------------------------------------
# Score extraction (mirrors prompt_metrics.reports._extract_numeric_score)
# ---------------------------------------------------------------------------

def _extract_numeric_score(result: Any) -> float | None:
    if result is None:
        return None
    if isinstance(result, (int, float)):
        return float(result)
    if not isinstance(result, dict):
        return None
    for key in ("score", "overall_score", "accuracy", "f1", "exact_match"):
        if key in result and isinstance(result[key], (int, float)):
            return float(result[key])
    if "percentage" in result and isinstance(result["percentage"], (int, float)):
        return float(result["percentage"]) / 100.0
    for v in result.values():
        if isinstance(v, dict):
            nested = _extract_numeric_score(v)
            if nested is not None:
                return nested
    return None


def extract_suite_score(path: Path, metric: str | None = None) -> float | None:
    """
    Extract an aggregate score from a SuiteResult JSON.

    If metric is given, looks for that evaluator specifically.
    Otherwise aggregates the mean across all evaluators / cases.

    Returns a float in [0, 1] or None if no score found.
    """
    try:
        with path.open() as f:
            raw = json.load(f)
    except Exception as e:
        print(f"Warning: failed to read {path}: {e}", file=sys.stderr)
        return None

    # Direct top-level metric?
    if isinstance(raw, dict) and metric:
        for mkey in (metric, metric.lower()):
            if mkey in raw and isinstance(raw[mkey], (int, float)):
                return float(raw[mkey])

    # Normalize to case list
    cases: list[dict[str, Any]] = []
    if isinstance(raw, list):
        cases = raw
    elif isinstance(raw, dict):
        if "results" in raw and isinstance(raw["results"], list):
            cases = raw["results"]
        else:
            # Single case?
            if "case_id" in raw or "evaluator_results" in raw:
                cases = [raw]

    scores: list[float] = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        eval_results = case.get("evaluator_results", {})
        if not isinstance(eval_results, dict):
            continue
        if metric:
            # Specific evaluator
            if metric in eval_results:
                s = _extract_numeric_score(eval_results[metric])
                if s is not None:
                    scores.append(s)
            continue
        # Aggregate: mean across all evaluators in this case
        case_scores = []
        for ev_out in eval_results.values():
            s = _extract_numeric_score(ev_out)
            if s is not None:
                case_scores.append(s)
        if case_scores:
            scores.append(statistics.mean(case_scores))

    if not scores:
        # Fallback: try top-level score keys
        if isinstance(raw, dict):
            s = _extract_numeric_score(raw)
            if s is not None:
                return s
        return None

    return statistics.mean(scores)


def extract_cost_info(path: Path) -> dict[str, Any] | None:
    """
    Extract token usage / cost info from a SuiteResult JSON.

    Looks for token_usage at:
      - top-level "token_usage"
      - summary.token_usage
      - summary.cost / summary.estimated_cost

    Returns dict with keys: estimated_cost, prompt_tokens, completion_tokens,
    total_tokens, model  — or None if not found.
    """
    try:
        with path.open() as f:
            raw = json.load(f)
    except Exception:
        return None

    # Hunt for token_usage dict in common locations
    tu = None
    if isinstance(raw, dict):
        # Top-level
        if "token_usage" in raw and isinstance(raw["token_usage"], dict):
            tu = raw["token_usage"]
        # summary.token_usage
        elif "summary" in raw and isinstance(raw["summary"], dict):
            summary = raw["summary"]
            if "token_usage" in summary and isinstance(summary["token_usage"], dict):
                tu = summary["token_usage"]
            # Legacy flat cost fields in summary
            elif "estimated_cost" in summary or "cost" in summary:
                tu = {
                    "estimated_cost": summary.get("estimated_cost", summary.get("cost")),
                    "prompt_tokens": summary.get("prompt_tokens"),
                    "completion_tokens": summary.get("completion_tokens"),
                    "total_tokens": summary.get("total_tokens"),
                    "model": summary.get("model"),
                }

    if not tu:
        return None

    # Normalize keys
    cost = tu.get("estimated_cost")
    if cost is None:
        cost = tu.get("cost")
    try:
        cost = float(cost) if cost is not None else None
    except Exception:
        cost = None

    def _int_or_none(v):
        try:
            return int(v) if v is not None else None
        except Exception:
            return None

    return {
        "estimated_cost": cost,
        "prompt_tokens": _int_or_none(tu.get("prompt_tokens")),
        "completion_tokens": _int_or_none(tu.get("completion_tokens")),
        "total_tokens": _int_or_none(tu.get("total_tokens")),
        "model": tu.get("model"),
    }


def variant_name_from_path(path: Path) -> str:
    """
    Derive a human-friendly variant name from a metrics file path.

    Handles:
      latest_metrics_v1_concise.json -> v1_concise
      metrics-v1_concise.json        -> v1_concise
      v1_concise.json                -> v1_concise
      latest_metrics_v1_concise--gpt-4o-mini.json -> v1_concise--gpt-4o-mini
    """
    name = path.stem
    # Strip common prefixes
    name = re.sub(r'^(latest_)?metrics[-_]', '', name)
    name = re.sub(r'^latest_metrics_?', '', name)
    return name or path.stem


def split_prompt_model(variant: str) -> tuple[str, str | None]:
    """
    Split a variant string like "v1_concise--gpt-4o-mini" or
    "v1_concise-gpt-4o-mini" into (prompt, model).

    Returns (prompt, model) where model may be None if not detectable.
    """
    # Common model name fragments to detect
    model_markers = [
        "gpt-4o", "gpt-3.5", "gpt-4", "claude", "gemini", "llama", "mistral",
        "haiku", "sonnet", "opus", "mini", "turbo", "flash", "pro",
    ]
    # Try double-dash separator first (prompt--model)
    if "--" in variant:
        prompt, model_part = variant.split("--", 1)
        return prompt, model_part
    # Try to find a model marker in the string
    vl = variant.lower()
    for marker in model_markers:
        # Look for -marker or _marker boundary
        for sep in ("-", "_"):
            needle = sep + marker
            idx = vl.find(needle)
            if idx > 0:
                prompt = variant[:idx]
                model = variant[idx + 1 :]
                # Heuristic: model part should be reasonably short and
                # contain a known model token
                if len(model) < 40:
                    return prompt, model
    return variant, None


def format_variant_display(prompt: str, model: str | None) -> str:
    """Format 'v1_concise' + 'gpt-4o-mini' → 'v1_concise (gpt-4o-mini)'."""
    if not model:
        return prompt
    # Avoid duplicating if model is already in prompt string
    if model.lower() in prompt.lower():
        return prompt
    return f"{prompt} ({model})"


def find_baseline_for_variant(variant: str, baseline_files: list[Path], baseline_dir: Path | None) -> Path | None:
    """Find the baseline file matching a variant name."""
    # Exact variant substring match
    for bf in baseline_files:
        if variant in bf.stem:
            return bf
    # Generic baseline.json
    if baseline_dir:
        generic = baseline_dir / "baseline.json"
        if generic.exists():
            return generic
    for bf in baseline_files:
        if bf.stem in ("baseline", "baseline_metrics"):
            return bf
    # If only one baseline file exists, use it for all
    if len(baseline_files) == 1:
        return baseline_files[0]
    return None


def classify_status(delta: float | None, threshold: float) -> str:
    if delta is None:
        return "⚪ n/a"
    if delta >= 0.01:
        return "🟢 improved"
    if delta <= -threshold:
        return "🔴 regression"
    if delta < 0:
        return "🟡 dip"
    return "⚪ stable"


# ---------------------------------------------------------------------------
# Single-file comparison (backward compat)
# ---------------------------------------------------------------------------

def run_single_comparison(args: argparse.Namespace) -> int:
    latest_path = Path(args.latest)
    baseline_path = Path(args.baseline)

    if not latest_path.exists():
        print(f"ERROR: {latest_path} not found", file=sys.stderr)
        return 2
    if not baseline_path.exists():
        print(f"ERROR: {baseline_path} not found", file=sys.stderr)
        return 2

    # If prompt_metrics is available, do a full suite comparison
    if HAS_PM:
        try:
            current_suite = load_suite_result(str(latest_path))
            baseline_suite = load_suite_result(str(baseline_path))
            comparison = compare_suites(
                current_suite,
                baseline_suite,
                latency_regression_threshold=1.0 + args.max_latency_pct / 100.0,
                score_drop_threshold=args.min_accuracy_drop,
            )
            generate_comparison_report(
                comparison,
                args.output,
                title=f"Regression Report — {baseline_path.name} vs {latest_path.name}",
            )
            print(f"Wrote {args.output}")
            # Check thresholds for CI gating
            metrics_drift = comparison.get("metrics_drift", {})
            failed = False
            for eval_name, stats in metrics_drift.items():
                mean_delta = stats.get("mean_delta", 0.0)
                if mean_delta < -args.min_accuracy_drop:
                    print(f"FAIL: {eval_name} mean_delta {mean_delta:.4f} < -{args.min_accuracy_drop}", file=sys.stderr)
                    failed = True
            # Latency / cost checks would need runtime/cost aggregation – skipped here
            return 1 if failed else 0
        except Exception as e:
            print(f"prompt_metrics suite comparison failed: {e} — falling back to simple score compare", file=sys.stderr)

    # Fallback: simple score extraction
    current_score = extract_suite_score(latest_path, args.metric)
    baseline_score = extract_suite_score(baseline_path, args.metric)
    if current_score is None or baseline_score is None:
        print(f"ERROR: could not extract '{args.metric or 'overall'}' score from input files", file=sys.stderr)
        return 2

    delta = current_score - baseline_score
    with open(args.output, "w") as f:
        f.write(f"# Comparison: {baseline_path.name} vs {latest_path.name}\n\n")
        f.write(f"| Metric | Baseline | Current | Delta |\n")
        f.write(f"|---|---|---:|---:|\n")
        metric_label = args.metric or "overall_score"
        f.write(f"| {metric_label} | {baseline_score:.4f} | {current_score:.4f} | {delta:+.4f} |\n")

    print(f"{args.metric or 'overall'}: baseline={baseline_score:.4f} current={current_score:.4f} delta={delta:+.4f}")
    print(f"Wrote {args.output}")

    if delta < -args.min_accuracy_drop:
        print(f"FAIL: score drop {delta:+.4f} exceeds threshold -{args.min_accuracy_drop}", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------
# Matrix aggregate / leaderboard
# ---------------------------------------------------------------------------

def run_aggregate(args: argparse.Namespace) -> int:
    # Find current metric files
    current_files: list[Path] = []
    for pattern in args.glob:
        current_files.extend(Path(".").glob(pattern))
    # Deduplicate, sort
    current_files = sorted(set(f for f in current_files if f.is_file()))
    if not current_files:
        print(f"ERROR: no current metric files matched: {args.glob}", file=sys.stderr)
        return 2

    # Find baseline files
    baseline_files: list[Path] = []
    for pattern in args.baseline_glob:
        baseline_files.extend(Path(".").glob(pattern))
    baseline_files = sorted(set(f for f in baseline_files if f.is_file()))

    # Also check baseline_dir
    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else None
    if baseline_dir and baseline_dir.is_dir():
        baseline_files.extend(baseline_dir.glob("*.json"))
        baseline_files = sorted(set(baseline_files))

    print(f"Found {len(current_files)} current metric file(s), {len(baseline_files)} baseline file(s)", file=sys.stderr)

    rows: list[dict[str, Any]] = []
    for cf in current_files:
        variant = variant_name_from_path(cf)
        current_score = extract_suite_score(cf, args.metric)

        # Find matching baseline
        bf = find_baseline_for_variant(variant, baseline_files, baseline_dir)
        baseline_score = extract_suite_score(bf, args.metric) if bf else None

        delta = None if (current_score is None or baseline_score is None) else current_score - baseline_score
        status = classify_status(delta, args.min_accuracy_drop)

        # Cost / token usage – also gives us the model name
        current_cost_info = extract_cost_info(cf)
        baseline_cost_info = extract_cost_info(bf) if bf else None

        def _get_cost(ci):
            return ci.get("estimated_cost") if ci else None

        current_cost = _get_cost(current_cost_info)
        baseline_cost = _get_cost(baseline_cost_info)
        cost_delta = None if (current_cost is None or baseline_cost is None) else current_cost - baseline_cost

        # Split variant into prompt + model for display
        # Prefer model from token_usage JSON (authoritative), fall back to parsing filename
        prompt_name, model_from_filename = split_prompt_model(variant)
        model_from_json = (current_cost_info or {}).get("model") if current_cost_info else None
        model_name = model_from_json or model_from_filename
        display_variant = format_variant_display(prompt_name, model_name)

        rows.append({
            "variant": variant,
            "prompt": prompt_name,
            "model": model_name,
            "display_variant": display_variant,
            "current_score": current_score,
            "baseline_score": baseline_score,
            "delta": delta,
            "status": status,
            "current_file": str(cf),
            "baseline_file": str(bf) if bf else None,
            "current_cost": current_cost,
            "baseline_cost": baseline_cost,
            "cost_delta": cost_delta,
            "current_cost_info": current_cost_info,
            "baseline_cost_info": baseline_cost_info,
        })

    # Sort by current_score descending (best first), Nones last
    rows.sort(key=lambda r: r["current_score"] if r["current_score"] is not None else -1, reverse=True)

    # Helper to format cost
    def fmt_cost(c: float | None) -> str:
        if c is None:
            return "—"
        if c < 0.001:
            return f"${c:.6f}"
        if c < 0.01:
            return f"${c:.5f}"
        if c < 1.0:
            return f"${c:.4f}"
        return f"${c:.2f}"

    def fmt_cost_delta(cd: float | None, base: float | None) -> str:
        if cd is None or base is None or base == 0:
            return ""
        pct = cd / base * 100
        sign = "+" if cd >= 0 else ""
        return f" ({sign}{pct:.0f}%)"

    # Render Markdown leaderboard
    lines: list[str] = []
    if args.title:
        lines.append(f"## {args.title}")
        lines.append("")
    lines.append("| Variant | Current Score | Baseline Score | Delta | Estimated Cost | Status |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for r in rows:
        cs = f"{r['current_score']:.4f}" if r["current_score"] is not None else "—"
        bs = f"{r['baseline_score']:.4f}" if r["baseline_score"] is not None else "—"
        d = f"{r['delta']:+.4f}" if r["delta"] is not None else "—"
        # Cost column: current cost, with baseline delta if available
        cost_str = fmt_cost(r.get("current_cost"))
        if r.get("current_cost") is not None and r.get("baseline_cost") is not None:
            cost_str += fmt_cost_delta(r.get("cost_delta"), r.get("baseline_cost"))
        # Use display_variant (prompt + model) if available, fall back to variant
        variant_label = r.get("display_variant") or r["variant"]
        lines.append(f"| `{variant_label}` | {cs} | {bs} | {d} | {cost_str} | {r['status']} |")
    lines.append("")

    # Summary footer
    improved = sum(1 for r in rows if "improved" in r["status"])
    regressed = sum(1 for r in rows if "regression" in r["status"])
    # Cost summary
    costs = [r["current_cost"] for r in rows if r.get("current_cost") is not None]
    cost_summary = ""
    if costs:
        total_cost = sum(costs)
        avg_cost = total_cost / len(costs)
        cost_summary = f" — avg cost {fmt_cost(avg_cost)}/variant, total {fmt_cost(total_cost)}"
    lines.append(f"<sub>{len(rows)} variant(s) — {improved} improved, {regressed} regression(s){cost_summary}</sub>")
    lines.append("")

    markdown = "\n".join(lines)
    Path(args.output).write_text(markdown)
    print(markdown)
    print(f"\nWrote {args.output}", file=sys.stderr)

    # -----------------------------------------------------------------------
    # Hard failure alert threshold (absolute score floor)
    # -----------------------------------------------------------------------
    hard_failures: list[dict[str, Any]] = []
    alert_threshold = getattr(args, "alert_threshold", None)
    if alert_threshold is not None:
        for r in rows:
            cs = r.get("current_score")
            if cs is not None and cs < alert_threshold:
                hard_failures.append(r)

        alert_output = getattr(args, "alert_output", None)
        if alert_output:
            Path(alert_output).write_text(
                json.dumps(
                    {
                        "threshold": alert_threshold,
                        "metric": args.metric,
                        "failures": [
                            {
                                "variant": r["variant"],
                                "prompt": r.get("prompt"),
                                "model": r.get("model"),
                                "display_variant": r.get("display_variant"),
                                "current_score": r["current_score"],
                                "baseline_score": r["baseline_score"],
                                "delta": r["delta"],
                                "current_file": r["current_file"],
                                "estimated_cost": r.get("current_cost"),
                                "cost_info": r.get("current_cost_info"),
                            }
                            for r in hard_failures
                        ],
                    },
                    indent=2,
                )
            )
            print(f"Wrote alert report to {alert_output} ({len(hard_failures)} failure(s))", file=sys.stderr)

        if hard_failures:
            print(
                f"ALERT: {len(hard_failures)} variant(s) below hard threshold {alert_threshold}: "
                + ", ".join(f"{r['variant']}={r['current_score']:.4f}" for r in hard_failures),
                file=sys.stderr,
            )

    # CI gate: fail if any regression exceeds threshold
    if args.fail_on_regression and regressed > 0:
        print(f"FAIL: {regressed} regression(s) detected", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Compare prompt evaluation metrics")
    # Mode switch
    ap.add_argument("--aggregate", action="store_true", help="Matrix leaderboard mode: scan multiple metric files")

    # Single-file mode args
    ap.add_argument("--latest", help="Current metrics JSON (single-file mode)")
    ap.add_argument("--baseline", help="Baseline metrics JSON (single-file mode)")

    # Aggregate mode args
    ap.add_argument("--glob", action="append", default=[],
                    help="Glob for current metric files (aggregate mode, repeatable). "
                         "Default: latest_metrics_*.json, metrics-*.json, *_metrics.json")
    ap.add_argument("--baseline-glob", action="append", default=[],
                    help="Glob for baseline metric files (aggregate mode, repeatable)")
    ap.add_argument("--baseline-dir", help="Directory containing baseline JSONs (aggregate mode)")

    # Shared
    ap.add_argument("--metric", default=None,
                    help="Specific evaluator/metric key to score (default: mean across all evaluators)")
    ap.add_argument("--output", "-o", default="comparison.md",
                    help="Output Markdown file (default: comparison.md)")
    ap.add_argument("--title", default="Prompt Variant Leaderboard",
                    help="Table title for aggregate mode")

    # Thresholds (for CI gating / status classification)
    ap.add_argument("--max-latency-pct", type=float, default=10.0,
                    help="Max allowed latency increase %% (single-file mode)")
    ap.add_argument("--min-accuracy-drop", type=float, default=0.02,
                    help="Score drop threshold for regression status (default: 0.02)")
    ap.add_argument("--max-cost-pct", type=float, default=15.0,
                    help="Max allowed cost increase %% (currently informational)")
    ap.add_argument("--fail-on-regression", action="store_true",
                    help="Exit 1 if any regression is detected")

    # Hard-failure alerting (aggregate mode)
    ap.add_argument("--alert-threshold", type=float, default=None,
                    help="Absolute score floor — variants below this trigger a hard-failure alert "
                         "(aggregate mode only, e.g. 0.5)")
    ap.add_argument("--alert-output", default=None,
                    help="JSON file to write hard-failure alerts to (aggregate mode only)")

    args = ap.parse_args()

    if args.aggregate:
        # Default globs if none supplied
        if not args.glob:
            args.glob = [
                "latest_metrics_*.json",
                "metrics-*.json",
                "metrics_*.json",
                "*_metrics.json",
            ]
        if not args.baseline_glob:
            args.baseline_glob = [
                "baseline_*.json",
                "baseline.json",
            ]
        return run_aggregate(args)
    else:
        # Single-file mode — require --latest / --baseline
        if not args.latest:
            ap.error("--latest is required in single-file mode (or use --aggregate)")
        if not args.baseline:
            ap.error("--baseline is required in single-file mode (or use --aggregate)")
        return run_single_comparison(args)


if __name__ == "__main__":
    sys.exit(main())
