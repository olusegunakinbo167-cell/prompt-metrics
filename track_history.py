#!/usr/bin/env python3
"""
track_history.py — Append prompt-metrics evaluation results to a rolling time-series history.

Reads metrics from latest_metrics.json and appends a timestamped entry
containing the current git commit SHA, branch name, and metric values
to history.json. Maintains a maximum of 100 historical entries.

Reporting:
  python track_history.py --report
    Reads history.json and prints a markdown trend table with rolling
    averages (5 / 10 / 50 runs) and ASCII sparklines.  If the most recent
    run falls significantly below the 10-run moving average, a drift
    warning is printed to stderr.

Usage:
  python track_history.py
  python track_history.py --input latest_metrics.json --output history.json --max-entries 100
  python track_history.py --report
  python track_history.py --report --output history.json --drift-threshold 0.05
"""

import argparse
import json
import math
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_INPUT = "latest_metrics.json"
DEFAULT_OUTPUT = "history.json"
DEFAULT_MAX_ENTRIES = 100

# Sparkline blocks (8 levels)
_SPARK = "▁▂▃▄▅▆▇█"


def get_git_sha() -> str | None:
    """Return the current git commit SHA, or None if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_branch() -> str | None:
    """Return the current git branch name, or None if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        branch = result.stdout.strip()
        return branch if branch != "HEAD" else None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def load_json(path: Path) -> Any:
    """Load JSON from path, raising a clear error on failure."""
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e


def load_history(path: Path) -> list[dict[str, Any]]:
    """Load existing history, or return an empty list if the file does not exist."""
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in history file {path}: {e}") from e

    if not isinstance(data, list):
        raise ValueError(f"History file {path} must contain a JSON list, got {type(data).__name__}")
    return data


def save_history(path: Path, history: list[dict[str, Any]]) -> None:
    """Write history atomically to avoid corruption on interruption."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _flatten_numeric(obj: Any, prefix: str = "") -> dict[str, float]:
    """Flatten a nested metrics object to {dotted_key: float} for numeric leaves only."""
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten_numeric(v, key))
    elif isinstance(obj, list):
        # skip lists – too ambiguous for trend reporting
        return {}
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if math.isfinite(float(obj)):
            out[prefix] = float(obj)
    return out


def _sparkline(values: list[float], width: int = 20) -> str:
    """Return an ASCII sparkline for the given values."""
    if not values:
        return "─" * width
    # Take the most recent `width` points
    vals = values[-width:]
    v_min = min(vals)
    v_max = max(vals)
    if v_max == v_min:
        # flat line
        return _SPARK[3] * len(vals)
    chars = []
    for v in vals:
        idx = int(round((v - v_min) / (v_max - v_min) * 7))
        idx = max(0, min(7, idx))
        chars.append(_SPARK[idx])
    return "".join(chars)


def _fmt(v: float | None, precision: int = 4) -> str:
    if v is None:
        return "—"
    # adaptive precision: trim trailing zeros
    s = f"{v:.{precision}f}"
    s = s.rstrip("0").rstrip(".")
    if s == "-0":
        s = "0"
    return s if s else "0"


def generate_report(
    history: list[dict[str, Any]],
    drift_threshold: float = 0.05,
    lower_is_better_metrics: set[str] | None = None,
) -> tuple[str, list[str]]:
    """
    Generate a trend report from history.

    Returns (markdown_report, drift_warnings)
    """
    if not history:
        return "# History Trend Report\n\n_No entries in history yet._\n", []

    lower_is_better_metrics = lower_is_better_metrics or set()

    # Extract all numeric metric series
    series: dict[str, list[tuple[int, float]]] = {}  # key -> [(history_idx, value), ...]
    for idx, entry in enumerate(history):
        metrics_obj = entry.get("metrics", {})
        flat = _flatten_numeric(metrics_obj)
        for k, v in flat.items():
            series.setdefault(k, []).append((idx, v))

    if not series:
        return "# History Trend Report\n\n_No numeric metrics found in history._\n", []

    metric_keys = sorted(series.keys())

    # Build markdown table
    lines = []
    lines.append("# History Trend Report")
    lines.append("")
    lines.append(f"Runs in history: **{len(history)}**")
    last_entry = history[-1]
    ts = last_entry.get("timestamp", "n/a")
    commit = last_entry.get("git_commit", "") or "n/a"
    commit_short = commit[:7] if commit != "n/a" else "n/a"
    branch = last_entry.get("git_branch", "") or "n/a"
    lines.append(f"Latest: `{ts}` · `{commit_short}` · `{branch}`")
    lines.append("")
    lines.append("| Metric | Current | Avg-5 | Avg-10 | Avg-50 | Trend (last 20) |")
    lines.append("|--------|---------|-------|--------|--------|-----------------|")

    drift_warnings: list[str] = []

    for key in metric_keys:
        points = series[key]
        # Build a dense value list aligned to history indices, forward-filling gaps as NaN
        value_by_idx = {i: v for i, v in points}
        values_ordered = [value_by_idx.get(i) for i in range(len(history))]
        # For rolling averages, use only non-None values, most recent first
        present_values = [v for v in reversed(values_ordered) if v is not None]

        if not present_values:
            continue

        current = present_values[0] if present_values else None

        def rolling_avg(n: int) -> float | None:
            vals = present_values[:n]
            return statistics.fmean(vals) if vals else None

        avg5 = rolling_avg(5)
        avg10 = rolling_avg(10)
        avg50 = rolling_avg(50)

        # Trend sparkline over all present values (chronological)
        trend_vals = [v for v in values_ordered if v is not None]
        trend = _sparkline(trend_vals, width=20)

        lines.append(
            f"| `{key}` | {_fmt(current)} | {_fmt(avg5)} | {_fmt(avg10)} | {_fmt(avg50)} | `{trend}` |"
        )

        # --- Drift detection ---
        # Compare current run against the 10-run moving average of PRIOR runs
        # (i.e. exclude the current run itself)
        prior_values = present_values[1:11]  # up to 10 prior points
        if current is not None and prior_values:
            mean_prior = statistics.fmean(prior_values)
            std_prior = statistics.stdev(prior_values) if len(prior_values) >= 2 else 0.0
            lower_is_better = key in lower_is_better_metrics

            drifted = False
            reason = ""

            if lower_is_better:
                # drift = increase
                rel_increase = (current - mean_prior) / abs(mean_prior) if mean_prior != 0 else float("inf") if current > 0 else 0
                if rel_increase > drift_threshold:
                    drifted = True
                    reason = f"{rel_increase*100:+.1f}% above 10-run mean ({_fmt(mean_prior)})"
                elif std_prior > 0 and current > mean_prior + 1.5 * std_prior:
                    drifted = True
                    reason = f"{(current-mean_prior)/std_prior:.1f}σ above mean"
            else:
                # drift = decrease (higher is better – standard for scores)
                rel_drop = (mean_prior - current) / abs(mean_prior) if mean_prior != 0 else 0.0
                if rel_drop > drift_threshold:
                    drifted = True
                    reason = f"{rel_drop*100:.1f}% below 10-run mean ({_fmt(mean_prior)})"
                elif std_prior > 0 and current < mean_prior - 1.5 * std_prior:
                    drifted = True
                    z = (current - mean_prior) / std_prior if std_prior else 0
                    reason = f"{z:.1f}σ below mean ({_fmt(mean_prior)})"

            if drifted:
                direction = "↑" if lower_is_better else "↓"
                drift_warnings.append(
                    f"{direction} {key}: current={_fmt(current)} — {reason}"
                )

    report = "\n".join(lines) + "\n"
    return report, drift_warnings


def run_report(output_path: Path, drift_threshold: float, lower_is_better: list[str]) -> int:
    try:
        history = load_history(output_path)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    report, drift_warnings = generate_report(
        history,
        drift_threshold=drift_threshold,
        lower_is_better_metrics=set(lower_is_better),
    )

    print(report, end="")

    if drift_warnings:
        warn_block = [
            "",
            "╔════════════════════════════════════════════════════════════╗",
            "║                  ⚠  PERFORMANCE DRIFT DETECTED  ⚠         ║",
            "╚════════════════════════════════════════════════════════════╝",
            "",
        ]
        for w in drift_warnings:
            warn_block.append(f"  • {w}")
        warn_block.append("")
        warn_block.append("  The current run falls significantly below the 10-run moving average.")
        warn_block.append("")
        print("\n".join(warn_block), file=sys.stderr)
        return 3  # distinct exit code for drift

    return 0


# ---------------------------------------------------------------------------
# Append mode
# ---------------------------------------------------------------------------

def run_append(input_path: Path, output_path: Path, max_entries: int) -> tuple[int, bool]:
    """Append a metrics entry. Returns (exit_code, appended_ok)."""
    # Load metrics
    try:
        metrics = load_json(input_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1, False

    # Load existing history
    try:
        history = load_history(output_path)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1, False

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_sha(),
        "git_branch": get_git_branch(),
        "metrics": metrics,
    }

    history.append(entry)

    if len(history) > max_entries:
        history = history[-max_entries:]

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_history(output_path, history)
    except OSError as e:
        print(f"error: failed to write {output_path}: {e}", file=sys.stderr)
        return 1, False

    print(
        f"Appended metrics to {output_path} "
        f"(entry {len(history)}/{max_entries}, "
        f"commit={entry['git_commit'][:7] if entry['git_commit'] else 'n/a'}, "
        f"branch={entry['git_branch'] or 'n/a'})"
    )
    return 0, True


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Append metrics from latest_metrics.json to a rolling history file, "
                    "with optional trend reporting."
    )
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_INPUT,
        help=f"Input metrics JSON file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT,
        help=f"History JSON file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--max-entries", "-n",
        type=int,
        default=DEFAULT_MAX_ENTRIES,
        help=f"Maximum number of historical entries to keep (default: {DEFAULT_MAX_ENTRIES})",
    )
    # Reporting flags
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate a trend report from history.json (skips append unless --append is also given)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="When used with --report, append metrics first before generating the report",
    )
    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=0.05,
        help="Relative change threshold for drift detection (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--lower-is-better",
        action="append",
        default=[],
        metavar="METRIC",
        help="Metric key where lower values are better (repeatable, supports dotted keys)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Report-only mode (default when --report is given without --append)
    if args.report and not args.append:
        return run_report(output_path, args.drift_threshold, args.lower_is_better)

    # Append mode
    if args.max_entries < 1:
        print(f"error: --max-entries must be >= 1, got {args.max_entries}", file=sys.stderr)
        return 2

    exit_code, ok = run_append(input_path, output_path, args.max_entries)
    if exit_code != 0:
        return exit_code

    # If --report was requested alongside append, generate report after
    if args.report:
        print()
        rc = run_report(output_path, args.drift_threshold, args.lower_is_better)
        # Prioritize drift exit code (3) over success, but don't mask append failures
        return rc

    return 0


if __name__ == "__main__":
    sys.exit(main())
