#!/usr/bin/env python3
"""
label_pr.py — Auto-label PRs based on eval accuracy deltas

Compares latest_metrics.json against baseline.json (or parses comparison.md
as a fallback) and applies GitHub labels:

  accuracy_delta > 0  → perf-improvement
  accuracy_delta < 0  → perf-regression
  accuracy_delta == 0 → no perf label (removes both)

Labels are applied via `gh pr edit`.

The script is CI-friendly:
- PR number: --pr / $PR_NUMBER / $GITHUB_PR_NUMBER / auto-detect via `gh pr view`
- Repo: --repo owner/repo / $GITHUB_REPOSITORY / auto-detect via `gh repo view`
- Configurable metric key (default: accuracy)
- Configurable epsilon for float noise (default: 1e-9)
- Dry-run mode
- Removes the opposing label automatically

Exit codes:
  0 – success
  2 – metrics files missing / unparseable
  3 – gh call failed / PR resolution failed
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parent
DEFAULT_LATEST = WORKSPACE_ROOT / "latest_metrics.json"
DEFAULT_BASELINE = WORKSPACE_ROOT / "baseline.json"
DEFAULT_COMPARISON = WORKSPACE_ROOT / "comparison.md"

REGRESSION_LABEL = "perf-regression"
IMPROVEMENT_LABEL = "perf-improvement"
REGRESSION_COLOR = "d73a4a"  # red
IMPROVEMENT_COLOR = "0e8a16"  # green


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=check)
    except FileNotFoundError as e:
        # Fake a failed CompletedProcess so callers that check returncode still work
        return subprocess.CompletedProcess(cmd, returncode=127, stdout="", stderr=str(e))


def load_json_metric(path: Path, metric_key: str) -> float | None:
    if not path.exists():
        return None
    try:
        with path.open() as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to parse {path}: {e}", file=sys.stderr)
        return None

    # Direct key
    if metric_key in data and isinstance(data[metric_key], (int, float)):
        return float(data[metric_key])

    # Case-insensitive top-level search
    lk = metric_key.lower()
    for k, v in data.items() if isinstance(data, dict) else []:
        if k.lower() == lk and isinstance(v, (int, float)):
            return float(v)

    # Nested: scores / metrics / evaluator_results
    for container_key in ("scores", "metrics", "evaluator_results", "results"):
        container = data.get(container_key) if isinstance(data, dict) else None
        if isinstance(container, dict) and metric_key in container:
            v = container[metric_key]
            if isinstance(v, (int, float)):
                return float(v)

    return None


def parse_comparison_md(path: Path, metric_key: str) -> tuple[float | None, float | None]:
    """Try to extract baseline/current accuracy from comparison.md.

    Looks for patterns like:
      | accuracy | 0.8500 | 0.8420 | -0.0080 |
      accuracy: 0.85 -> 0.87
      accuracy 0.85 vs 0.87
    Returns (baseline, current) or (None, None).
    """
    if not path.exists():
        return None, None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None, None

    mk = re.escape(metric_key)
    # Table row: | accuracy | 0.85 | 0.87 |
    m = re.search(
        rf"\|\s*{mk}\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)",
        text,
        re.IGNORECASE,
    )
    if m:
        try:
            return float(m.group(1)), float(m.group(2))
        except ValueError:
            pass

    # "accuracy: 0.85 -> 0.87" / "accuracy 0.85 vs 0.87"
    m = re.search(
        rf"{mk}\s*[:=]\s*([0-9.]+)\s*(?:->|→|vs|to)\s*([0-9.]+)",
        text,
        re.IGNORECASE,
    )
    if m:
        try:
            return float(m.group(1)), float(m.group(2))
        except ValueError:
            pass

    return None, None


def resolve_pr_number(explicit: int | None) -> int | None:
    import os

    if explicit:
        return explicit
    for env_key in ("PR_NUMBER", "GITHUB_PR_NUMBER", "PR"):
        v = os.environ.get(env_key)
        if v and v.isdigit():
            return int(v)
    # gh auto-detect
    try:
        result = run(
            ["gh", "pr", "view", "--json", "number", "-q", ".number"],
            check=False,
        )
        if result.returncode == 0:
            n = result.stdout.strip()
            if n.isdigit():
                return int(n)
    except FileNotFoundError:
        pass
    return None


def resolve_repo(explicit: str | None) -> str | None:
    import os

    if explicit:
        return explicit
    v = os.environ.get("GITHUB_REPOSITORY")
    if v:
        return v
    try:
        result = run(
            ["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
            check=False,
        )
        if result.returncode == 0:
            repo = result.stdout.strip()
            return repo or None
    except FileNotFoundError:
        pass
    return None


def get_pr_labels(pr: int, repo: str | None) -> set[str]:
    cmd = ["gh", "pr", "view", str(pr), "--json", "labels", "-q", ".labels[].name"]
    if repo:
        cmd.extend(["--repo", repo])
    result = run(cmd, check=False)
    if result.returncode != 0:
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def get_repo_labels(repo: str | None) -> set[str] | None:
    """Return all label names defined in the repo, or None if gh failed."""
    cmd = ["gh", "label", "list", "--json", "name", "-q", ".[].name", "--limit", "1000"]
    if repo:
        cmd.extend(["--repo", repo])
    result = run(cmd, check=False)
    if result.returncode != 0:
        # If we can't list labels, return None so callers can distinguish
        # "empty repo" from "gh lookup failed".
        print(f"Warning: gh label list failed: {result.stderr.strip()}", file=sys.stderr)
        return None
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def ensure_labels_exist(
    labels_with_colors: dict[str, str],
    repo: str | None,
    dry_run: bool,
) -> bool:
    """
    Ensure each label in labels_with_colors exists in the repo.
    Creates missing labels via `gh label create`.

    labels_with_colors: {label_name: hex_color_without_hash}
    Returns True if all labels exist / were created successfully.
    """
    existing = get_repo_labels(repo)

    # If label listing failed (gh unavailable / API error), assume labels exist
    # to avoid blocking the pipeline – apply_labels will fail naturally if needed.
    if existing is None:
        return True

    ok = True
    for name, color in labels_with_colors.items():
        if name in existing:
            continue
        cmd = ["gh", "label", "create", name, "--color", color.lstrip("#")]
        if repo:
            cmd.extend(["--repo", repo])
        # Add descriptions for the default perf labels
        if name == REGRESSION_LABEL:
            cmd.extend(["--description", "Accuracy decreased vs baseline"])
        elif name == IMPROVEMENT_LABEL:
            cmd.extend(["--description", "Accuracy increased vs baseline"])
        print(f"$ {' '.join(cmd)}  # auto-creating missing label")
        if dry_run:
            continue
        result = run(cmd, check=False)
        if result.returncode != 0:
            # Label might have been created concurrently – check if that's the case
            if "already exists" in result.stderr.lower():
                print(f"Label '{name}' already exists (race).", file=sys.stderr)
                continue
            print(f"gh label create '{name}' failed: {result.stderr}", file=sys.stderr)
            ok = False
    return ok


def apply_labels(
    pr: int,
    add: list[str],
    remove: list[str],
    repo: str | None,
    dry_run: bool,
) -> bool:
    ok = True
    if remove:
        cmd = ["gh", "pr", "edit", str(pr), "--remove-label", ",".join(remove)]
        if repo:
            cmd.extend(["--repo", repo])
        print(f"$ {' '.join(cmd)}")
        if not dry_run:
            result = run(cmd, check=False)
            if result.returncode != 0:
                print(f"gh remove-label failed: {result.stderr}", file=sys.stderr)
                ok = False
    if add:
        cmd = ["gh", "pr", "edit", str(pr), "--add-label", ",".join(add)]
        if repo:
            cmd.extend(["--repo", repo])
        print(f"$ {' '.join(cmd)}")
        if not dry_run:
            result = run(cmd, check=False)
            if result.returncode != 0:
                print(f"gh add-label failed: {result.stderr}", file=sys.stderr)
                ok = False
    return ok


def write_step_summary(
    metric: str,
    baseline: float,
    current: float,
    delta: float,
    verdict: str,
    pr_number: int | None,
    repo: str | None,
    applied_add: list[str],
    applied_remove: list[str],
    dry_run: bool,
) -> None:
    """Append a performance summary to $GITHUB_STEP_SUMMARY if set."""
    import os

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    emoji_map = {
        "regression": "🚨",
        "improvement": "🎉",
        "unchanged": "🤝",
    }
    emoji = emoji_map.get(verdict, "📊")
    title_map = {
        "regression": f"{emoji} Performance Regression",
        "improvement": f"{emoji} Performance Improvement",
        "unchanged": f"{emoji} Performance Unchanged",
    }
    title = title_map.get(verdict, f"{emoji} Performance Report")

    pct_delta = (delta / baseline * 100) if baseline != 0 else 0.0
    delta_sign = "+" if delta >= 0 else ""

    try:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(f"### {title}\n\n")
            if pr_number:
                pr_link = f"#{pr_number}"
                if repo:
                    pr_link = f"[{pr_link}](https://github.com/{repo}/pull/{pr_number})"
                f.write(f"**PR:** {pr_link}\n\n")
            f.write(f"**Metric:** `{metric}`\n\n")
            f.write("|  | Baseline | Current | Delta |\n")
            f.write("|---|---|---|---|\n")
            f.write(
                f"| **{metric}** | {baseline:.6f} | {current:.6f} | "
                f"{delta_sign}{delta:.6f} ({delta_sign}{pct_delta:.2f}%) |\n"
            )
            f.write("\n")

            if applied_add or applied_remove:
                f.write("**Labels:**\n")
                if applied_add:
                    f.write(f"- ✅ Added: `{'`, `'.join(applied_add)}`\n")
                if applied_remove:
                    f.write(f"- 🗑️ Removed: `{'`, `'.join(applied_remove)}`\n")
                f.write("\n")
            elif verdict == "unchanged":
                f.write("_No label changes — performance is within tolerance._\n\n")

            if dry_run:
                f.write("> _Dry run — no labels were actually modified._\n\n")
    except Exception as e:
        print(f"Warning: failed to write GITHUB_STEP_SUMMARY: {e}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description="Auto-label PR based on eval accuracy delta")
    ap.add_argument("--latest", type=Path, default=DEFAULT_LATEST, help=f"latest metrics json (default: {DEFAULT_LATEST})")
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE, help=f"baseline json (default: {DEFAULT_BASELINE})")
    ap.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON, help="comparison.md fallback")
    ap.add_argument("--metric", default="accuracy", help="metric key to compare (default: accuracy)")
    ap.add_argument("--pr", type=int, default=None, help="PR number (auto-detect if omitted)")
    ap.add_argument("--repo", default=None, help="owner/repo (auto-detect if omitted)")
    ap.add_argument("--epsilon", type=float, default=1e-9, help="float tolerance for equality (default: 1e-9)")
    ap.add_argument("--regression-label", default=REGRESSION_LABEL)
    ap.add_argument("--improvement-label", default=IMPROVEMENT_LABEL)
    ap.add_argument("--regression-color", default=REGRESSION_COLOR, help="hex color for auto-created regression label")
    ap.add_argument("--improvement-color", default=IMPROVEMENT_COLOR, help="hex color for auto-created improvement label")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fail-on-regression", action="store_true", help="exit 1 if regression detected (gating, off by default)")
    args = ap.parse_args()

    # 1. Load metrics
    current = load_json_metric(args.latest, args.metric)
    baseline = load_json_metric(args.baseline, args.metric)

    # Fallback to comparison.md if either is missing
    if current is None or baseline is None:
        cmp_base, cmp_curr = parse_comparison_md(args.comparison, args.metric)
        if baseline is None and cmp_base is not None:
            baseline = cmp_base
            print(f"baseline {args.metric}={baseline} parsed from {args.comparison}", file=sys.stderr)
        if current is None and cmp_curr is not None:
            current = cmp_curr
            print(f"current {args.metric}={current} parsed from {args.comparison}", file=sys.stderr)

    if current is None or baseline is None:
        print(
            f"ERROR: could not resolve '{args.metric}' from {args.latest} / {args.baseline} / {args.comparison}",
            file=sys.stderr,
        )
        return 2

    delta = current - baseline
    print(f"{args.metric}: baseline={baseline:.6f} current={current:.6f} delta={delta:+.6f}")

    # 2. Decide labels
    if delta < -args.epsilon:
        verdict = "regression"
        add_labels = [args.regression_label]
        remove_labels = [args.improvement_label]
    elif delta > args.epsilon:
        verdict = "improvement"
        add_labels = [args.improvement_label]
        remove_labels = [args.regression_label]
    else:
        verdict = "unchanged"
        add_labels = []
        remove_labels = [args.regression_label, args.improvement_label]

    print(f"verdict: {verdict}")

    # 3. Resolve PR
    pr_number = resolve_pr_number(args.pr)
    if pr_number is None:
        print("ERROR: could not resolve PR number (use --pr / $PR_NUMBER / $GITHUB_PR_NUMBER)", file=sys.stderr)
        return 3

    repo = resolve_repo(args.repo)
    print(f"target: PR #{pr_number}" + (f" in {repo}" if repo else ""))

    # 4. Ensure labels exist in the repo before trying to apply them
    labels_to_ensure = {
        args.regression_label: args.regression_color,
        args.improvement_label: args.improvement_color,
    }
    if not ensure_labels_exist(labels_to_ensure, repo, args.dry_run):
        print("ERROR: failed to ensure required labels exist", file=sys.stderr)
        return 3

    # 5. Make labeling idempotent
    existing = get_pr_labels(pr_number, repo)
    add_labels = [l for l in add_labels if l not in existing]
    remove_labels = [l for l in remove_labels if l in existing]

    label_ok = True
    if not add_labels and not remove_labels:
        print("labels already correct — nothing to do")
    else:
        label_ok = apply_labels(pr_number, add_labels, remove_labels, repo, args.dry_run)
        if not label_ok:
            print("ERROR: label application failed", file=sys.stderr)
        elif args.dry_run:
            print("(dry-run — no labels were actually changed)")

    # 6. Write GitHub Actions step summary
    write_step_summary(
        metric=args.metric,
        baseline=baseline,
        current=current,
        delta=delta,
        verdict=verdict,
        pr_number=pr_number,
        repo=repo,
        applied_add=add_labels,
        applied_remove=remove_labels,
        dry_run=args.dry_run,
    )

    if not label_ok:
        return 3
    if args.fail_on_regression and verdict == "regression":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
