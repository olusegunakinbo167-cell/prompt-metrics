#!/usr/bin/env python3
"""
track_metrics.py — prompt-metrics evaluation asset health check

Verifies:
  1. Rubric archive exists and is a valid zip
  2. eval_config.json inside parses and has required rubric structure
  3. Local error logs are scanned for recent failures

Usage:
  python track_metrics.py
  python track_metrics.py --zip ./rubrics/prompt-metrics-rubric-v1.zip --rubric eval_config.json

Exit codes:
  0 – all checks passed
  1 – asset missing / corrupt / invalid schema
  2 – errors found in logs
"""

import argparse
import json
import sys
import zipfile
from pathlib import Path
from datetime import datetime, timezone

# --- Defaults (override via CLI) ---
DEFAULT_ZIP = "prompt-metrics-rubric-v1.zip"
DEFAULT_RUBRIC_NAME = "eval_config.json"

REQUIRED_DIMENSIONS = {"accuracy", "fluency", "hallucination_check"}


def check_zip_exists(zip_path: Path) -> tuple[bool, str]:
    if not zip_path.exists():
        return False, f"Missing: {zip_path}"
    if not zipfile.is_zipfile(zip_path):
        return False, f"Not a valid zip: {zip_path}"
    size = zip_path.stat().st_size
    return True, f"OK — {zip_path.name} exists ({size} bytes)"


def check_rubric_structure(zip_path: Path, rubric_name: str) -> tuple[bool, str]:
    try:
        with zipfile.ZipFile(zip_path) as z:
            if rubric_name not in z.namelist():
                # also try finding any *.json that looks like a rubric
                candidates = [n for n in z.namelist() if n.endswith('.json')]
                if candidates:
                    rubric_name = candidates[0]
                else:
                    return False, f"{rubric_name} not found inside zip (contents: {z.namelist()})"
            with z.open(rubric_name) as f:
                rubric = json.load(f)
    except Exception as e:
        return False, f"Failed to parse rubric JSON: {e}"

    dimensions = rubric.get("dimensions", [])
    if not isinstance(dimensions, list):
        return False, "rubric.dimensions is not a list"

    found_ids = {d.get("id") for d in dimensions if isinstance(d, dict)}
    missing = REQUIRED_DIMENSIONS - found_ids
    if missing:
        return False, f"Missing required dimensions: {sorted(missing)}"

    errors = []
    for d in dimensions:
        dim_id = d.get("id")
        if dim_id not in REQUIRED_DIMENSIONS:
            continue
        if d.get("scale_min") != 1 or d.get("scale_max") != 5:
            errors.append(f"{dim_id}: scale must be 1-5")
        levels = d.get("levels", {})
        expected_levels = {"1", "2", "3", "4", "5"}
        if set(str(k) for k in levels.keys()) != expected_levels:
            errors.append(f"{dim_id}: levels must define 1-5")

    if errors:
        return False, "; ".join(errors)

    return True, f"OK — rubric valid: {len(dimensions)} dimensions, {', '.join(sorted(found_ids))}"


def check_error_logs(log_paths: list[Path]) -> tuple[bool, str]:
    scanned = []
    error_lines = []
    for log_path in log_paths:
        if not log_path.exists():
            continue
        scanned.append(str(log_path))
        try:
            with log_path.open("r", encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f, 1):
                    low = line.lower()
                    if any(kw in low for kw in ("error", "traceback", "exception", "fail", "critical")):
                        error_lines.append(f"{log_path.name}:{i}: {line.strip()[:200]}")
                        if len(error_lines) >= 20:
                            break
        except Exception as e:
            error_lines.append(f"{log_path}: read failed: {e}")
        if error_lines:
            break

    if error_lines:
        preview = "\n  ".join(error_lines[:10])
        return False, f"Errors found in {scanned[0]}:\n  {preview}"
    if scanned:
        return True, f"OK — scanned {len(scanned)} log(s), no errors"
    return True, "OK — no error logs found (nothing to check)"


def main() -> int:
    parser = argparse.ArgumentParser(description="prompt-metrics asset health check")
    parser.add_argument("--zip", dest="zip_path", default=DEFAULT_ZIP,
                        help=f"Path to rubric zip (default: {DEFAULT_ZIP})")
    parser.add_argument("--rubric", default=DEFAULT_RUBRIC_NAME,
                        help=f"Rubric filename inside zip (default: {DEFAULT_RUBRIC_NAME})")
    parser.add_argument("--log", action="append", dest="logs", default=[],
                        help="Error log to scan (repeatable, default: ./error.log)")
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    log_paths = [Path(p) for p in args.logs] if args.logs else [
        Path("error.log"),
        Path("track_metrics_error.log"),
    ]

    ts = datetime.now(timezone.utc).isoformat()
    print(f"track_metrics.py — {ts}")
    print(f"zip: {zip_path.resolve()}")
    print()

    ok_zip, msg_zip = check_zip_exists(zip_path)
    print(f"[zip]     {'PASS' if ok_zip else 'FAIL'} — {msg_zip}")

    ok_rubric = False
    msg_rubric = "SKIP — zip check failed"
    if ok_zip:
        ok_rubric, msg_rubric = check_rubric_structure(zip_path, args.rubric)
    print(f"[rubric]  {'PASS' if ok_rubric else 'FAIL'} — {msg_rubric}")

    ok_logs, msg_logs = check_error_logs(log_paths)
    print(f"[logs]    {'PASS' if ok_logs else 'FAIL'} — {msg_logs}")

    print()
    if not (ok_zip and ok_rubric):
        print("Result: ASSET FAILURE")
        return 1
    if not ok_logs:
        print("Result: LOG ERRORS DETECTED")
        return 2
    print("Result: ALL CHECKS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
