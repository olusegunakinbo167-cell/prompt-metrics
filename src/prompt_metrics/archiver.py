# src/prompt_metrics/archiver.py
"""
Run artifact archiver for prompt_metrics.

Bundles completed evaluation run outputs (results.json, results.csv,
report.md, comparison.md) into a timestamped compressed archive.
"""

from __future__ import annotations

import datetime
import json
import zipfile
from pathlib import Path
from typing import Any


# Core run artifacts to include in archives (if present)
ARTIFACTS = [
    "results.json",
    "results.csv",
    "report.md",
    "comparison.md",
]


def create_run_archive(
    output_dir: str,
    archive_dir: str = "archives",
    *,
    archive_format: str = "zip",
) -> str:
    """
    Bundle a completed evaluation run into a compressed archive.

    Scans `output_dir` for run artifacts (`results.json`, `results.csv`,
    `report.md`, `comparison.md`), extracts run metadata from
    `results.json`, and creates a timestamped archive.

    Args:
        output_dir: Directory containing run output files.
            Expected files (any subset is OK):
            - results.json  (used for metadata extraction)
            - results.csv
            - report.md
            - comparison.md
        archive_dir: Destination directory for archives.
            Created if it doesn't exist. Default: "archives".
        archive_format: Archive format. Currently only "zip" is supported.
            Future: "tar.gz".

    Returns:
        Absolute path to the created archive file (str).

    Archive naming:
        run_<YYYYMMDD>_<HHMMSS>_<score>.zip

        Where <score> is the mean aggregated score across all evaluators,
        formatted as a 0-padded 3-digit integer percentage (e.g. 085 for 0.85).
        If no score can be determined, uses "XXX".

        Example: run_20260716_132045_087.zip

    Raises:
        FileNotFoundError: If output_dir does not exist.
        ValueError: If archive_format is not supported.
        RuntimeError: If no artifacts were found to archive.

    Example:
        >>> archive_path = create_run_archive("./results/run_001")
        >>> print(archive_path)
        /home/user/archives/run_20260716_132045_087.zip
    """
    output_path = Path(output_dir).resolve()
    if not output_path.exists() or not output_path.is_dir():
        raise FileNotFoundError(f"Output directory not found: {output_path}")

    if archive_format != "zip":
        raise ValueError(f"Unsupported archive_format: {archive_format!r} (only 'zip' is supported)")

    # Collect existing artifacts
    found_files: list[Path] = []
    for name in ARTIFACTS:
        f = output_path / name
        if f.exists() and f.is_file():
            found_files.append(f)

    if not found_files:
        raise RuntimeError(
            f"No run artifacts found in {output_path}. "
            f"Looked for: {', '.join(ARTIFACTS)}"
        )

    # Extract metadata from results.json if available
    metadata = _extract_run_metadata(output_path / "results.json")
    timestamp_str = metadata["timestamp"]
    score_str = metadata["score_str"]

    # Build archive filename
    archive_name = f"run_{timestamp_str}_{score_str}.{archive_format}"
    
    # Create archive directory
    archive_dir_path = Path(archive_dir).resolve()
    archive_dir_path.mkdir(parents=True, exist_ok=True)

    archive_path = archive_dir_path / archive_name

    # Avoid overwriting — if file exists, append _v2, _v3, etc.
    counter = 2
    while archive_path.exists():
        stem = archive_path.stem
        suffix = archive_path.suffix
        archive_path = archive_dir_path / f"{stem}_v{counter}{suffix}"
        counter += 1

    # Create zip archive
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for file_path in found_files:
            # Store files at archive root (no directory prefix)
            zf.write(file_path, arcname=file_path.name)

        # Add a small metadata manifest
        manifest = (
            f"prompt_metrics run archive\n"
            f"==========================\n"
            f"source_dir: {output_path}\n"
            f"timestamp:  {metadata['timestamp_readable']}\n"
            f"files:      {', '.join(p.name for p in found_files)}\n"
        )
        if metadata.get("case_count") is not None:
            manifest += f"cases:      {metadata['case_count']}\n"
        if metadata.get("mean_score") is not None:
            manifest += f"mean_score: {metadata['mean_score']:.4f}\n"
        zf.writestr("MANIFEST.txt", manifest)

    return str(archive_path)


# ---- helpers ----

def _extract_run_metadata(results_json_path: Path) -> dict[str, Any]:
    """
    Extract timestamp, case count, and mean score from results.json.

    Handles multiple results.json formats:
      1. export_results() format — list of case dicts
         [{case_id, input_prompt, generated_response, evaluator_results: {...}, latency_ms}, ...]
      2. SuiteResult.save_json() / SuiteResult.to_dict() envelope
         {summary: {...}, results: [...]}

    Returns dict with:
      timestamp: str           — "YYYYMMDD_HHMMSS" for filename
      timestamp_readable: str  — "YYYY-MM-DD HH:MM:SS UTC" for manifest
      score_str: str           — "087" / "XXX"
      mean_score: float | None
      case_count: int | None
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    timestamp_readable = now.strftime("%Y-%m-%d %H:%M:%S UTC")

    mean_score: float | None = None
    case_count: int | None = None

    if results_json_path.exists():
        try:
            with results_json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            # Normalize to list of case records
            if isinstance(data, dict) and "results" in data:
                # SuiteResult envelope
                records = data["results"]
                # Try to get timestamp from summary if available
                summary = data.get("summary", {})
                # No standard timestamp field in SuiteResult, use file mtime
            elif isinstance(data, list):
                # export_results format
                records = data
            else:
                records = []

            if isinstance(records, list):
                case_count = len(records)

                # Extract all numeric scores
                scores: list[float] = []
                for rec in records:
                    if not isinstance(rec, dict):
                        continue
                    eval_results = rec.get("evaluator_results", {})
                    if not isinstance(eval_results, dict):
                        continue
                    for ev_name, ev_result in eval_results.items():
                        if isinstance(ev_result, dict):
                            score = ev_result.get("score")
                            if isinstance(score, (int, float)):
                                # Clamp QA-style 1-5 scores to 0-1 range heuristically:
                                # if score > 1.0, assume 1-5 scale → normalize
                                if score > 1.0:
                                    score = max(0.0, min(1.0, (score - 1.0) / 4.0))
                                scores.append(float(score))

                if scores:
                    mean_score = sum(scores) / len(scores)

            # Try to use results.json mtime as timestamp (run completion time)
            try:
                mtime = results_json_path.stat().st_mtime
                dt = datetime.datetime.fromtimestamp(mtime, tz=datetime.timezone.utc)
                timestamp_str = dt.strftime("%Y%m%d_%H%M%S")
                timestamp_readable = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            except Exception:
                pass

        except Exception:
            # JSON parse failed — use defaults
            pass

    # Format score for filename: 0.8742 → "087"
    if mean_score is not None:
        score_pct = int(round(max(0.0, min(1.0, mean_score)) * 100))
        score_str = f"{score_pct:03d}"
    else:
        score_str = "XXX"

    return {
        "timestamp": timestamp_str,
        "timestamp_readable": timestamp_readable,
        "score_str": score_str,
        "mean_score": mean_score,
        "case_count": case_count,
    }


__all__ = ["create_run_archive"]
