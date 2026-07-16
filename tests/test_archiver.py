# tests/test_archiver.py
"""
Tests for prompt_metrics.archiver — run artifact bundling.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from prompt_metrics.archiver import create_run_archive


def test_create_run_archive_bundles_artifacts(tmp_path: Path):
    output_dir = tmp_path / "run_output"
    output_dir.mkdir()
    results = [
        {"case_id": "c1", "input_prompt": "q1", "generated_response": "r1",
         "evaluator_results": {"exact_match": {"score": 1.0}, "keyword": {"score": 0.75}}, "latency_ms": 42.0},
        {"case_id": "c2", "input_prompt": "q2", "generated_response": "r2",
         "evaluator_results": {"exact_match": {"score": 0.0}, "keyword": {"score": 0.5}}, "latency_ms": 55.0},
    ]
    (output_dir / "results.json").write_text(json.dumps(results), encoding="utf-8")
    (output_dir / "results.csv").write_text("case_id,score\nc1,1.0\nc2,0.0\n", encoding="utf-8")
    (output_dir / "report.md").write_text("# Report\n", encoding="utf-8")
    archive_dir = tmp_path / "my_archives"
    archive_path = create_run_archive(str(output_dir), archive_dir=str(archive_dir))
    assert Path(archive_path).exists()
    assert Path(archive_path).parent == archive_dir.resolve()
    assert Path(archive_path).name.startswith("run_")
    assert Path(archive_path).name.endswith(".zip")
    with zipfile.ZipFile(archive_path, "r") as zf:
        names = set(zf.namelist())
        assert "results.json" in names
        assert "results.csv" in names
        assert "report.md" in names
        assert "MANIFEST.txt" in names
        assert "comparison.md" not in names


def test_archiver_extracts_score_for_filename(tmp_path: Path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    results = [{"case_id": "x", "evaluator_results": {"test": {"score": 0.9}}}]
    (output_dir / "results.json").write_text(json.dumps(results), encoding="utf-8")
    archive_path = create_run_archive(str(output_dir), archive_dir=str(tmp_path / "a"))
    name = Path(archive_path).name
    assert "_090" in name
    assert name.startswith("run_")
    assert name.endswith(".zip")


def test_archiver_handles_missing_results_json(tmp_path: Path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "report.md").write_text("# Report", encoding="utf-8")
    archive_path = create_run_archive(str(output_dir), archive_dir=str(tmp_path / "a"))
    assert Path(archive_path).exists()
    assert "_XXX" in Path(archive_path).name
    with zipfile.ZipFile(archive_path, "r") as zf:
        names = set(zf.namelist())
        assert "report.md" in names
        assert "MANIFEST.txt" in names


def test_archiver_versions_on_collision(tmp_path: Path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "results.json").write_text("[]", encoding="utf-8")
    archive_dir = tmp_path / "archives"
    path1 = create_run_archive(str(output_dir), archive_dir=str(archive_dir))
    path2 = create_run_archive(str(output_dir), archive_dir=str(archive_dir))
    assert path1 != path2
    assert Path(path1).exists()
    assert Path(path2).exists()
    assert "_v2" in Path(path2).name


def test_cli_run_with_archive_flag(tmp_path: Path):
    import sys, types
    fake_module = types.ModuleType("fake_gen_archive")
    fake_module.generate = lambda prompt: f"response to {prompt}"
    sys.modules["fake_gen_archive"] = fake_module
    try:
        from prompt_metrics.cli import main as cli_main
        dataset_path = tmp_path / "data.json"
        dataset_path.write_text(json.dumps([{"id": "c1", "input_prompt": "hello"}]), encoding="utf-8")
        output_dir = tmp_path / "results"
        archive_dir = tmp_path / "my_archives"
        exit_code = cli_main([
            "run", "--dataset", str(dataset_path),
            "--output-dir", str(output_dir),
            "--evaluators", "exact_match",
            "--generator", "fake_gen_archive:generate",
            "--formats", "json",
            "--archive", str(archive_dir),
        ])
        assert exit_code == 0
        assert (output_dir / "results.json").exists()
        archives = list(archive_dir.glob("run_*.zip"))
        assert len(archives) == 1
        with zipfile.ZipFile(archives[0], "r") as zf:
            names = set(zf.namelist())
            assert "results.json" in names
            assert "MANIFEST.txt" in names
    finally:
        sys.modules.pop("fake_gen_archive", None)
