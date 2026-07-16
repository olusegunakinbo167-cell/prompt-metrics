import json
import sys
import os
import pytest

from prompt_metrics.cli import main


def test_generator(prompt: str) -> str:
    """Dummy generator for dynamic loading test."""
    return "CUSTOM_GEN:" + prompt


def test_cli_successful_run(tmp_path, monkeypatch, capsys):
    """Standard successful CLI run writes all three output formats."""
    # Create temp dataset
    dataset = [
        {"id": "case_001", "input_prompt": "hello", "expected_text": "Echo: hello", "keywords": ["hello"]},
        {"id": "case_002", "input_prompt": "foo", "expected_text": "bar", "keywords": ["foo"]},
    ]
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    output_dir = tmp_path / "results"

    argv = [
        "prompt_metrics",
        "--dataset", str(dataset_path),
        "--output-dir", str(output_dir),
        "--evaluators", "exact_match,keyword",
        "--formats", "json,csv,md",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    exit_code = main()

    assert exit_code == 0

    # Assert expected output files exist
    assert (output_dir / "results.json").is_file()
    assert (output_dir / "results.csv").is_file()
    assert (output_dir / "report.md").is_file()

    # Validate JSON structure
    with open(output_dir / "results.json", encoding="utf-8") as f:
        data = json.load(f)
    assert "results" in data
    assert len(data["results"]) == 2
    assert data["metadata"]["success_count"] == 2

    # CSV is non-empty and has expected headers
    csv_text = (output_dir / "results.csv").read_text(encoding="utf-8")
    assert "case_id" in csv_text
    assert "evaluator_results_exact_match_score" in csv_text

    # Markdown report contains expected sections
    md_text = (output_dir / "report.md").read_text(encoding="utf-8")
    assert "Experiment Report" in md_text
    assert "## Summary" in md_text
    assert "case_001" in md_text


def test_cli_missing_dataset_argument(capsys):
    """Missing required --dataset exits with code 2 and prints usage."""
    # argparse calls sys.exit(2) on missing required args
    with pytest.raises(SystemExit) as exc_info:
        main([])
    
    assert exc_info.value.code == 2

    captured = capsys.readouterr()
    # argparse prints usage to stderr
    stderr = captured.err
    assert "usage:" in stderr.lower()
    assert "--dataset" in stderr


def test_cli_dynamic_generator_loading(tmp_path, monkeypatch):
    """--generator tests.test_cli:test_generator is imported and executed."""
    dataset = [
        {"id": "case_001", "input_prompt": "test_input", "expected_text": "CUSTOM_GEN:test_input", "keywords": []}
    ]
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    output_dir = tmp_path / "results"

    # Ensure tests.test_cli is importable – when running pytest from /tmp/pmtest,
    # the tests/ directory needs to be on sys.path. Add tmp_path parent.
    # In normal repo layout, `tests` is a top-level package.
    # Here we run with PYTHONPATH including /tmp/pmtest, so import as test_cli
    # Try both import paths.
    generator_spec = None
    try:
        __import__("tests.test_cli")
        generator_spec = "tests.test_cli:test_generator"
    except ModuleNotFoundError:
        # fallback for when running as a single-file module
        generator_spec = "test_cli:test_generator"

    argv = [
        "prompt_metrics",
        "--dataset", str(dataset_path),
        "--output-dir", str(output_dir),
        "--evaluators", "exact_match",
        "--formats", "json",
        "--generator", generator_spec,
    ]
    monkeypatch.setattr(sys, "argv", argv)

    exit_code = main()
    assert exit_code == 0

    # Verify the custom generator was actually used
    with open(output_dir / "results.json", encoding="utf-8") as f:
        data = json.load(f)
    
    result = data["results"][0]
    assert result["generated_response"] == "CUSTOM_GEN:test_input"
    # Exact match should succeed
    assert result["evaluator_results"]["exact_match"]["score"] == 1.0
