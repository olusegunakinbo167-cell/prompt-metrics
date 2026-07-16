# tests/test_cli.py
"""
CLI tests for prompt_metrics.

Covers:
  - End-to-end successful run with mocked sys.argv
  - Argparse validation (missing required --dataset)
  - Custom generator resolution via --generator module:callable
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Dummy generator for test case 3
# ---------------------------------------------------------------------------

def dummy_generator(prompt: str) -> str:
    """
    Custom generator for testing --generator module resolution.

    Returns a deterministic, easily identifiable response so the test
    can assert the CLI actually used this generator instead of the built-in mock.
    """
    return f"DUMMY_GENERATED::{prompt.upper()}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_dataset(tmp_path: Path) -> Path:
    """Create a small JSON dataset file in a temp directory."""
    dataset = [
        {
            "id": "cli_case_001",
            "input_prompt": "What is the capital of France?",
            "expected_text": "Paris",
            "keywords": ["Paris", "France", "capital"],
        },
        {
            "id": "cli_case_002",
            "input_prompt": "2 + 2 = ?",
            "expected_text": "4",
            "keywords": ["4", "four"],
        },
    ]
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    return dataset_path


# ---------------------------------------------------------------------------
# Test 1: Successful CLI run
# ---------------------------------------------------------------------------

def test_cli_successful_run(
    temp_dataset: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Verify a standard CLI invocation:
      - parses arguments correctly
      - loads the dataset
      - instantiates the requested evaluators
      - runs the full ExperimentRunner suite
      - writes results.json, results.csv, and report.md
      - exits with code 0
    """
    from src.prompt_metrics.cli import main

    output_dir = tmp_path / "cli_outputs"

    # Mock sys.argv
    monkeypatch.setattr(
        "sys.argv",
        [
            "prompt_metrics",
            "--dataset", str(temp_dataset),
            "--output-dir", str(output_dir),
            "--evaluators", "exact_match,keyword",
            "--formats", "csv,json,md",
        ],
    )

    exit_code = main()

    # ---- Assertions ----
    assert exit_code == 0, "CLI should exit with code 0 on success"

    # All three output files should exist
    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"
    md_path = output_dir / "report.md"

    assert json_path.exists(), "results.json was not created"
    assert csv_path.exists(), "results.csv was not created"
    assert md_path.exists(), "report.md was not created"

    # JSON should contain both test cases
    with open(json_path, encoding="utf-8") as f:
        json_data = json.load(f)
    assert isinstance(json_data, list)
    assert len(json_data) == 2
    assert {c["case_id"] for c in json_data} == {"cli_case_001", "cli_case_002"}

    # CSV should be parseable and contain flattened evaluator columns
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "case_id" in csv_text
    assert "evaluator_results_exact_match_score" in csv_text
    assert "evaluator_results_keyword_score" in csv_text

    # Markdown report should contain summary + both cases
    md_text = md_path.read_text(encoding="utf-8")
    assert "Summary" in md_text
    assert "Evaluator Breakdown" in md_text
    assert "cli_case_001" in md_text
    assert "cli_case_002" in md_text

    # Stdout should contain progress / completion messages
    captured = capsys.readouterr()
    assert "Loaded 2 test case" in captured.out
    assert "Evaluators: exact_match, keyword" in captured.out
    assert "Suite complete" in captured.out
    assert "Outputs written" in captured.out


# ---------------------------------------------------------------------------
# Test 2: Argument validation — missing required --dataset
# ---------------------------------------------------------------------------

def test_cli_missing_dataset_exits_with_code_2(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Verify argparse correctly rejects invocations missing the required
    --dataset argument: exit code 2, error message to stderr.
    """
    from src.prompt_metrics.cli import main

    # Call main() with no --dataset argument
    # argparse.error() calls sys.exit(2), which raises SystemExit
    with pytest.raises(SystemExit) as exc_info:
        main(["--output-dir", "/tmp/irrelevant"])

    assert exc_info.value.code == 2, (
        "argparse should exit with code 2 for missing required arguments"
    )

    # Argparse prints usage / error to stderr
    captured = capsys.readouterr()
    stderr = captured.err.lower()
    assert "dataset" in stderr, f"Expected 'dataset' in stderr, got: {captured.err!r}"
    assert any(
        kw in stderr for kw in ["required", "error", "usage"]
    ), f"Expected argparse error/help context in stderr, got: {captured.err!r}"


def test_cli_unknown_evaluator_exits_with_code_2(
    temp_dataset: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Verify the CLI rejects unknown evaluator names with exit code 2.
    """
    from src.prompt_metrics.cli import main

    exit_code = main([
        "--dataset", str(temp_dataset),
        "--output-dir", str(tmp_path / "out"),
        "--evaluators", "exact_match,not_a_real_evaluator",
    ])

    assert exit_code == 2, "CLI should exit with code 2 for unknown evaluator"

    captured = capsys.readouterr()
    stderr = captured.err.lower()
    assert "unknown evaluator" in stderr
    assert "not_a_real_evaluator" in stderr


# ---------------------------------------------------------------------------
# Test 3: Custom generator loading
# ---------------------------------------------------------------------------

def test_cli_custom_generator_resolution(
    temp_dataset: Path,
    tmp_path: Path,
) -> None:
    """
    Verify --generator module:callable resolution:
      --generator tests.test_cli:dummy_generator
    should import and use dummy_generator() instead of the built-in mock.

    The dummy_generator returns "DUMMY_GENERATED::<PROMPT_UPPER>",
    which is easily distinguishable from the mock "[MOCK] <prompt>" output.
    """
    from src.prompt_metrics.cli import main

    output_dir = tmp_path / "gen_test"

    exit_code = main([
        "--dataset", str(temp_dataset),
        "--output-dir", str(output_dir),
        "--evaluators", "exact_match",
        "--formats", "json",
        "--generator", "tests.test_cli:dummy_generator",
    ])

    assert exit_code == 0, "CLI should succeed with custom generator"

    # Verify the custom generator was actually used
    json_path = output_dir / "results.json"
    assert json_path.exists()

    with open(json_path, encoding="utf-8") as f:
        results = json.load(f)

    # Every generated_response should start with "DUMMY_GENERATED::"
    # and contain the uppercased prompt — this is the dummy_generator signature
    for case in results:
        response = case["generated_response"]
        prompt = case["input_prompt"]

        assert response.startswith("DUMMY_GENERATED::"), (
            f"Custom generator was not used! "
            f"Expected response starting with 'DUMMY_GENERATED::', "
            f"got {response!r}"
        )
        assert prompt.upper() in response, (
            f"dummy_generator should echo uppercased prompt. "
            f"Prompt: {prompt!r}, response: {response!r}"
        )

    # Sanity check: ensure this is NOT the default mock generator output
    for case in results:
        assert "[MOCK]" not in case["generated_response"], (
            "CLI fell back to default mock generator instead of using "
            "the custom --generator"
        )


def test_cli_generator_import_error_exits_with_code_2(
    temp_dataset: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Verify that a bogus --generator spec fails gracefully with exit code 2
    and a helpful error message.
    """
    from src.prompt_metrics.cli import main

    exit_code = main([
        "--dataset", str(temp_dataset),
        "--output-dir", str(tmp_path / "out"),
        "--generator", "nonexistent_module:fake_fn",
    ])

    assert exit_code == 2

    captured = capsys.readouterr()
    assert "import" in captured.err.lower() or "generator" in captured.err.lower()


def test_cli_generator_missing_colon_is_rejected(
    temp_dataset: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Verify --generator without a ':' separator is rejected with a clear error.
    """
    from src.prompt_metrics.cli import main

    exit_code = main([
        "--dataset", str(temp_dataset),
        "--output-dir", str(tmp_path / "out"),
        "--generator", "just_a_module_no_colon",
    ])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "module:callable" in captured.err.lower()
