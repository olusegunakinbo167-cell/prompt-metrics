# tests/test_curation.py
"""
Tests for prompt_metrics.curation — interactive test case curation.

Covers:
  - Accept / reject / skip flow
  - Edit workflow with field-by-field updates
  - Final filtered list correctness
  - CLI --interactive flag wiring
"""

from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from prompt_metrics.curation import CurationReviewer
from prompt_metrics.cli import main as cli_main


# ---------------------------------------------------------------------------
# Test: accept / reject / skip
# ---------------------------------------------------------------------------

def test_curation_accept_reject_skip():
    """Reviewer should keep accepted cases, drop rejected/skipped."""
    cases = [
        {"id": "c1", "input_prompt": "q1", "expected_text": "a1", "keywords": ["x"]},
        {"id": "c2", "input_prompt": "q2", "expected_text": "a2", "keywords": ["y"]},
        {"id": "c3", "input_prompt": "q3", "expected_text": "a3", "keywords": ["z"]},
    ]

    # Simulate user input: accept c1, reject c2, skip c3
    inputs = iter(["a", "r", "s"])
    mock_input = Mock(side_effect=lambda prompt="": next(inputs))
    printed: list[str] = []

    reviewer = CurationReviewer(
        cases,
        input_fn=mock_input,
        print_fn=lambda s="": printed.append(s),
    )
    curated = reviewer.review_interactively()

    # Only c1 should be kept
    assert len(curated) == 1
    assert curated[0]["id"] == "c1"
    assert curated[0]["input_prompt"] == "q1"


# ---------------------------------------------------------------------------
# Test: edit workflow
# ---------------------------------------------------------------------------

def test_curation_edit_case():
    """Edit should allow updating input_prompt, expected_text, keywords."""
    cases = [
        {
            "id": "edit_me",
            "input_prompt": "original question?",
            "expected_text": "original answer",
            "keywords": ["foo", "bar"],
        }
    ]

    # User flow: choose "edit", then update all 3 fields
    # Action prompt, then 3 field prompts
    inputs = iter([
        "e",  # action: edit
        "edited question!",  # new input_prompt
        "edited answer",  # new expected_text
        "new, keywords, here",  # new keywords (comma-separated)
    ])
    mock_input = Mock(side_effect=lambda prompt="": next(inputs))
    printed: list[str] = []

    reviewer = CurationReviewer(
        cases,
        input_fn=mock_input,
        print_fn=lambda s="": printed.append(s),
    )
    curated = reviewer.review_interactively()

    assert len(curated) == 1
    assert curated[0]["input_prompt"] == "edited question!"
    assert curated[0]["expected_text"] == "edited answer"
    assert curated[0]["keywords"] == ["new", "keywords", "here"]
    assert curated[0]["id"] == "edit_me"  # ID preserved


# ---------------------------------------------------------------------------
# Test: edit with keep-current (Enter to skip field)
# ---------------------------------------------------------------------------

def test_curation_edit_keep_some_fields():
    """Pressing Enter during edit should keep the current field value."""
    cases = [
        {
            "id": "partial",
            "input_prompt": "keep this",
            "expected_text": "change this",
            "keywords": ["keep", "these"],
        }
    ]

    inputs = iter([
        "edit",  # action (test full word too)
        "",  # keep input_prompt (Enter)
        "new answer",  # change expected_text
        "",  # keep keywords (Enter)
    ])
    mock_input = Mock(side_effect=lambda prompt="": next(inputs))

    reviewer = CurationReviewer(cases, input_fn=mock_input, print_fn=lambda s="": None)
    curated = reviewer.review_interactively()

    assert len(curated) == 1
    assert curated[0]["input_prompt"] == "keep this"  # unchanged
    assert curated[0]["expected_text"] == "new answer"  # changed
    assert curated[0]["keywords"] == ["keep", "these"]  # unchanged


# ---------------------------------------------------------------------------
# Test: edit cancel with :q
# ---------------------------------------------------------------------------

def test_curation_edit_cancel():
    """Entering :q during edit should cancel and re-prompt for action."""
    cases = [
        {"id": "x", "input_prompt": "q", "expected_text": "a", "keywords": []}
    ]

    # User: edit → cancel at first field (:q) → then accept
    inputs = iter(["e", ":q", "a"])
    mock_input = Mock(side_effect=lambda prompt="": next(inputs))

    reviewer = CurationReviewer(cases, input_fn=mock_input, print_fn=lambda s="": None)
    curated = reviewer.review_interactively()

    # Case should be accepted with original values (edit was cancelled)
    assert len(curated) == 1
    assert curated[0]["input_prompt"] == "q"
    assert curated[0]["expected_text"] == "a"


# ---------------------------------------------------------------------------
# Test: all rejected
# ---------------------------------------------------------------------------

def test_curation_all_rejected():
    """If all cases are rejected, return empty list."""
    cases = [
        {"id": "r1", "input_prompt": "q1", "expected_text": "a1", "keywords": []},
        {"id": "r2", "input_prompt": "q2", "expected_text": "a2", "keywords": []},
    ]
    inputs = iter(["reject", "r"])  # test full word "reject" too
    mock_input = Mock(side_effect=lambda prompt="": next(inputs))

    reviewer = CurationReviewer(cases, input_fn=mock_input, print_fn=lambda s="": None)
    curated = reviewer.review_interactively()

    assert curated == []


# ---------------------------------------------------------------------------
# Test: CLI --interactive flag
# ---------------------------------------------------------------------------

def test_cli_synthesize_interactive(tmp_path):
    """CLI --interactive should invoke CurationReviewer."""
    import sys
    import types

    # Fake model that generates 2 cases
    fake_module = types.ModuleType("fake_curate_model")

    def fake_model(prompt: str) -> str:
        return json.dumps([
            {"input_prompt": "q1", "expected_text": "a1", "keywords": ["x"]},
            {"input_prompt": "q2", "expected_text": "a2", "keywords": ["y"]},
        ])

    fake_module.gen = fake_model
    sys.modules["fake_curate_model"] = fake_module

    # Mock input: accept first case, reject second
    from unittest.mock import patch

    output_path = tmp_path / "curated.json"

    try:
        with patch("builtins.input", side_effect=["a", "r"]):
            exit_code = cli_main([
                "synthesize",
                "--description", "Test task",
                "--num-cases", "2",
                "--model", "fake_curate_model:gen",
                "--output", str(output_path),
                "--interactive",
            ])

        assert exit_code == 0
        assert output_path.exists()

        data = json.loads(output_path.read_text(encoding="utf-8"))
        # Only 1 case should be in output (q1 accepted, q2 rejected)
        assert len(data) == 1
        assert data[0]["input_prompt"] == "q1"

    finally:
        sys.modules.pop("fake_curate_model", None)
