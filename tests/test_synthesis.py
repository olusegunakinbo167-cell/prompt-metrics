# tests/test_synthesis.py
"""
Tests for prompt_metrics.synthesis — synthetic dataset generation.

Covers:
  - DatasetSynthesizer prompt construction
  - Model execution with mocked model_client
  - JSON output parsing (direct, markdown-fenced, embedded)
  - Field normalization (flexible field names)
  - Case ID assignment
  - CLI synthesize subcommand
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from prompt_metrics.synthesis import DatasetSynthesizer
from prompt_metrics.cli import main as cli_main


# ---------------------------------------------------------------------------
# Test: basic synthesis with mocked model
# ---------------------------------------------------------------------------

def test_synthesizer_generates_dataset():
    """DatasetSynthesizer should call model_client and parse JSON output."""
    # Mock model returns 2 test cases as JSON
    mock_response = json.dumps([
        {
            "input_prompt": "What is 2+2?",
            "expected_text": "4",
            "keywords": ["math", "addition", "four"]
        },
        {
            "input_prompt": "Explain photosynthesis.",
            "expected_text": "Photosynthesis is the process by which plants convert light into chemical energy.",
            "keywords": ["plants", "light", "energy", "chlorophyll"]
        }
    ])
    mock_client = Mock(return_value=mock_response)

    synth = DatasetSynthesizer(model_client=mock_client, case_id_prefix="test")
    dataset = synth.generate_dataset(
        seed_prompts=["What is 1+1?", "Tell me about biology."],
        description="A general knowledge Q&A bot.",
        num_cases=2,
    )

    # Verify model_client was called once
    assert mock_client.call_count == 1
    # Verify the prompt sent to the model contains key elements
    model_prompt = mock_client.call_args[0][0]
    assert "general knowledge q&a bot" in model_prompt.lower()
    assert "What is 1+1?" in model_prompt
    assert "diverse" in model_prompt.lower()
    assert "edge cases" in model_prompt.lower() or "failure modes" in model_prompt.lower()

    # Verify dataset structure
    assert len(dataset) == 2
    assert dataset[0]["id"] == "test_001"
    assert dataset[1]["id"] == "test_002"
    assert dataset[0]["input_prompt"] == "What is 2+2?"
    assert dataset[0]["expected_text"] == "4"
    assert dataset[0]["keywords"] == ["math", "addition", "four"]


# ---------------------------------------------------------------------------
# Test: parsing with markdown fences
# ---------------------------------------------------------------------------

def test_synthesizer_parses_markdown_fenced_json():
    """Should strip ```json fences and parse correctly."""
    mock_response = """```json
[
  {"input_prompt": "test query", "expected_text": "test answer", "keywords": ["test"]}
]
```"""
    mock_client = Mock(return_value=mock_response)

    synth = DatasetSynthesizer(model_client=mock_client)
    dataset = synth.generate_dataset(
        seed_prompts=[],
        description="Test task",
        num_cases=1,
    )

    assert len(dataset) == 1
    assert dataset[0]["input_prompt"] == "test query"
    assert dataset[0]["expected_text"] == "test answer"


# ---------------------------------------------------------------------------
# Test: parsing with embedded JSON array
# ---------------------------------------------------------------------------

def test_synthesizer_parses_embedded_json():
    """Should extract JSON array from surrounding prose."""
    mock_response = """Here are the test cases you requested:

[
  {"input_prompt": "query 1", "expected_text": "answer 1", "keywords": ["a", "b"]},
  {"input_prompt": "query 2", "expected_text": "answer 2", "keywords": ["c", "d"]}
]

Let me know if you need more!"""
    mock_client = Mock(return_value=mock_response)

    synth = DatasetSynthesizer(model_client=mock_client)
    dataset = synth.generate_dataset(
        seed_prompts=[],
        description="Test",
        num_cases=2,
    )

    assert len(dataset) == 2
    assert dataset[0]["input_prompt"] == "query 1"
    assert dataset[1]["input_prompt"] == "query 2"


# ---------------------------------------------------------------------------
# Test: field name normalization
# ---------------------------------------------------------------------------

def test_synthesizer_normalizes_field_names():
    """Should accept flexible field names: prompt/input/query, output/answer, etc."""
    mock_response = json.dumps([
        # Variation 1: prompt / answer / keyword (singular)
        {"prompt": "q1", "answer": "a1", "keyword": "x, y, z"},
        # Variation 2: input / output / keywords
        {"input": "q2", "output": "a2", "keywords": ["p", "q"]},
        # Variation 3: query / expected / keywords
        {"query": "q3", "expected": "a3", "keywords": ["r"]},
        # Variation 4: question / reference
        {"question": "q4", "reference": "a4", "keywords": []},
    ])
    mock_client = Mock(return_value=mock_response)

    synth = DatasetSynthesizer(model_client=mock_client)
    dataset = synth.generate_dataset([], "Test", num_cases=4)

    assert len(dataset) == 4
    # All should be normalized to input_prompt / expected_text / keywords
    assert dataset[0]["input_prompt"] == "q1"
    assert dataset[0]["expected_text"] == "a1"
    assert dataset[0]["keywords"] == ["x", "y", "z"]  # comma-separated string → list

    assert dataset[1]["input_prompt"] == "q2"
    assert dataset[1]["expected_text"] == "a2"
    assert dataset[1]["keywords"] == ["p", "q"]

    assert dataset[2]["input_prompt"] == "q3"
    assert dataset[2]["expected_text"] == "a3"

    assert dataset[3]["input_prompt"] == "q4"
    assert dataset[3]["expected_text"] == "a4"
    assert dataset[3]["keywords"] == []


# ---------------------------------------------------------------------------
# Test: case ID assignment and num_cases trimming
# ---------------------------------------------------------------------------

def test_synthesizer_trims_to_num_cases():
    """Should trim excess cases and assign sequential IDs."""
    # Model returns 5 cases, but we only request 3
    mock_response = json.dumps([
        {"input_prompt": f"q{i}", "expected_text": f"a{i}", "keywords": []}
        for i in range(1, 6)
    ])
    mock_client = Mock(return_value=mock_response)

    synth = DatasetSynthesizer(model_client=mock_client, case_id_prefix="custom")
    dataset = synth.generate_dataset([], "Test", num_cases=3)

    assert len(dataset) == 3
    assert [c["id"] for c in dataset] == ["custom_001", "custom_002", "custom_003"]
    assert dataset[0]["input_prompt"] == "q1"
    assert dataset[2]["input_prompt"] == "q3"
    # q4, q5 should be trimmed


# ---------------------------------------------------------------------------
# Test: model_client error handling
# ---------------------------------------------------------------------------

def test_synthesizer_handles_model_error():
    """Should raise RuntimeError with context if model_client fails."""
    mock_client = Mock(side_effect=ValueError("API rate limit exceeded"))

    synth = DatasetSynthesizer(model_client=mock_client)

    with pytest.raises(RuntimeError, match="model_client failed.*rate limit"):
        synth.generate_dataset([], "Test", num_cases=1)


# ---------------------------------------------------------------------------
# Test: empty / invalid model output
# ---------------------------------------------------------------------------

def test_synthesizer_raises_on_empty_output():
    """Should raise if synthesis produces no valid cases."""
    mock_client = Mock(return_value="Sorry, I can't help with that.")

    synth = DatasetSynthesizer(model_client=mock_client)

    with pytest.raises(RuntimeError, match="produced no valid test cases"):
        synth.generate_dataset([], "Test", num_cases=1)


# ---------------------------------------------------------------------------
# Test: CLI synthesize subcommand
# ---------------------------------------------------------------------------

def test_cli_synthesize_subcommand(tmp_path: Path, monkeypatch):
    """CLI synthesize should generate a dataset JSON file."""
    # Create a fake model module
    import sys
    import types

    fake_module = types.ModuleType("fake_synth_model")

    def fake_model(prompt: str) -> str:
        # Verify the synthesis prompt was constructed correctly
        assert "SQL query generator" in prompt
        assert "SELECT" in prompt  # seed prompt should be in there
        return json.dumps([
            {"input_prompt": "Find all users", "expected_text": "SELECT * FROM users;", "keywords": ["users", "select"]},
            {"input_prompt": "Count orders", "expected_text": "SELECT COUNT(*) FROM orders;", "keywords": ["orders", "count"]},
        ])

    fake_module.generate = fake_model
    sys.modules["fake_synth_model"] = fake_module

    try:
        output_path = tmp_path / "out.json"

        # Run: prompt_metrics synthesize --description ... --seed-prompts ... --num-cases 2 --model fake_synth_model:generate --output out.json
        exit_code = cli_main([
            "synthesize",
            "--description", "A SQL query generator",
            "--seed-prompts", "SELECT * FROM products", "Find revenue",
            "--num-cases", "2",
            "--model", "fake_synth_model:generate",
            "--output", str(output_path),
        ])

        assert exit_code == 0
        assert output_path.exists()

        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert len(data) == 2
        assert data[0]["id"] == "synth_001"
        assert data[0]["input_prompt"] == "Find all users"
        assert data[0]["expected_text"] == "SELECT * FROM users;"
        assert "users" in data[0]["keywords"]

    finally:
        sys.modules.pop("fake_synth_model", None)


# ---------------------------------------------------------------------------
# Test: CLI synthesize with seed prompts file
# ---------------------------------------------------------------------------

def test_cli_synthesize_with_seed_file(tmp_path: Path):
    """CLI should accept --seed-prompts-file."""
    import sys
    import types

    # Create seed file
    seed_file = tmp_path / "seeds.txt"
    seed_file.write_text("seed one\n\nseed two\n  seed three  \n", encoding="utf-8")

    # Fake model that echoes back seed prompts it received
    fake_module = types.ModuleType("fake_synth_model2")
    captured_prompt = {}

    def fake_model(prompt: str) -> str:
        captured_prompt["text"] = prompt
        return json.dumps([
            {"input_prompt": "generated", "expected_text": "output", "keywords": []}
        ])

    fake_module.gen = fake_model
    sys.modules["fake_synth_model2"] = fake_module

    try:
        output_path = tmp_path / "out2.json"
        exit_code = cli_main([
            "synthesize",
            "--description", "Test task",
            "--seed-prompts-file", str(seed_file),
            "--num-cases", "1",
            "--model", "fake_synth_model2:gen",
            "--output", str(output_path),
        ])

        assert exit_code == 0
        # Verify seed prompts were read and passed to the model
        model_prompt = captured_prompt["text"]
        assert "seed one" in model_prompt
        assert "seed two" in model_prompt
        assert "seed three" in model_prompt

    finally:
        sys.modules.pop("fake_synth_model2", None)
