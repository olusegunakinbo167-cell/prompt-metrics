#!/usr/bin/env python3
"""
eval_harness.py — Prompt evaluation harness with token usage tracking.

Runs a prompt template against a test dataset, evaluates outputs,
and records token usage + estimated cost.

Usage:
    python eval_harness.py --prompt prompts/v1_concise.txt \\
        --output latest_metrics_v1_concise.json --seed 42

    python eval_harness.py --prompt prompts/v2_detailed.txt \\
        --dataset data/test_cases.json \\
        --model gpt-4o-mini \\
        --generator my_llm:generate \\
        --output results.json

Output JSON includes:
{
  "summary": {
    "total_cases": 10,
    "successful_cases": 10,
    ...
    "token_usage": {
      "prompt_tokens": 4523,
      "completion_tokens": 1821,
      "total_tokens": 6344,
      "estimated_cost": 0.001771,
      "model": "gpt-4o-mini",
      "pricing": {"input_per_1m": 0.15, "output_per_1m": 0.6}
    }
  },
  "token_usage": { ... },
  "results": [...]
}
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable

# Make src/ importable when running as a script
sys.path.insert(0, str(Path(__file__).parent / "src"))

from prompt_metrics.runner import ExperimentRunner, load_dataset, SuiteResult
from prompt_metrics.evaluators import (
    ExactMatchEvaluator,
    KeywordEvaluator,
    ContainsEvaluator,
    SemanticSimilarityEvaluator,
)

# ---------------------------------------------------------------------------
# Model pricing table (USD per 1M tokens)
# Sources: OpenAI / Anthropic public pricing, update as needed
# ---------------------------------------------------------------------------

MODEL_PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"input_per_1m": 2.50, "output_per_1m": 10.00},
    "gpt-4o-2024-08-06": {"input_per_1m": 2.50, "output_per_1m": 10.00},
    "gpt-4o-mini": {"input_per_1m": 0.15, "output_per_1m": 0.60},
    "gpt-4o-mini-2024-07-18": {"input_per_1m": 0.15, "output_per_1m": 0.60},
    "gpt-4-turbo": {"input_per_1m": 10.00, "output_per_1m": 30.00},
    "gpt-3.5-turbo": {"input_per_1m": 0.50, "output_per_1m": 1.50},
    # Anthropic
    "claude-3-5-sonnet-20241022": {"input_per_1m": 3.00, "output_per_1m": 15.00},
    "claude-3-5-sonnet": {"input_per_1m": 3.00, "output_per_1m": 15.00},
    "claude-3-5-haiku-20241022": {"input_per_1m": 0.80, "output_per_1m": 4.00},
    "claude-3-5-haiku": {"input_per_1m": 0.80, "output_per_1m": 4.00},
    "claude-3-opus-20240229": {"input_per_1m": 15.00, "output_per_1m": 75.00},
    "claude-3-opus": {"input_per_1m": 15.00, "output_per_1m": 75.00},
    # Google
    "gemini-1.5-pro": {"input_per_1m": 1.25, "output_per_1m": 5.00},
    "gemini-1.5-flash": {"input_per_1m": 0.075, "output_per_1m": 0.30},
    # Mock / local (free)
    "mock": {"input_per_1m": 0.0, "output_per_1m": 0.0},
}

def get_pricing(model: str) -> dict[str, float]:
    """Get pricing for a model, with fuzzy matching."""
    # Exact match
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]
    # Prefix match (e.g. gpt-4o-mini-2024-07-18 matches gpt-4o-mini)
    ml = model.lower()
    for key, pricing in MODEL_PRICING.items():
        if ml.startswith(key.lower()) or key.lower() in ml:
            return pricing
    # Default: gpt-4o-mini pricing (cheap, conservative)
    print(f"Warning: no pricing found for model {model!r}, using gpt-4o-mini rates", file=sys.stderr)
    return MODEL_PRICING["gpt-4o-mini"]

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

class TokenCounter:
    """Token counter with tiktoken → fallback chain."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._enc = None
        try:
            import tiktoken  # type: ignore
            try:
                self._enc = tiktoken.encoding_for_model(model)
            except Exception:
                # Fall back to cl100k_base (used by gpt-4 / gpt-3.5-turbo)
                self._enc = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            self._enc = None

    def count(self, text: str) -> int:
        """Count tokens in text."""
        if self._enc is not None:
            return len(self._enc.encode(text))
        # Fallback: ~4 chars per token, or ~0.75 words per token
        # Use max of both heuristics to be conservative
        char_est = len(text) / 4.0
        word_est = len(text.split()) * 1.3
        return int(max(char_est, word_est, 1))

_token_counter: TokenCounter | None = None

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    global _token_counter
    if _token_counter is None or _token_counter.model != model:
        _token_counter = TokenCounter(model)
    return _token_counter.count(text)

# ---------------------------------------------------------------------------
# Generator resolution
# ---------------------------------------------------------------------------

def resolve_callable(spec: str) -> Callable[[str], str]:
    """Resolve 'module:callable' → callable."""
    if ":" not in spec:
        raise ValueError(f"Generator spec {spec!r} must be 'module:callable'")
    module_name, attr = spec.split(":", 1)
    mod = importlib.import_module(module_name)
    obj: Any = mod
    for part in attr.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise TypeError(f"Generator {spec!r} is not callable")
    return obj  # type: ignore

def mock_generator(prompt: str) -> str:
    """Deterministic mock generator for testing without API keys."""
    # Simple deterministic response based on prompt hash
    h = abs(hash(prompt)) % 10000
    responses = [
        "The capital of France is Paris, known for the Eiffel Tower.",
        "Quantum entanglement is when particles remain connected across distance.",
        "Machine learning models learn patterns from training data to make predictions.",
        "The quick brown fox jumps over the lazy dog in a demonstration of typography.",
        "Water boils at 100°C at standard atmospheric pressure.",
    ]
    # Pick based on hash, add some prompt-specific flavor
    base = responses[h % len(responses)]
    if len(prompt) > 50:
        return f"{base} This response addresses the query comprehensively with relevant context."
    return base

# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Prompt evaluation harness with token usage tracking")
    ap.add_argument("--prompt", required=True, help="Path to prompt template file (.txt)")
    ap.add_argument("--output", "-o", required=True, help="Output metrics JSON path")
    ap.add_argument("--dataset", help="Path to test dataset JSON (default: auto-discover or built-in)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    ap.add_argument("--model", default="gpt-4o-mini", help="Model name for token counting and cost estimation (default: gpt-4o-mini)")
    ap.add_argument("--generator", help='Generator function as "module:callable", e.g. "my_llm:generate". Default: mock generator')
    ap.add_argument("--evaluators", default="exact_match,keyword,contains,semantic", help="Comma-separated evaluator list")
    args = ap.parse_args()

    random.seed(args.seed)

    # --- Load prompt template ---
    prompt_path = Path(args.prompt)
    if not prompt_path.exists():
        print(f"Error: prompt file not found: {prompt_path}", file=sys.stderr)
        return 2
    prompt_template = prompt_path.read_text(encoding="utf-8")
    print(f"Loaded prompt template: {prompt_path} ({len(prompt_template)} chars)")

    # --- Load dataset ---
    dataset = None
    dataset_path = None
    if args.dataset:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            print(f"Error: dataset file not found: {dataset_path}", file=sys.stderr)
            return 2
        dataset = load_dataset(str(dataset_path))
        print(f"Loaded {len(dataset)} test case(s) from {dataset_path}")
    else:
        # Auto-discover common dataset locations
        for candidate in [
            "data/test_cases.json",
            "test_cases.json",
            "datasets/test_cases.json",
            "eval_dataset.json",
        ]:
            if Path(candidate).exists():
                dataset_path = Path(candidate)
                dataset = load_dataset(str(dataset_path))
                print(f"Loaded {len(dataset)} test case(s) from {dataset_path} (auto-discovered)")
                break
    if dataset is None:
        # Fall back to built-in minimal dataset
        print("No dataset file found — using built-in minimal dataset (5 cases)", file=sys.stderr)
        from prompt_metrics.runner import TestCase
        dataset = [
            TestCase(id="case_001", input_prompt="What is the capital of France?",
                     expected_text="Paris", keywords=["Paris", "France"]),
            TestCase(id="case_002", input_prompt="Explain quantum entanglement in one sentence.",
                     expected_text=None, keywords=["entangled", "particles", "quantum"]),
            TestCase(id="case_003", input_prompt="What is 2+2?",
                     expected_text="4", keywords=["4", "four"]),
            TestCase(id="case_004", input_prompt="Name a programming language.",
                     expected_text=None, keywords=["Python", "JavaScript", "Java", "C++", "Rust", "Go"]),
            TestCase(id="case_005", input_prompt="What is the boiling point of water?",
                     expected_text=None, keywords=["100", "boil", "water", "Celsius"]),
        ]

    # --- Set up evaluators ---
    evaluator_registry = {
        "exact_match": ExactMatchEvaluator,
        "exact": ExactMatchEvaluator,
        "keyword": KeywordEvaluator,
        "keywords": KeywordEvaluator,
        "contains": ContainsEvaluator,
        "semantic": SemanticSimilarityEvaluator,
        "semantic_similarity": SemanticSimilarityEvaluator,
    }
    requested = [e.strip() for e in args.evaluators.split(",") if e.strip()]
    evaluators = []
    for name in requested:
        if name not in evaluator_registry:
            print(f"Error: unknown evaluator {name!r}. Available: {', '.join(sorted(evaluator_registry))}", file=sys.stderr)
            return 2
        cls = evaluator_registry[name]
        try:
            # KeywordEvaluator needs keywords from test case, not ctor
            evaluators.append(cls())
        except TypeError:
            # Try with defaults
            evaluators.append(cls())
    print(f"Evaluators: {', '.join(requested)}")

    # --- Resolve generator ---
    if args.generator:
        try:
            base_generator = resolve_callable(args.generator)
            print(f"Generator: {args.generator}")
        except Exception as e:
            print(f"Error resolving generator {args.generator!r}: {e}", file=sys.stderr)
            return 2
    else:
        base_generator = mock_generator
        print("Generator: mock_generator (built-in, no API key needed)")

    # --- Token-tracking wrapper ---
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def tracked_generator(case_input: str) -> str:
        # Render full prompt
        if "{{input}}" in prompt_template:
            full_prompt = prompt_template.replace("{{input}}", case_input)
        elif "{input}" in prompt_template:
            full_prompt = prompt_template.format(input=case_input)
        else:
            # Concatenate template + case input
            full_prompt = prompt_template.rstrip() + "\n\n" + case_input

        # Count prompt tokens
        prompt_tokens = count_tokens(full_prompt, args.model)
        token_usage["prompt_tokens"] += prompt_tokens

        # Generate response
        response = base_generator(full_prompt)

        # Count completion tokens
        completion_tokens = count_tokens(response, args.model)
        token_usage["completion_tokens"] += completion_tokens

        return response

    # --- Run experiment ---
    runner = ExperimentRunner(evaluators)
    print(f"\nRunning {len(dataset)} case(s)...")
    suite = runner.run_suite(dataset, tracked_generator, continue_on_error=True, verbose=True)

    # --- Compute cost ---
    pricing = get_pricing(args.model)
    prompt_tokens = token_usage["prompt_tokens"]
    completion_tokens = token_usage["completion_tokens"]
    total_tokens = prompt_tokens + completion_tokens
    estimated_cost = (
        prompt_tokens / 1_000_000 * pricing["input_per_1m"]
        + completion_tokens / 1_000_000 * pricing["output_per_1m"]
    )

    token_usage_report = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost": round(estimated_cost, 6),
        "model": args.model,
        "pricing": pricing,
    }

    print(f"\n{'─'*60}")
    print(f"Suite complete: {suite.successful_cases}/{suite.total_cases} passed, "
          f"{suite.failed_cases} failed, {suite.total_runtime_s:.2f}s")
    print(f"Tokens: {prompt_tokens} prompt + {completion_tokens} completion = {total_tokens} total")
    print(f"Estimated cost ({args.model}): ${estimated_cost:.6f}")
    print(f"{'─'*60}")

    # --- Save results with token_usage injected ---
    result_dict = suite.to_dict()
    result_dict["token_usage"] = token_usage_report
    # Also inject into summary for easy access
    if "summary" in result_dict and isinstance(result_dict["summary"], dict):
        result_dict["summary"]["token_usage"] = token_usage_report

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)

    print(f"\nWrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
