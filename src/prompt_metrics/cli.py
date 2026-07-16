# src/prompt_metrics/cli.py
"""
Command-line interface for prompt_metrics.

Usage:
    python -m prompt_metrics.cli --dataset data/test_cases.json
    python -m prompt_metrics.cli --dataset data/test_cases.json \\
        --output-dir ./results/run_001 \\
        --evaluators exact_match,keyword \\
        --formats csv,json,md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

from .evaluators import (
    ContainsEvaluator,
    ExactMatchEvaluator,
    KeywordEvaluator,
    RegexMatchEvaluator,
)
from .export import export_results
from .reports import generate_markdown_report
from .runner import ExperimentRunner, load_dataset


# ---------------------------------------------------------------------------
# Evaluator registry
# ---------------------------------------------------------------------------

# Evaluator name → factory
EVALUATOR_REGISTRY: dict[str, Callable[[], Any]] = {
    "exact_match": ExactMatchEvaluator,
    "exact": ExactMatchEvaluator,  # alias
    "keyword": KeywordEvaluator,
    "keywords": KeywordEvaluator,  # alias
    "contains": ContainsEvaluator,
    "regex": RegexMatchEvaluator,
    "regex_match": RegexMatchEvaluator,
}


def _try_import_external_evaluators() -> None:
    """
    Optionally register evaluators from llm-eval-toolkit if it's importable.
    Registered under names: rubric, rubric_qa
    """
    try:
        from rubric_qa.evaluator import RubricEvaluator  # type: ignore
    except Exception:
        return

    def rubric_factory() -> Any:
        # Look for a rubric file in common locations, or require RUBRIC_PATH env
        import os

        rubric_path = os.environ.get("RUBRIC_PATH", "rubric.json")
        if not Path(rubric_path).exists():
            raise FileNotFoundError(
                f"RubricEvaluator requested, but rubric file not found at {rubric_path!r}. "
                "Set RUBRIC_PATH env var or place a rubric.json in the working directory."
            )
        return RubricEvaluator.from_file(rubric_path)

    EVALUATOR_REGISTRY["rubric"] = rubric_factory
    EVALUATOR_REGISTRY["rubric_qa"] = rubric_factory


# ---------------------------------------------------------------------------
# Generator resolution
# ---------------------------------------------------------------------------


def _default_generator(prompt: str) -> str:
    """
    Default generator: echoes the prompt back with a [MOCK] prefix.

    Replace this by passing --generator <module:callable>, e.g.:
        my_llm:generate
    """
    return f"[MOCK] {prompt[:200]}"


def _resolve_generator(spec: str | None) -> Callable[[str], str]:
    """
    Resolve a generator function from a spec string.

    spec formats:
      None / "mock"              → built-in mock generator
      "module:callable"          → import module.callable
      "module.sub:callable"      → import module.sub.callable
    """
    if spec is None or spec == "mock":
        return _default_generator

    if ":" not in spec:
        raise ValueError(
            f"--generator {spec!r} must be in \"module:callable\" format, "
            'e.g. "my_llm:generate"'
        )

    module_name, attr = spec.split(":", 1)
    import importlib

    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        raise ImportError(f"Could not import generator module {module_name!r}: {e}") from e

    # Support dotted attribute paths: foo.bar.baz
    obj: Any = mod
    for part in attr.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            raise AttributeError(
                f"Generator {spec!r}: attribute {part!r} not found on {obj!r}"
            ) from None

    if not callable(obj):
        raise TypeError(f"Generator {spec!r} resolved to {obj!r}, which is not callable")

    return obj  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    # Try to register external evaluators for --help text
    _try_import_external_evaluators()

    available = ", ".join(sorted(EVALUATOR_REGISTRY.keys()))

    p = argparse.ArgumentParser(
        prog="prompt_metrics",
        description="Run prompt evaluation experiments over a JSON test dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""Available evaluators: {available}

examples:
  prompt_metrics --dataset data/test_cases.json
  prompt_metrics --dataset data/test_cases.json --output-dir ./results/run_001
  prompt_metrics --dataset data/test_cases.json --evaluators exact_match,keyword
  prompt_metrics --dataset data/test_cases.json --formats csv,md
  prompt_metrics --dataset data/test_cases.json --generator my_llm:generate
""",
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="Path to the input JSON dataset file. "
        "Each entry must have: id, input_prompt. "
        "Optional: expected_text, keywords.",
    )
    p.add_argument(
        "--output-dir",
        default="./results",
        help="Destination directory for generated outputs (default: ./results).",
    )
    p.add_argument(
        "--evaluators",
        default="",
        help=(
            "Comma-separated list of evaluators to run. "
            f"Available: {available}. "
            "If omitted, runs all built-in evaluators."
        ),
    )
    p.add_argument(
        "--formats",
        default="csv,json,md",
        help="Comma-separated output formats: csv, json, md (default: all three).",
    )
    p.add_argument(
        "--generator",
        default=None,
        help=(
            "Generator function in 'module:callable' format "
            '(e.g. "my_llm:generate"). '
            "If omitted, uses a built-in mock generator. "
            "The callable must accept (prompt: str) -> str."
        ),
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first case/generator error instead of continuing.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # ---- Validate & resolve evaluators ----
    _try_import_external_evaluators()

    if args.evaluators.strip():
        requested = [e.strip() for e in args.evaluators.split(",") if e.strip()]
    else:
        # Default: all built-in evaluators (skip external ones like rubric_qa
        # unless explicitly requested, since they need a rubric file)
        requested = ["exact_match", "keyword", "contains"]

    unknown = [e for e in requested if e not in EVALUATOR_REGISTRY]
    if unknown:
        print(
            f"Error: unknown evaluator(s): {', '.join(unknown)}\n"
            f"Available: {', '.join(sorted(EVALUATOR_REGISTRY.keys()))}",
            file=sys.stderr,
        )
        return 2

    evaluators = []
    for name in requested:
        try:
            evaluators.append(EVALUATOR_REGISTRY[name]())
        except Exception as e:
            print(f"Error instantiating evaluator {name!r}: {e}", file=sys.stderr)
            return 2

    # ---- Resolve generator ----
    try:
        generator_fn = _resolve_generator(args.generator)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    # ---- Load dataset ----
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: dataset file not found: {dataset_path}", file=sys.stderr)
        return 2

    try:
        dataset = load_dataset(str(dataset_path))
    except Exception as e:
        print(f"Error loading dataset {dataset_path}: {e}", file=sys.stderr)
        return 2

    print(
        f"Loaded {len(dataset)} test case(s) from {dataset_path}\n"
        f"Evaluators: {', '.join(requested)}\n"
        f"Generator: {generator_fn.__module__}.{getattr(generator_fn, '__name__', '<callable>')}"
    )

    # ---- Run suite ----
    runner = ExperimentRunner(evaluators)
    try:
        suite = runner.run_suite(
            dataset,
            generator_fn,
            continue_on_error=not args.fail_fast,
            verbose=True,
        )
    except Exception as e:
        print(f"\nExperiment failed: {e}", file=sys.stderr)
        return 1

    print(
        f"\n"
        f"Suite complete — {suite.successful_cases}/{suite.total_cases} passed, "
        f"{suite.failed_cases} failed, "
        f"{suite.total_runtime_s:.2f}s total"
    )

    # ---- Export ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    formats = {f.strip().lower() for f in args.formats.split(",") if f.strip()}
    valid_formats = {"csv", "json", "md"}
    invalid = formats - valid_formats
    if invalid:
        print(
            f"Warning: ignoring unknown format(s): {', '.join(sorted(invalid))}\n"
            f"Valid formats: {', '.join(sorted(valid_formats))}",
            file=sys.stderr,
        )
        formats &= valid_formats

    if not formats:
        print("No valid output formats specified — nothing to write.")
        return 0

    written: list[str] = []

    if "json" in formats:
        json_path = output_dir / "results.json"
        export_results(suite, str(json_path), format="json")
        written.append(str(json_path))

    if "csv" in formats:
        csv_path = output_dir / "results.csv"
        export_results(suite, str(csv_path), format="csv")
        written.append(str(csv_path))

    if "md" in formats:
        md_path = output_dir / "report.md"
        generate_markdown_report(suite, str(md_path))
        written.append(str(md_path))

    print("\nOutputs written:")
    for path in written:
        print(f"  {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
