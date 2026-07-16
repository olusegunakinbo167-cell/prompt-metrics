# src/prompt_metrics/cli.py
"""
Command-line interface for prompt_metrics.

Usage:
    # Run an evaluation suite:
    prompt_metrics run --dataset data/test_cases.json
    prompt_metrics run --dataset data/test_cases.json \\
        --output-dir ./results/run_001 \\
        --evaluators exact_match,keyword \\
        --formats csv,json,md

    # Generate a synthetic dataset:
    prompt_metrics synthesize \\
        --description "A SQL query generator" \\
        --seed-prompts "SELECT users...", "Find all orders..." \\
        --num-cases 20 \\
        --model my_llm:generate \\
        --output dataset.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

from .evaluators import (
    ContainsEvaluator,
    CritiqueEvaluator,
    ExactMatchEvaluator,
    KeywordEvaluator,
    QAEvaluator,
    RegexMatchEvaluator,
    SemanticSimilarityEvaluator,
    RUBRIC_TEMPLATES,
)
from .export import export_results
from .reports import generate_markdown_report
from .runner import ExperimentRunner, load_dataset
from .synthesis import DatasetSynthesizer


# ---------------------------------------------------------------------------
# Callable resolution
# ---------------------------------------------------------------------------


def _resolve_callable(spec: str, kind: str = "callable") -> Callable[[str], str]:
    """
    Resolve a callable from a 'module:attribute' spec string.

    Args:
        spec: "module:callable" or "module.sub:callable.attr"
        kind: Human-readable label for error messages (e.g. "generator", "judge model")

    Returns:
        The resolved callable.

    Raises:
        ValueError, ImportError, AttributeError, TypeError with helpful messages.
    """
    if ":" not in spec:
        raise ValueError(
            f"{kind} {spec!r} must be in \"module:callable\" format, "
            'e.g. "my_llm:generate"'
        )

    module_name, attr = spec.split(":", 1)
    import importlib

    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        raise ImportError(f"Could not import {kind} module {module_name!r}: {e}") from e

    # Support dotted attribute paths: foo.bar.baz
    obj: Any = mod
    for part in attr.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            raise AttributeError(
                f"{kind.capitalize()} {spec!r}: attribute {part!r} not found on {obj!r}"
            ) from None

    if not callable(obj):
        raise TypeError(f"{kind.capitalize()} {spec!r} resolved to {obj!r}, which is not callable")

    return obj  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Evaluator registry
# ---------------------------------------------------------------------------

# Evaluator name → factory
# Factories are 0-arg callables that return an evaluator instance.
# For evaluators that need external clients (judge model, embedding model),
# the factory reads the corresponding CLI arg / env var.
EVALUATOR_REGISTRY: dict[str, Callable[[], Any]] = {}


def _make_judge_factory(
    evaluator_cls: type,
    *,
    rubric: str | None = None,
) -> Callable[[], Any]:
    """
    Create a factory function for a model-based evaluator (LLM-as-a-judge).

    The factory resolves the judge model client from, in order:
      1. --judge-model CLI argument (stored in _JUDGE_MODEL_SPEC global)
      2. JUDGE_MODEL environment variable
      3. Raises a helpful error if neither is set.
    """
    def factory() -> Any:
        # Resolve judge model client
        spec = getattr(_make_judge_factory, "_judge_model_spec", None) or os.environ.get(
            "JUDGE_MODEL"
        )
        if not spec:
            raise RuntimeError(
                f"{evaluator_cls.__name__} requires a judge model client. "
                "Pass --judge-model <module:callable> on the command line, "
                "or set the JUDGE_MODEL environment variable, e.g.:\n"
                '  export JUDGE_MODEL="my_llm:judge"\n'
                "  prompt_metrics run --dataset data.json --evaluators qa_judge --judge-model my_llm:judge\n\n"
                "The judge model callable must accept (prompt: str) -> str "
                "and return the judge LLM's raw text response."
            )
        try:
            model_client = _resolve_callable(spec, kind="judge model")
        except Exception as e:
            raise RuntimeError(f"Failed to resolve judge model {spec!r}: {e}") from e

        kwargs: dict[str, Any] = {"model_client": model_client}
        if rubric is not None:
            kwargs["rubric"] = rubric
        return evaluator_cls(**kwargs)

    return factory


def _make_semantic_factory() -> Callable[[], Any]:
    """
    Create a factory function for SemanticSimilarityEvaluator.

    The factory resolves an embedding client from, in order:
      1. --embedding-model CLI argument
      2. EMBEDDING_MODEL environment variable
      3. Falls back to built-in TF-IDF / sentence-transformers auto-detect
         (no error — local fallback is always available)
    """
    def factory() -> Any:
        # Try to resolve an explicit embedding client
        spec = getattr(_make_semantic_factory, "_embedding_model_spec", None) or os.environ.get(
            "EMBEDDING_MODEL"
        )
        if spec:
            try:
                embedding_client = _resolve_callable(spec, kind="embedding model")
            except Exception as e:
                raise RuntimeError(f"Failed to resolve embedding model {spec!r}: {e}") from e
            # Wrap to adapt signature: embedding_client(text: str) -> list[float]
            # _resolve_callable already validates it's callable
            return SemanticSimilarityEvaluator(embedding_client=embedding_client)  # type: ignore[arg-type]

        # No explicit client — use built-in fallback (TF-IDF / sentence-transformers auto-detect)
        return SemanticSimilarityEvaluator()

    return factory


def _register_builtin_evaluators() -> None:
    """Populate EVALUATOR_REGISTRY with all built-in evaluators."""
    EVALUATOR_REGISTRY.clear()
    EVALUATOR_REGISTRY.update(
        {
            # Text evaluators
            "exact_match": ExactMatchEvaluator,
            "exact": ExactMatchEvaluator,  # alias
            "keyword": KeywordEvaluator,
            "keywords": KeywordEvaluator,  # alias
            "contains": ContainsEvaluator,
            "regex": RegexMatchEvaluator,
            "regex_match": RegexMatchEvaluator,
            # Semantic evaluator
            # Uses TF-IDF fallback by default, or --embedding-model for custom embeddings
            "semantic": _make_semantic_factory(),
            "semantic_similarity": _make_semantic_factory(),
            # Model-based evaluators (LLM-as-a-judge)
            # These require --judge-model or JUDGE_MODEL env var
            "qa_judge": _make_judge_factory(QAEvaluator),
            "qa": _make_judge_factory(QAEvaluator),  # alias
            "critique_judge": _make_judge_factory(CritiqueEvaluator),
            "critique": _make_judge_factory(CritiqueEvaluator),  # alias
        }
    )
    # Register rubric-specific critique variants
    # e.g. critique_helpfulness, critique_safety, critique_conciseness, critique_correctness
    for rubric_name in RUBRIC_TEMPLATES.keys():
        key = f"critique_{rubric_name}"
        EVALUATOR_REGISTRY[key] = _make_judge_factory(
            CritiqueEvaluator, rubric=rubric_name
        )


# Populate registry at import time
_register_builtin_evaluators()


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
    return _resolve_callable(spec, kind="generator")  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    # Try to register external evaluators for --help text
    _try_import_external_evaluators()

    available = ", ".join(sorted(EVALUATOR_REGISTRY.keys()))

    p = argparse.ArgumentParser(
        prog="prompt_metrics",
        description="LLM prompt evaluation toolkit — run experiments and generate synthetic test datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = p.add_subparsers(dest="command", help="subcommand")

    # ---- run subcommand ----
    run_p = subparsers.add_parser(
        "run",
        help="Run a prompt evaluation experiment over a JSON dataset.",
        description=(
            "Run prompt evaluation experiments over a JSON test dataset.\n\n"
            "Semantic evaluator: uses TF-IDF fallback by default, or configure\n"
            "  --embedding-model <module:callable> / EMBEDDING_MODEL for custom embeddings.\n"
            "Model-based evaluators (qa_judge, critique_judge) require a judge model.\n"
            "Configure with --judge-model <module:callable> or JUDGE_MODEL env var."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""Available evaluators: {available}

examples:
  prompt_metrics run --dataset data/test_cases.json
  prompt_metrics run --dataset data/test_cases.json --output-dir ./results/run_001
  prompt_metrics run --dataset data/test_cases.json --evaluators exact_match,keyword
  prompt_metrics run --dataset data/test_cases.json --formats csv,md
  prompt_metrics run --dataset data/test_cases.json --generator my_llm:generate

  # Semantic similarity (TF-IDF fallback, zero dependencies):
  prompt_metrics run --dataset data/test_cases.json --evaluators semantic

  # Semantic similarity with custom embedding API:
  prompt_metrics run --dataset data/test_cases.json \\
    --evaluators semantic --embedding-model my_embeddings:embed

  # LLM-as-a-judge evaluation:
  prompt_metrics run --dataset data/test_cases.json \\
    --evaluators qa_judge --judge-model my_llm:judge
  prompt_metrics run --dataset data/test_cases.json \\
    --evaluators critique_helpfulness --judge-model my_llm:judge
""",
    )
    run_p.add_argument(
        "--dataset",
        required=True,
        help="Path to the input JSON dataset file. "
        "Each entry must have: id, input_prompt. "
        "Optional: expected_text, keywords.",
    )
    run_p.add_argument(
        "--output-dir",
        default="./results",
        help="Destination directory for generated outputs (default: ./results).",
    )
    run_p.add_argument(
        "--evaluators",
        default="",
        help=(
            "Comma-separated list of evaluators to run. "
            f"Available: {available}. "
            "If omitted, runs all built-in text evaluators "
            "(exact_match, keyword, contains, semantic). "
            "Model-based evaluators (qa_judge, critique_judge) require "
            "--judge-model."
        ),
    )
    run_p.add_argument(
        "--formats",
        default="csv,json,md",
        help="Comma-separated output formats: csv, json, md (default: all three).",
    )
    run_p.add_argument(
        "--generator",
        default=None,
        help=(
            "Generator function in 'module:callable' format "
            '(e.g. "my_llm:generate"). '
            "If omitted, uses a built-in mock generator. "
            "The callable must accept (prompt: str) -> str."
        ),
    )
    run_p.add_argument(
        "--judge-model",
        default=None,
        help=(
            "Judge model client for LLM-as-a-judge evaluators "
            "(qa_judge, critique_judge), in 'module:callable' format. "
            'e.g. "my_llm:judge". '
            "The callable must accept (prompt: str) -> str and return "
            "the judge LLM's raw text response. "
            "Can also be set via JUDGE_MODEL environment variable."
        ),
    )
    run_p.add_argument(
        "--embedding-model",
        default=None,
        help=(
            "Embedding client for semantic similarity evaluator, "
            "in 'module:callable' format. e.g. \"my_embeddings:embed\". "
            "The callable must accept (text: str) -> list[float] and return "
            "an embedding vector. "
            "If omitted, uses built-in TF-IDF fallback (zero dependencies) "
            "or auto-detects sentence-transformers if installed. "
            "Can also be set via EMBEDDING_MODEL environment variable."
        ),
    )
    run_p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first case/generator error instead of continuing.",
    )

    # ---- synthesize subcommand ----
    synth_p = subparsers.add_parser(
        "synthesize",
        help="Generate a synthetic evaluation dataset using an LLM.",
        description=(
            "Generate a synthetic test dataset for prompt evaluation.\n\n"
            "Uses an LLM to expand seed examples + a task description "
            "into a diverse evaluation suite with varied inputs, "
            "expected outputs, and keywords.\n\n"
            "The model client must accept (prompt: str) -> str and return "
            "the LLM's raw text response. Configure with --model or "
            "SYNTHESIS_MODEL env var."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  # Generate 20 test cases for a SQL generator:
  prompt_metrics synthesize \\
    --description "A SQL query generator that converts natural language to PostgreSQL" \\
    --seed-prompts "Find all users who signed up last week" "List orders over $100" \\
    --num-cases 20 \\
    --model my_llm:generate \\
    --output sql_test_cases.json

  # Generate from a description only (no seeds):
  prompt_metrics synthesize \\
    --description "A customer support chatbot that answers questions about order status, returns, and shipping" \\
    --num-cases 15 \\
    --model my_llm:generate \\
    --output support_test_cases.json

  # Use seed prompts from a file (one per line):
  prompt_metrics synthesize \\
    --description "Text summarization tool" \\
    --seed-prompts-file ./seeds.txt \\
    --num-cases 30 \\
    --model my_llm:generate \\
    --output summary_test_cases.json
""",
    )
    synth_p.add_argument(
        "--description",
        required=True,
        help=(
            "Description of the application / prompt being tested. "
            "Be specific — this drives the diversity and relevance of "
            "generated cases. E.g. \"A SQL query generator that converts "
            "natural language to PostgreSQL\" or \"A customer support "
            "chatbot that answers questions about order status\"."
        ),
    )
    synth_seed_group = synth_p.add_mutually_exclusive_group()
    synth_seed_group.add_argument(
        "--seed-prompts",
        nargs="*",
        default=[],
        metavar="PROMPT",
        help=(
            "Seed example prompts / inputs (space-separated). "
            "These guide style and structure. Can be omitted if "
            "--description is detailed enough."
        ),
    )
    synth_seed_group.add_argument(
        "--seed-prompts-file",
        metavar="PATH",
        help=(
            "Path to a text file with seed prompts, one per line. "
            "Empty lines are ignored."
        ),
    )
    synth_p.add_argument(
        "--num-cases",
        type=int,
        default=10,
        help="Number of test cases to generate (default: 10).",
    )
    synth_p.add_argument(
        "--model",
        default=None,
        help=(
            "Model client for dataset synthesis, in 'module:callable' format. "
            'e.g. "my_llm:generate". '
            "The callable must accept (prompt: str) -> str and return "
            "the model's raw text response. "
            "Can also be set via SYNTHESIS_MODEL environment variable."
        ),
    )
    synth_p.add_argument(
        "--output",
        "-o",
        default="synthetic_dataset.json",
        help="Output path for the generated dataset JSON (default: synthetic_dataset.json).",
    )
    synth_p.add_argument(
        "--case-id-prefix",
        default="synth",
        help="Prefix for generated case IDs (default: synth → synth_001, synth_002, ...).",
    )

    return p


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    """Execute the `run` subcommand."""
    # ---- Store judge_model / embedding_model specs for evaluator factories ----
    # Evaluator factories read these globals to resolve their clients
    if args.judge_model:
        setattr(_make_judge_factory, "_judge_model_spec", args.judge_model)
    if args.embedding_model:
        setattr(_make_semantic_factory, "_embedding_model_spec", args.embedding_model)

    # Re-register evaluators so semantic factory picks up the --embedding-model arg
    # (factory closures capture the spec at instantiation time)
    _register_builtin_evaluators()
    _try_import_external_evaluators()

    # ---- Validate & resolve evaluators ----
    if args.evaluators.strip():
        requested = [e.strip() for e in args.evaluators.split(",") if e.strip()]
    else:
        # Default: all built-in TEXT evaluators only
        # (skip model-based judges and external evaluators like rubric_qa
        # unless explicitly requested, since they need a judge model / rubric file)
        # Semantic evaluator is included in the default set — TF-IDF fallback is free
        requested = ["exact_match", "keyword", "contains", "semantic"]

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


def _cmd_synthesize(args: argparse.Namespace) -> int:
    """Execute the `synthesize` subcommand."""
    # ---- Resolve seed prompts ----
    seed_prompts: list[str] = []
    if args.seed_prompts_file:
        seed_path = Path(args.seed_prompts_file)
        if not seed_path.exists():
            print(f"Error: seed prompts file not found: {seed_path}", file=sys.stderr)
            return 2
        try:
            with seed_path.open("r", encoding="utf-8") as f:
                seed_prompts = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Error reading seed prompts file {seed_path}: {e}", file=sys.stderr)
            return 2
    else:
        seed_prompts = list(args.seed_prompts or [])

    # ---- Resolve model client ----
    model_spec = args.model or os.environ.get("SYNTHESIS_MODEL")
    if not model_spec:
        print(
            "Error: synthesize requires a model client.\n"
            "Pass --model <module:callable> on the command line, "
            "or set the SYNTHESIS_MODEL environment variable, e.g.:\n"
            '  export SYNTHESIS_MODEL="my_llm:generate"\n'
            "  prompt_metrics synthesize --description \"...\" --model my_llm:generate\n\n"
            "The model callable must accept (prompt: str) -> str "
            "and return the model's raw text response.",
            file=sys.stderr,
        )
        return 2

    try:
        model_client = _resolve_callable(model_spec, kind="synthesis model")
    except Exception as e:
        print(f"Error resolving synthesis model {model_spec!r}: {e}", file=sys.stderr)
        return 2

    # ---- Validate num_cases ----
    if args.num_cases < 1:
        print("Error: --num-cases must be >= 1", file=sys.stderr)
        return 2

    # ---- Generate dataset ----
    print(
        f"Synthesizing {args.num_cases} test case(s)...\n"
        f"Description: {args.description[:120]}"
        + ("…" if len(args.description) > 120 else "")
        + f"\nSeed prompts: {len(seed_prompts)}"
        + (f" ({', '.join(repr(s[:40] + '…' if len(s) > 40 else s) for s in seed_prompts[:2])}" if seed_prompts else " (none)")
        + (f" … +{len(seed_prompts) - 2} more" if len(seed_prompts) > 2 else "")
        + f"\nModel: {model_spec}\n"
    )

    synthesizer = DatasetSynthesizer(
        model_client=model_client,  # type: ignore[arg-type]
        case_id_prefix=args.case_id_prefix,
    )

    try:
        dataset = synthesizer.generate_dataset(
            seed_prompts=seed_prompts,
            description=args.description,
            num_cases=args.num_cases,
        )
    except Exception as e:
        print(f"\nSynthesis failed: {e}", file=sys.stderr)
        return 1

    # ---- Write output ----
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error writing dataset to {output_path}: {e}", file=sys.stderr)
        return 1

    print(
        f"✅ Generated {len(dataset)} test case(s)\n"
        f"   Output: {output_path}\n"
        f"   Case IDs: {dataset[0]['id']} … {dataset[-1]['id'] if len(dataset) > 1 else ''}\n"
        f"\nRun your evaluation with:\n"
        f"  prompt_metrics run --dataset {output_path}"
    )

    if len(dataset) < args.num_cases:
        print(
            f"\n⚠️  Warning: requested {args.num_cases} cases, "
            f"but only {len(dataset)} were generated. "
            "The model may have returned fewer cases than requested, "
            "or some cases were dropped during parsing. "
            "Try running again or increasing the model's max_tokens.",
            file=sys.stderr,
        )

    return 0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()

    # Backwards compatibility: if invoked without a subcommand
    # (e.g. `prompt_metrics --dataset ...`), auto-prepend "run".
    # We need to detect this BEFORE parse_args(), otherwise argparse
    # will fail with "unrecognized arguments".
    argv_list = list(argv if argv is not None else sys.argv[1:])
    if argv_list:
        first = argv_list[0]
        # If first arg is NOT a known subcommand and looks like a flag
        # or a run-specific option, assume legacy "run" mode.
        if first not in ("run", "synthesize", "-h", "--help"):
            # Heuristic: if it starts with "-" OR contains a known run flag,
            # prepend "run"
            run_flags = ("--dataset", "--output-dir", "--evaluators", "--formats",
                         "--generator", "--judge-model", "--embedding-model", "--fail-fast")
            if first.startswith("-") or any(f in argv_list for f in run_flags):
                argv_list = ["run"] + argv_list

    args = parser.parse_args(argv_list if argv is not None else None)
    command = getattr(args, "command", None)

    if command == "synthesize":
        return _cmd_synthesize(args)
    elif command == "run":
        return _cmd_run(args)
    else:
        parser.print_help()
        return 2


if __name__ == "__main__":
    sys.exit(main())
