# src/prompt_metrics/cli.py
"""
Command-line interface for prompt_metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

from .archiver import create_run_archive
from .curation import CurationReviewer
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
from .monitoring import compare_suites, load_suite_result
from .reports import generate_comparison_report, generate_markdown_report
from .runner import ExperimentRunner, load_dataset
from .synthesis import DatasetSynthesizer


def _resolve_callable(spec: str, kind: str = "callable") -> Callable[[str], str]:
    if ":" not in spec:
        raise ValueError(f'{kind} {spec!r} must be "module:callable"')
    module_name, attr = spec.split(":", 1)
    import importlib
    mod = importlib.import_module(module_name)
    obj: Any = mod
    for part in attr.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise TypeError(f"{kind.capitalize()} {spec!r} is not callable")
    return obj  # type: ignore[return-value]


EVALUATOR_REGISTRY: dict[str, Callable[[], Any]] = {}


def _make_judge_factory(evaluator_cls: type, *, rubric: str | None = None) -> Callable[[], Any]:
    def factory() -> Any:
        spec = getattr(_make_judge_factory, "_judge_model_spec", None) or os.environ.get("JUDGE_MODEL")
        if not spec:
            raise RuntimeError(f"{evaluator_cls.__name__} requires a judge model. Pass --judge-model <module:callable> or set JUDGE_MODEL env var.")
        model_client = _resolve_callable(spec, kind="judge model")
        kwargs: dict[str, Any] = {"model_client": model_client}
        if rubric is not None:
            kwargs["rubric"] = rubric
        return evaluator_cls(**kwargs)
    return factory


def _make_semantic_factory() -> Callable[[], Any]:
    def factory() -> Any:
        spec = getattr(_make_semantic_factory, "_embedding_model_spec", None) or os.environ.get("EMBEDDING_MODEL")
        if spec:
            embedding_client = _resolve_callable(spec, kind="embedding model")
            return SemanticSimilarityEvaluator(embedding_client=embedding_client)  # type: ignore[arg-type]
        return SemanticSimilarityEvaluator()
    return factory


def _register_builtin_evaluators() -> None:
    EVALUATOR_REGISTRY.clear()
    EVALUATOR_REGISTRY.update({
        "exact_match": ExactMatchEvaluator, "exact": ExactMatchEvaluator,
        "keyword": KeywordEvaluator, "keywords": KeywordEvaluator,
        "contains": ContainsEvaluator,
        "regex": RegexMatchEvaluator, "regex_match": RegexMatchEvaluator,
        "semantic": _make_semantic_factory(), "semantic_similarity": _make_semantic_factory(),
        "qa_judge": _make_judge_factory(QAEvaluator), "qa": _make_judge_factory(QAEvaluator),
        "critique_judge": _make_judge_factory(CritiqueEvaluator), "critique": _make_judge_factory(CritiqueEvaluator),
    })
    for rubric_name in RUBRIC_TEMPLATES.keys():
        EVALUATOR_REGISTRY[f"critique_{rubric_name}"] = _make_judge_factory(CritiqueEvaluator, rubric=rubric_name)


_register_builtin_evaluators()


def _try_import_external_evaluators() -> None:
    try:
        from rubric_qa.evaluator import RubricEvaluator  # type: ignore
    except Exception:
        return
    def rubric_factory() -> Any:
        rubric_path = os.environ.get("RUBRIC_PATH", "rubric.json")
        if not Path(rubric_path).exists():
            raise FileNotFoundError(f"Rubric file not found at {rubric_path!r}")
        return RubricEvaluator.from_file(rubric_path)
    EVALUATOR_REGISTRY["rubric"] = rubric_factory
    EVALUATOR_REGISTRY["rubric_qa"] = rubric_factory


def _default_generator(prompt: str) -> str:
    return f"[MOCK] {prompt[:200]}"


def _resolve_generator(spec: str | None) -> Callable[[str], str]:
    if spec is None or spec == "mock":
        return _default_generator
    return _resolve_callable(spec, kind="generator")  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    _try_import_external_evaluators()
    available = ", ".join(sorted(EVALUATOR_REGISTRY.keys()))
    p = argparse.ArgumentParser(
        prog="prompt_metrics",
        description="LLM prompt evaluation toolkit — run experiments, generate test datasets, curate and archive results.",
    )
    subparsers = p.add_subparsers(dest="command", metavar="<command>", help="run | synthesize")

    run_p = subparsers.add_parser("run", help="Run a prompt evaluation experiment.",
        description="Evaluate prompts against a test dataset with configurable evaluators.")
    run_p.add_argument("--dataset", required=True,
        help="Path to test dataset JSON. Each entry needs: id, input_prompt. Optional: expected_text, keywords.")
    run_p.add_argument("--output-dir", default="./results",
        help="Output directory for results (default: ./results).")
    run_p.add_argument("--evaluators", default="",
        help=f"Comma-separated evaluators to run. Available: {available}. "
             "Default: exact_match,keyword,contains,semantic. "
             "Model-based evaluators (qa_judge, critique_*) require --judge-model.")
    run_p.add_argument("--formats", default="csv,json,md",
        help="Output formats to generate, comma-separated (default: csv,json,md).")
    run_p.add_argument("--generator", default=None,
        help='Generator function as "module:callable" (e.g. "my_llm:generate"). '
             "Must accept (prompt: str) -> str. Defaults to a mock generator.")
    run_p.add_argument("--judge-model", default=None,
        help='Judge model for LLM-as-a-judge evaluators (qa_judge, critique_*), as "module:callable". '
             "Callable must accept (prompt: str) -> str. Also reads JUDGE_MODEL env var.")
    run_p.add_argument("--embedding-model", default=None,
        help='Embedding client for semantic similarity, as "module:callable" (text: str) -> list[float]. '
             "If omitted, uses built-in TF-IDF fallback (zero dependencies) or auto-detects sentence-transformers. "
             "Also reads EMBEDDING_MODEL env var.")
    run_p.add_argument("--fail-fast", action="store_true",
        help="Stop on the first case/generator error instead of continuing.")

    comp = run_p.add_argument_group("regression monitoring")
    comp.add_argument("--compare-with", metavar="BASELINE_JSON", default=None,
        help="Compare this run against a baseline results JSON. "
             "Prints a regression summary, writes comparison.md, "
             "and exits with code 3 if regressions are detected.")
    comp.add_argument("--score-drop-threshold", type=float, default=0.15,
        help="Score decrease that counts as a regression (default: 0.15).")
    comp.add_argument("--latency-regression-factor", type=float, default=1.2,
        help="Latency increase factor that counts as a regression (default: 1.2 = 20%% slower).")

    run_p.add_argument("--archive", nargs="?", const="archives", metavar="ARCHIVE_DIR",
        help="Bundle run artifacts into a timestamped zip after successful completion. "
             "Optionally specify a custom archive directory (default: ./archives). "
             "Archive name: run_YYYYMMDD_HHMMSS_<score>.zip")

    synth_p = subparsers.add_parser("synthesize",
        help="Generate a synthetic evaluation dataset using an LLM.",
        description="Expand seed examples + a task description into a diverse test suite with an LLM.")
    synth_p.add_argument("--description", required=True,
        help="What the application / prompt being tested does. "
             "Be specific — this drives test case diversity. "
             'E.g. "A SQL query generator that converts natural language to PostgreSQL".')
    synth_seed = synth_p.add_mutually_exclusive_group()
    synth_seed.add_argument("--seed-prompts", nargs="*", default=[], metavar="PROMPT",
        help="Seed example prompts / inputs (space-separated). Guides style and structure. Optional if --description is detailed.")
    synth_seed.add_argument("--seed-prompts-file", metavar="PATH",
        help="Path to a text file with seed prompts, one per line. Empty lines are ignored.")
    synth_p.add_argument("--num-cases", type=int, default=10,
        help="Number of test cases to generate (default: 10).")
    synth_p.add_argument("--model", default=None,
        help='Model client for synthesis, as "module:callable" (e.g. "my_llm:generate"). '
             "Callable must accept (prompt: str) -> str. "
             "Also reads SYNTHESIS_MODEL env var.")
    synth_p.add_argument("--output", "-o", default="synthetic_dataset.json",
        help="Output path for the generated dataset JSON (default: synthetic_dataset.json).")
    synth_p.add_argument("--case-id-prefix", default="synth",
        help='Prefix for generated case IDs (default: "synth" → synth_001, synth_002, …).')
    synth_p.add_argument("--interactive", "-i", action="store_true",
        help="Review, edit, accept, or reject generated test cases interactively before saving. "
             "Opens a terminal UI for human-in-the-loop curation.")
    return p


def _cmd_run(args: argparse.Namespace) -> int:
    if args.judge_model:
        setattr(_make_judge_factory, "_judge_model_spec", args.judge_model)
    if args.embedding_model:
        setattr(_make_semantic_factory, "_embedding_model_spec", args.embedding_model)
    _register_builtin_evaluators()
    _try_import_external_evaluators()
    if args.evaluators.strip():
        requested = [e.strip() for e in args.evaluators.split(",") if e.strip()]
    else:
        requested = ["exact_match", "keyword", "contains", "semantic"]
    unknown = [e for e in requested if e not in EVALUATOR_REGISTRY]
    if unknown:
        print(f"Error: unknown evaluator(s): {', '.join(unknown)}\n"
              f"Available: {', '.join(sorted(EVALUATOR_REGISTRY.keys()))}", file=sys.stderr)
        return 2
    evaluators = []
    for name in requested:
        try:
            evaluators.append(EVALUATOR_REGISTRY[name]())
        except Exception as e:
            print(f"Error instantiating evaluator {name!r}: {e}", file=sys.stderr)
            return 2
    try:
        generator_fn = _resolve_generator(args.generator)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: dataset file not found: {dataset_path}", file=sys.stderr)
        return 2
    try:
        dataset = load_dataset(str(dataset_path))
    except Exception as e:
        print(f"Error loading dataset {dataset_path}: {e}", file=sys.stderr)
        return 2
    print(f"Loaded {len(dataset)} test case(s) from {dataset_path}\n"
          f"Evaluators: {', '.join(requested)}\n"
          f"Generator: {generator_fn.__module__}.{getattr(generator_fn, '__name__', '<callable>')}")
    runner = ExperimentRunner(evaluators)
    try:
        suite = runner.run_suite(dataset, generator_fn, continue_on_error=not args.fail_fast, verbose=True)
    except Exception as e:
        print(f"\nExperiment failed: {e}", file=sys.stderr)
        return 1
    print(f"\nSuite complete — {suite.successful_cases}/{suite.total_cases} passed, "
          f"{suite.failed_cases} failed, {suite.total_runtime_s:.2f}s total")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = {f.strip().lower() for f in args.formats.split(",") if f.strip()}
    valid_formats = {"csv", "json", "md"}
    invalid = formats - valid_formats
    if invalid:
        print(f"Warning: ignoring unknown format(s): {', '.join(sorted(invalid))}", file=sys.stderr)
        formats &= valid_formats
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
    if written:
        print("\nOutputs written:")
        for path in written:
            print(f"  {path}")
    comparison_exit_code = 0
    if getattr(args, "compare_with", None):
        baseline_path = Path(args.compare_with)  # type: ignore[attr-defined]
        if not baseline_path.exists():
            print(f"\nError: baseline file not found: {baseline_path}", file=sys.stderr)
            return 2
        print(f"\n{'─' * 60}\nRegression comparison: {baseline_path.name} → current run\n{'─' * 60}")
        try:
            baseline_suite = load_suite_result(str(baseline_path))
            comparison = compare_suites(
                suite, baseline_suite,
                latency_regression_threshold=args.latency_regression_factor,  # type: ignore[attr-defined]
                score_drop_threshold=args.score_drop_threshold,  # type: ignore[attr-defined]
            )
        except Exception as e:
            print(f"Error comparing suites: {e}", file=sys.stderr)
            return 2
        summary = comparison["summary"]
        metrics_drift = comparison["metrics_drift"]
        score_drops = comparison["score_drops"]
        latency_regressions = comparison["latency_regressions"]
        print(f"Cases: baseline={summary['baseline_cases']}, "
              f"current={summary['current_cases']}, common={summary['common_cases']}")
        print("\nMetrics drift:")
        if metrics_drift:
            for eval_name in sorted(metrics_drift.keys()):
                d = metrics_drift[eval_name]
                mean_delta = d["mean_delta"]
                icon = "🟢" if mean_delta >= 0.01 else "⚪" if mean_delta >= -0.01 else "🟡" if mean_delta >= -0.05 else "🔴"
                print(f"  {icon} {eval_name}: mean {mean_delta:+.4f}, "
                      f"median {d['median_delta']:+.4f}, "
                      f"↑{d['improved']} ↓{d['regressed']} ＝{d['unchanged']}")
        else:
            print("  (no comparable scores)")
        if score_drops:
            print(f"\n⚠️  Score drops (>{args.score_drop_threshold}): {len(score_drops)} case(s)")
            for drop in score_drops[:10]:
                print(f"  🔴 {drop['case_id']} / {drop['evaluator']}: "
                      f"{drop['baseline_score']:.3f} → {drop['current_score']:.3f} ({drop['delta']:+.3f})")
            if len(score_drops) > 10:
                print(f"  … +{len(score_drops) - 10} more")
            comparison_exit_code = 3
        else:
            print(f"\n✅ No score drops > {args.score_drop_threshold}")
        if latency_regressions:
            pct = (args.latency_regression_factor - 1.0) * 100  # type: ignore[attr-defined]
            print(f"\n⚠️  Latency regressions (>{pct:.0f}% slower): {len(latency_regressions)} case(s)")
            for reg in latency_regressions[:10]:
                print(f"  ⏱️  {reg['case_id']}: {reg['baseline_ms']:.1f}ms → {reg['current_ms']:.1f}ms "
                      f"({reg['factor']:.2f}×)")
            if len(latency_regressions) > 10:
                print(f"  … +{len(latency_regressions) - 10} more")
            comparison_exit_code = 3
        else:
            pct = (args.latency_regression_factor - 1.0) * 100  # type: ignore[attr-defined]
            print(f"\n✅ No latency regressions > {pct:.0f}%")
        comparison_md_path = output_dir / "comparison.md"
        try:
            generate_comparison_report(comparison, str(comparison_md_path))
            print(f"\n📊 Comparison report: {comparison_md_path}")
        except Exception as e:
            print(f"\nWarning: failed to write comparison report: {e}", file=sys.stderr)
    if args.archive:
        archive_dir = args.archive if isinstance(args.archive, str) else "archives"
        print(f"\n📦 Archiving run artifacts to {archive_dir}/ ...")
        try:
            archive_path = create_run_archive(str(output_dir), archive_dir=archive_dir)
            print(f"✅ Archive created: {archive_path}")
        except Exception as e:
            print(f"⚠️  Archiving failed: {e}", file=sys.stderr)
    return comparison_exit_code


def _cmd_synthesize(args: argparse.Namespace) -> int:
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
    model_spec = args.model or os.environ.get("SYNTHESIS_MODEL")
    if not model_spec:
        print(
            "Error: synthesize requires a model client.\n"
            "Pass --model <module:callable> or set SYNTHESIS_MODEL env var, e.g.:\n"
            '  export SYNTHESIS_MODEL="my_llm:generate"\n'
            '  prompt_metrics synthesize --description "..." --model my_llm:generate',
            file=sys.stderr,
        )
        return 2
    try:
        model_client = _resolve_callable(model_spec, kind="synthesis model")
    except Exception as e:
        print(f"Error resolving synthesis model {model_spec!r}: {e}", file=sys.stderr)
        return 2
    if args.num_cases < 1:
        print("Error: --num-cases must be >= 1", file=sys.stderr)
        return 2
    print(f"Synthesizing {args.num_cases} test case(s)...\n"
          f"Description: {args.description[:120]}"
          + ("…" if len(args.description) > 120 else "") +
          f"\nSeed prompts: {len(seed_prompts)}\n"
          f"Model: {model_spec}\n")
    synthesizer = DatasetSynthesizer(model_client=model_client, case_id_prefix=args.case_id_prefix)  # type: ignore[arg-type]
    try:
        dataset = synthesizer.generate_dataset(
            seed_prompts=seed_prompts,
            description=args.description,
            num_cases=args.num_cases,
        )
    except Exception as e:
        print(f"\nSynthesis failed: {e}", file=sys.stderr)
        return 1
    if args.interactive:
        print(f"\n📝 Opening interactive curation for {len(dataset)} case(s)...\n")
        try:
            reviewer = CurationReviewer(dataset)
            dataset = reviewer.review_interactively()
        except (KeyboardInterrupt, EOFError):
            print("\n\n⚠️  Curation interrupted.", file=sys.stderr)
            return 130
        if not dataset:
            print("\n⚠️  All cases were rejected. Nothing to save.", file=sys.stderr)
            return 0
        print(f"\n✅ Curation complete: {len(dataset)} case(s) accepted.\n")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error writing dataset to {output_path}: {e}", file=sys.stderr)
        return 1
    print(
        f"✅ {'Curated' if args.interactive else 'Generated'} {len(dataset)} test case(s)\n"
        f"   Output: {output_path}\n"
        f"   Case IDs: {dataset[0]['id'] if dataset else '(none)'}"
        + (f" … {dataset[-1]['id']}" if len(dataset) > 1 else "") +
        f"\n\nRun your evaluation with:\n  prompt_metrics run --dataset {output_path}"
    )
    if not args.interactive and len(dataset) < args.num_cases:
        print(f"\n⚠️  Warning: requested {args.num_cases} cases, but only {len(dataset)} were generated.",
              file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    argv_list = list(argv if argv is not None else sys.argv[1:])
    if argv_list:
        first = argv_list[0]
        if first not in ("run", "synthesize", "-h", "--help"):
            run_flags = ("--dataset", "--output-dir", "--evaluators", "--formats",
                         "--generator", "--judge-model", "--embedding-model",
                         "--fail-fast", "--archive", "--compare-with")
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
