from __future__ import annotations
import argparse
import importlib
import json
import os
import sys
from typing import Any, Callable, List

from .runner import ExperimentRunner
from .export import export_results
from .reports import generate_markdown_report


# ---------------------------------------------------------------------------
# Built-in evaluators
# ---------------------------------------------------------------------------

class ExactMatchEvaluator:
    name = "exact_match"

    def evaluate(self, case: dict, output_text: str) -> dict:
        expected = case.get("expected_text") or ""
        score = 1.0 if output_text.strip().lower() == expected.strip().lower() else 0.0
        return {"score": score}


class KeywordEvaluator:
    name = "keyword"

    def evaluate(self, case: dict, output_text: str) -> dict:
        keywords = case.get("keywords") or []
        if not keywords:
            return {"score": 1.0, "matched": 0, "total": 0}
        text_lower = output_text.lower()
        matched = sum(1 for kw in keywords if str(kw).lower() in text_lower)
        total = len(keywords)
        score = matched / total if total else 1.0
        return {"score": score, "matched": matched, "total": total}


EVALUATOR_REGISTRY: dict[str, Callable[[], Any]] = {
    "exact_match": ExactMatchEvaluator,
    "exact": ExactMatchEvaluator,
    "keyword": KeywordEvaluator,
    "keywords": KeywordEvaluator,
}


def resolve_evaluators(names: List[str] | None) -> List[Any]:
    """Instantiate evaluators by name. If names is None/empty, use all built-ins."""
    if not names:
        # default: all unique built-in evaluators
        seen = set()
        evaluators = []
        for cls in [ExactMatchEvaluator, KeywordEvaluator]:
            if cls not in seen:
                evaluators.append(cls())
                seen.add(cls)
        return evaluators

    evaluators = []
    for name in names:
        name = name.strip()
        if not name:
            continue
        factory = EVALUATOR_REGISTRY.get(name)
        if factory is None:
            available = ", ".join(sorted(set(EVALUATOR_REGISTRY.keys())))
            raise ValueError(
                f"Unknown evaluator '{name}'. Available: {available}"
            )
        evaluators.append(factory())
    return evaluators


# ---------------------------------------------------------------------------
# Generator resolution
# ---------------------------------------------------------------------------

def mock_echo_generator(prompt: str) -> str:
    """Fallback generator – echoes the input prompt."""
    return f"Echo: {prompt}"


def resolve_generator(spec: str | None) -> Callable[[str], str]:
    if not spec:
        return mock_echo_generator
    # spec format: module:callable  or module.callable
    if ":" in spec:
        module_name, attr_name = spec.split(":", 1)
    elif "." in spec:
        module_name, attr_name = spec.rsplit(".", 1)
    else:
        raise ValueError(
            f"Invalid --generator spec {spec!r}. "
            "Use module:callable (e.g. my_module:my_llm_call)"
        )
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        raise ImportError(f"Could not import generator module '{module_name}': {e}") from e
    try:
        fn = getattr(mod, attr_name)
    except AttributeError:
        raise AttributeError(
            f"Module '{module_name}' has no attribute '{attr_name}'"
        )
    if not callable(fn):
        raise TypeError(f"Generator '{spec}' is not callable")
    return fn


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="prompt_metrics",
        description="Run LLM prompt evaluation pipelines"
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="Path to JSON dataset file"
    )
    p.add_argument(
        "--output-dir",
        default="./results",
        help="Directory where results are saved (default: ./results)"
    )
    p.add_argument(
        "--evaluators",
        default="",
        help="Comma-separated list of evaluator names to run "
             "(e.g. exact_match,keyword). Default: all built-in evaluators"
    )
    p.add_argument(
        "--formats",
        default="json,csv,md",
        help="Comma-separated output formats: json,csv,md (default: all three)"
    )
    p.add_argument(
        "--generator",
        default=None,
        help="Custom generator to import dynamically, in module:callable format "
             "(e.g. my_module:my_llm_call). If omitted, uses mock echo generator"
    )
    return p


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # 1. Load dataset
    if not os.path.isfile(args.dataset):
        print(f"Error: dataset file not found: {args.dataset}", file=sys.stderr)
        return 2
    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if not isinstance(dataset, list):
        print("Error: dataset JSON must be a list of test cases", file=sys.stderr)
        return 2

    # 2. Resolve generator
    try:
        generator_fn = resolve_generator(args.generator)
    except Exception as e:
        print(f"Error resolving --generator: {e}", file=sys.stderr)
        return 2

    # 3. Resolve evaluators
    evaluator_names = [
        s.strip() for s in args.evaluators.split(",") if s.strip()
    ] if args.evaluators else None
    try:
        evaluators = resolve_evaluators(evaluator_names)
    except Exception as e:
        print(f"Error resolving --evaluators: {e}", file=sys.stderr)
        return 2

    if not evaluators:
        print("Error: no evaluators selected", file=sys.stderr)
        return 2

    # 4. Run suite
    print(f"Running {len(dataset)} cases with {len(evaluators)} evaluator(s): "
          f"{', '.join(getattr(ev, 'name', ev.__class__.__name__) for ev in evaluators)}",
          file=sys.stderr)
    runner = ExperimentRunner(evaluators)
    suite_result = runner.run_suite(dataset, generator_fn)

    # 5. Export
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    formats = {s.strip().lower() for s in args.formats.split(",") if s.strip()}
    valid_formats = {"json", "csv", "md", "markdown"}
    unknown = formats - valid_formats
    if unknown:
        print(f"Warning: ignoring unknown format(s): {', '.join(sorted(unknown))}",
              file=sys.stderr)
        formats &= valid_formats

    written = []
    if "json" in formats:
        path = os.path.join(output_dir, "results.json")
        export_results(suite_result, path, format="json")
        written.append(path)
    if "csv" in formats:
        path = os.path.join(output_dir, "results.csv")
        export_results(suite_result, path, format="csv")
        written.append(path)
    if "md" in formats or "markdown" in formats:
        path = os.path.join(output_dir, "report.md")
        generate_markdown_report(suite_result, path)
        written.append(path)

    # Summary
    meta = suite_result.to_dict().get("metadata", {})
    print(
        f"\nDone. Cases: {len(dataset)} | "
        f"Success: {meta.get('success_count', '?')} | "
        f"Fail: {meta.get('fail_count', '?')} | "
        f"Runtime: {meta.get('total_runtime_ms', '?')} ms",
        file=sys.stderr
    )
    for p in written:
        print(f"  → {p}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
