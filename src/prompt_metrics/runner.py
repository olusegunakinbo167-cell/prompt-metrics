# src/prompt_metrics/runner.py
"""
ExperimentRunner — batch execution engine for prompt evaluation.

Dataset format (JSON list):
[
  {
    "id": "case_001",
    "input_prompt": "Explain quantum entanglement in simple terms.",
    "expected_text": "Quantum entanglement is ... (optional ground truth)",
    "keywords": ["entangled", "particles", "quantum"]  // optional
  },
  ...
]
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from .evaluators.base import Evaluator, EvaluatorAdapter

# Re-export for backwards compatibility
__all_runner_protocols__ = ["Evaluator", "EvaluatorAdapter"]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    """A single test case from the dataset."""
    id: str
    input_prompt: str
    expected_text: str | None = None
    keywords: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TestCase":
        return cls(
            id=data["id"],
            input_prompt=data["input_prompt"],
            expected_text=data.get("expected_text"),
            keywords=data.get("keywords"),
            metadata={k: v for k, v in data.items()
                      if k not in {"id", "input_prompt", "expected_text", "keywords"}}
        )


def load_dataset(path: str) -> list[TestCase]:
    """Load a JSON dataset file into a list of TestCase objects."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError("Dataset JSON must be a list of test case objects")
    return [TestCase.from_dict(item) for item in raw]


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    """Structured result for a single test case."""
    case_id: str
    input_prompt: str
    generated_response: str
    expected_text: str | None = None
    keywords: list[str] | None = None
    evaluator_results: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SuiteResult:
    """Aggregated results for a full experiment suite run."""
    results: list[CaseResult]
    evaluator_names: list[str]
    total_cases: int
    successful_cases: int
    failed_cases: int
    total_runtime_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "total_cases": self.total_cases,
                "successful_cases": self.successful_cases,
                "failed_cases": self.failed_cases,
                "total_runtime_s": round(self.total_runtime_s, 3),
                "evaluators": self.evaluator_names,
            },
            "results": [r.to_dict() for r in self.results],
        }

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# ExperimentRunner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """
    Batch experiment execution engine.

    Runs a generator function over a dataset and evaluates each output
    with a configured list of evaluators.

    Example:
        runner = ExperimentRunner([
            RubricEvaluator.from_file("rubrics/standard_qa_rubric.json"),
            KeywordMatchEvaluator(),
        ])
        suite = runner.run_suite(
            dataset=load_dataset("data/test_cases.json"),
            generator_fn=lambda prompt: my_llm.generate(prompt)
        )
        suite.save_json("results/run_2026-07-16.json")
    """

    def __init__(self, evaluators: list[Any]):
        """
        Args:
            evaluators: List of evaluator instances. Each must either:
                - implement .evaluate(prompt, response, ...)
                - implement .score(prompt, response)  (RubricEvaluator-compatible)
                - or be wrapped in EvaluatorAdapter
        """
        self.evaluators: list[Evaluator] = []
        for ev in evaluators:
            if isinstance(ev, EvaluatorAdapter):
                self.evaluators.append(ev)  # type: ignore
            elif hasattr(ev, "evaluate"):
                # Already protocol-compliant — ensure it has a name
                if not hasattr(ev, "name"):
                    setattr(ev, "name", ev.__class__.__name__)
                self.evaluators.append(ev)  # type: ignore
            else:
                # Auto-wrap legacy evaluators (e.g. RubricEvaluator)
                wrapped = EvaluatorAdapter(ev)
                self.evaluators.append(wrapped)  # type: ignore

        if not self.evaluators:
            raise ValueError("ExperimentRunner requires at least one evaluator")

    # ---- dataset loading helpers ----

    @staticmethod
    def load_dataset(path: str) -> list[TestCase]:
        """Convenience alias: ExperimentRunner.load_dataset(path)"""
        return load_dataset(path)

    # ---- core execution ----

    def run_suite(
        self,
        dataset: list[dict[str, Any] | TestCase],
        generator_fn: Callable[[str], str],
        *,
        continue_on_error: bool = True,
        verbose: bool = False,
    ) -> SuiteResult:
        """
        Run the full experiment suite.

        Args:
            dataset: List of test case dicts or TestCase objects.
                Each dict must have: id, input_prompt
                Optional: expected_text, keywords
            generator_fn: Callable that takes (input_prompt: str) -> str
                This is where you plug in your LLM / mock generator.
            continue_on_error: If True, log errors per-case and continue.
                If False, raise on first error.
            verbose: Print progress to stdout.

        Returns:
            SuiteResult with all case results and aggregate stats.
        """
        # Normalise dataset → list[TestCase]
        cases: list[TestCase] = [
            c if isinstance(c, TestCase) else TestCase.from_dict(c)
            for c in dataset
        ]

        results: list[CaseResult] = []
        suite_start = time.perf_counter()
        failed = 0

        for i, case in enumerate(cases, 1):
            if verbose:
                print(f"[{i}/{len(cases)}] {case.id} ...", end=" ", flush=True)

            case_result = self._run_single_case(case, generator_fn)

            if case_result.error:
                failed += 1
                if verbose:
                    print(f"ERROR: {case_result.error}")
                if not continue_on_error:
                    raise RuntimeError(
                        f"Case {case.id} failed: {case_result.error}"
                    )
            elif verbose:
                print("✓")

            results.append(case_result)

        total_runtime = time.perf_counter() - suite_start

        return SuiteResult(
            results=results,
            evaluator_names=[ev.name for ev in self.evaluators],
            total_cases=len(cases),
            successful_cases=len(cases) - failed,
            failed_cases=failed,
            total_runtime_s=total_runtime,
        )

    def _run_single_case(
        self,
        case: TestCase,
        generator_fn: Callable[[str], str],
    ) -> CaseResult:
        """Execute one test case: generate → evaluate → record."""
        # --- 1. Generate ---
        gen_start = time.perf_counter()
        try:
            generated = generator_fn(case.input_prompt)
        except Exception as e:
            return CaseResult(
                case_id=case.id,
                input_prompt=case.input_prompt,
                generated_response="",
                expected_text=case.expected_text,
                keywords=case.keywords,
                error=f"generator_fn failed: {e}\n{traceback.format_exc()}",
                metadata=case.metadata,
            )
        latency_ms = (time.perf_counter() - gen_start) * 1000

        # --- 2. Evaluate ---
        evaluator_results: dict[str, Any] = {}
        eval_error: str | None = None

        for ev in self.evaluators:
            try:
                score = ev.evaluate(  # type: ignore[attr-defined]
                    prompt=case.input_prompt,
                    response=generated,
                    expected_text=case.expected_text,
                    keywords=case.keywords,
                    case_id=case.id,
                )
                evaluator_results[ev.name] = score  # type: ignore[attr-defined]
            except Exception as e:
                evaluator_results[getattr(ev, "name", str(ev))] = {
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
                eval_error = f"Evaluator {getattr(ev, 'name', ev)} failed: {e}"

        # --- 3. Record ---
        return CaseResult(
            case_id=case.id,
            input_prompt=case.input_prompt,
            generated_response=generated,
            expected_text=case.expected_text,
            keywords=case.keywords,
            evaluator_results=evaluator_results,
            latency_ms=round(latency_ms, 2),
            error=eval_error,
            metadata=case.metadata,
        )
