from __future__ import annotations
import time
import json
from typing import Callable, List, Dict, Any, Protocol, Optional


class Evaluator(Protocol):
    name: str

    def evaluate(self, case: dict, output_text: str) -> dict: ...


class RubricEvaluatorAdapter:
    """Adapter: wraps legacy RubricEvaluator.score(output_text) -> float|dict"""

    def __init__(self, evaluator: Any):
        self._inner = evaluator
        self.name = getattr(evaluator, "name", evaluator.__class__.__name__)

    def evaluate(self, case: dict, output_text: str) -> dict:
        if hasattr(self._inner, "evaluate"):
            return self._inner.evaluate(case, output_text)
        score_fn = getattr(self._inner, "score", None)
        if not callable(score_fn):
            raise AttributeError(
                f"Evaluator {self.name} has neither evaluate() nor score()"
            )
        result = score_fn(output_text)
        if isinstance(result, dict):
            return result
        return {"score": float(result)}


def _wrap_evaluator(ev: Any) -> Evaluator:
    if hasattr(ev, "evaluate") and callable(ev.evaluate):
        return ev  # type: ignore
    return RubricEvaluatorAdapter(ev)  # type: ignore


class CaseResult:
    def __init__(
        self,
        case_id: Optional[str],
        input_prompt: str,
        generated_response: str,
        expected_text: Optional[str],
        keywords: Optional[List[str]],
        evaluator_results: Dict[str, Any],
        latencies_ms: Dict[str, int],
        errors: List[Dict[str, str]],
    ):
        self.case_id = case_id
        self.input_prompt = input_prompt
        self.generated_response = generated_response
        self.expected_text = expected_text
        self.keywords = keywords
        self.evaluator_results = evaluator_results
        self.latencies_ms = latencies_ms
        self.errors = errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "input_prompt": self.input_prompt,
            "generated_response": self.generated_response,
            "expected_text": self.expected_text,
            "keywords": self.keywords,
            "evaluator_results": self.evaluator_results,
            "latencies_ms": self.latencies_ms,
            "errors": self.errors,
        }


class SuiteResult:
    def __init__(
        self,
        results: List[CaseResult],
        evaluator_names: List[str],
        total_runtime_ms: int,
        success_count: int,
        fail_count: int,
    ):
        self.results = results
        self.evaluator_names = evaluator_names
        self.total_runtime_ms = total_runtime_ms
        self.success_count = success_count
        self.fail_count = fail_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": {
                "evaluator_names": self.evaluator_names,
                "total_runtime_ms": self.total_runtime_ms,
                "success_count": self.success_count,
                "fail_count": self.fail_count,
                "total_cases": len(self.results),
            },
            "results": [r.to_dict() for r in self.results],
        }

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


class ExperimentRunner:
    def __init__(self, evaluators: List[Any]):
        self.evaluators: List[Evaluator] = [_wrap_evaluator(ev) for ev in evaluators]

    def run_suite(
        self,
        dataset: List[Dict[str, Any]],
        generator_fn: Callable[[str], str],
        continue_on_error: bool = True,
        verbose: bool = False,
    ) -> SuiteResult:
        suite_start = time.perf_counter()
        results: List[CaseResult] = []
        success_count = 0
        fail_count = 0

        for case in dataset:
            case_errors: List[Dict[str, str]] = []
            case_latencies: Dict[str, int] = {}

            # Step 1: Run Generator
            gen_start = time.perf_counter()
            try:
                response_text = generator_fn(case["input_prompt"])
                success_count += 1
            except Exception as e:
                response_text = ""
                case_errors.append({"stage": "generator", "error": str(e)})
                fail_count += 1
                if not continue_on_error:
                    raise e
            case_latencies["generator"] = int(
                (time.perf_counter() - gen_start) * 1000
            )

            # Step 2: Run Evaluators
            evaluator_results = {}
            for ev in self.evaluators:
                ev_start = time.perf_counter()
                try:
                    evaluator_results[ev.name] = ev.evaluate(case, response_text)
                except Exception as ev_err:
                    evaluator_results[ev.name] = {"error": str(ev_err)}
                    case_errors.append(
                        {"stage": f"evaluator_{ev.name}", "error": str(ev_err)}
                    )
                case_latencies[ev.name] = int(
                    (time.perf_counter() - ev_start) * 1000
                )

            results.append(
                CaseResult(
                    case_id=case.get("id"),
                    input_prompt=case["input_prompt"],
                    generated_response=response_text,
                    expected_text=case.get("expected_text"),
                    keywords=case.get("keywords"),
                    evaluator_results=evaluator_results,
                    latencies_ms=case_latencies,
                    errors=case_errors,
                )
            )

        total_runtime_ms = int((time.perf_counter() - suite_start) * 1000)
        eval_names = [ev.name for ev in self.evaluators]

        return SuiteResult(
            results=results,
            evaluator_names=eval_names,
            total_runtime_ms=total_runtime_ms,
            success_count=success_count,
            fail_count=fail_count,
        )
