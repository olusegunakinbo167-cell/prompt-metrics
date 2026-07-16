# src/prompt_metrics/__init__.py
from .evaluators import (
    ContainsEvaluator,
    Evaluator,
    EvaluatorAdapter,
    ExactMatchEvaluator,
    KeywordEvaluator,
    RegexMatchEvaluator,
)
from .export import export_results, flatten_dict
from .reports import generate_markdown_report
from .runner import (
    ExperimentRunner,
    CaseResult,
    SuiteResult,
    TestCase,
    load_dataset,
)

__all__ = [
    # Core runner
    "ExperimentRunner",
    "TestCase",
    "CaseResult",
    "SuiteResult",
    "load_dataset",
    # Export / reporting
    "export_results",
    "flatten_dict",
    "generate_markdown_report",
    # Evaluators
    "Evaluator",
    "EvaluatorAdapter",
    "ExactMatchEvaluator",
    "KeywordEvaluator",
    "RegexMatchEvaluator",
    "ContainsEvaluator",
]
