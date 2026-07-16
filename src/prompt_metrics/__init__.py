# src/prompt_metrics/__init__.py
from .evaluators import (
    ContainsEvaluator,
    CritiqueEvaluator,
    EmbeddingClient,
    Evaluator,
    EvaluatorAdapter,
    ExactMatchEvaluator,
    KeywordEvaluator,
    ModelEvaluator,
    QAEvaluator,
    RegexMatchEvaluator,
    SemanticSimilarityEvaluator,
    RUBRIC_TEMPLATES,
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
    # Evaluators — base
    "Evaluator",
    "EvaluatorAdapter",
    "ModelEvaluator",
    # Evaluators — text
    "ExactMatchEvaluator",
    "KeywordEvaluator",
    "RegexMatchEvaluator",
    "ContainsEvaluator",
    # Evaluators — semantic
    "SemanticSimilarityEvaluator",
    "EmbeddingClient",
    # Evaluators — model-based
    "QAEvaluator",
    "CritiqueEvaluator",
    "RUBRIC_TEMPLATES",
]
