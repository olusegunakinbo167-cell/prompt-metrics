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
from .monitoring import compare_suites, load_suite_result
from .curation import CurationReviewer
from .archiver import create_run_archive
from .reports import generate_comparison_report, generate_markdown_report
from .runner import (
    ExperimentRunner,
    CaseResult,
    SuiteResult,
    TestCase,
    load_dataset,
)
from .synthesis import DatasetSynthesizer

__all__ = [
    # Core runner
    "ExperimentRunner",
    "TestCase",
    "CaseResult",
    "SuiteResult",
    "load_dataset",
    # Export / reporting / archiving
    "export_results",
    "flatten_dict",
    "generate_markdown_report",
    "generate_comparison_report",
    "create_run_archive",
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
    # Monitoring / regression
    "compare_suites",
    "load_suite_result",
    # Synthesis
    "DatasetSynthesizer",
    # Curation
    "CurationReviewer",
]
