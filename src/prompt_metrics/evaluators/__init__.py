# src/prompt_metrics/evaluators/__init__.py
"""
Evaluator library for prompt_metrics.

Text-based evaluators:
  - ExactMatchEvaluator
  - KeywordEvaluator
  - RegexMatchEvaluator
  - ContainsEvaluator

Model-based evaluators (LLM-as-a-judge):
  - QAEvaluator
  - CritiqueEvaluator
"""

from .base import Evaluator, EvaluatorAdapter
from .model import (
    CritiqueEvaluator,
    ModelEvaluator,
    QAEvaluator,
    RUBRIC_TEMPLATES,
)
from .text import (
    ContainsEvaluator,
    ExactMatchEvaluator,
    KeywordEvaluator,
    RegexMatchEvaluator,
)

__all__ = [
    # Base
    "Evaluator",
    "EvaluatorAdapter",
    "ModelEvaluator",
    # Text evaluators
    "ExactMatchEvaluator",
    "KeywordEvaluator",
    "RegexMatchEvaluator",
    "ContainsEvaluator",
    # Model-based evaluators
    "QAEvaluator",
    "CritiqueEvaluator",
    "RUBRIC_TEMPLATES",
]
