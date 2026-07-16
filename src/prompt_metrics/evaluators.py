"""
Deprecated: src/prompt_metrics/evaluators.py

This module has been superseded by prompt_metrics.evaluators (a package).
Import from prompt_metrics.evaluators instead:

    from prompt_metrics.evaluators import (
        ExactMatchEvaluator,
        KeywordEvaluator,
        RegexMatchEvaluator,
        ContainsEvaluator,
    )

This shim exists for backwards compatibility and will be removed in a future release.
"""

import warnings
warnings.warn(
    "prompt_metrics.evaluators (singular .py file) is deprecated; "
    "use `from prompt_metrics.evaluators import ...` instead",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from the new package
from .evaluators import (
    ContainsEvaluator,
    ExactMatchEvaluator,
    KeywordEvaluator,
    RegexMatchEvaluator,
    Evaluator,
    EvaluatorAdapter,
)

__all__ = [
    "ExactMatchEvaluator",
    "KeywordEvaluator",
    "RegexMatchEvaluator",
    "ContainsEvaluator",
    "Evaluator",
    "EvaluatorAdapter",
]
