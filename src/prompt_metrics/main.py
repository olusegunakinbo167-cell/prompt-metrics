"""
Deprecated: src/prompt_metrics/main.py

Use `python -m prompt_metrics` or `from prompt_metrics.cli import main` instead.

This shim exists for backwards compatibility and will be removed in a future release.
"""

import warnings
warnings.warn(
    "prompt_metrics.main is deprecated; "
    "use `python -m prompt_metrics` or `from prompt_metrics.cli import main` instead",
    DeprecationWarning,
    stacklevel=2,
)

from .cli import main

def run_evaluation() -> None:
    """Legacy entry point — forwards to cli.main()."""
    raise SystemExit(main())

if __name__ == "__main__":
    run_evaluation()
