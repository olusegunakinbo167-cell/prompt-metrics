# Allow: python -m prompt_metrics --dataset ...
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
