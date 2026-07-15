# prompt-metrics

Scripts and experiments for scoring and testing LLM prompt responses.

`prompt-metrics` is a lightweight evaluation framework for quantifying LLM output quality across automated benchmarks and human-graded rubrics. Built for reproducible prompt engineering, regression testing, and multi-model comparison as part of Project Obsidian.

---

## Features

- **Automated scoring:** BLEU, ROUGE, METEOR, exact-match, and embedding-based similarity
- **Human rubric harness:** structured JSON rubrics with inter-rater aggregation
- **Multi-domain benchmarks:** MMLU-style multitask evaluation, TruthfulQA-style hallucination tracking
- **Regression testing:** snapshot prompt/response pairs with delta reporting
- **Model-agnostic:** OpenAI, Anthropic, local Ollama, and Hugging Face compatible
- **Export:** CSV / JSONL / Parquet score tables for statistical analysis

---

## Quick installation

```bash
# Clone
git clone https://github.com/olusegunakinbo167-cell/prompt-metrics.git
cd prompt-metrics

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Install dev tools
pip install -r requirements-dev.txt
```

### Minimum requirements
- Python ≥ 3.10
- `numpy`, `scipy`, `pandas`
- `nltk` (for BLEU/ROUGE tokenization — run `python -m nltk.downloader punkt` on first use)

### API keys (optional)
Export any provider keys you plan to use:
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Quick start

```bash
# Score a single response file against a reference
python score_response.py \
  --response outputs/gpt-4o_run1.json \
  --reference datasets/mmlu_subset.jsonl \
  --metrics bleu,rougeL,exact_match \
  --output scores/run1.csv

# Run a full benchmark suite
python score_response.py \
  --config configs/mmlu_truthfulqa.yaml \
  --model gpt-4o \
  --output scores/mmlu_truthfulqa_$(date +%Y%m%d).csv
```

Example Python API:
```python
from prompt_metrics import Scorer

scorer = Scorer(metrics=["bleu", "rougeL", "bertscore"])
result = scorer.score(
    candidate="Paris is the capital of France.",
    reference="The capital of France is Paris."
)
print(result)  # {'bleu': 0.82, 'rougeL': 0.89, 'bertscore': 0.94}
```

---

## Automated metrics vs. human rubrics

`prompt-metrics` treats automated scores and human judgement as complementary layers, not substitutes.

### Automated metrics

| Metric | What it measures | Best for | Limitation |
|---|---|---|---|
| **BLEU** | n-gram precision vs. human references. IBM, 2001. | MT, summarization fidelity | Poor on paraphrase / creativity |
| **ROUGE-L** | Longest common subsequence recall | Summarization coverage | Ignores semantic equivalence |
| **METEOR** | Precision/recall with stemming + synonymy | MT with paraphrase tolerance | Slower, language-dependent |
| **Exact match / F1** | Token-level correctness | QA, structured extraction | Brittle to formatting |
| **BERTScore / embedding sim** | Semantic similarity in embedding space | Open-ended generation | Model-dependent calibration |
| **MMLU-style accuracy** | Multi-domain MCQA accuracy across 57 subjects | General knowledge / reasoning | MCQ format only; contamination risk |
| **TruthfulQA-style** | Truthfulness vs. imitative falsehoods | Hallucination tracking | Requires curated adversarial prompts |

> Background: *BLEU — "Quality is considered to be the correspondence between a machine's output and that of a human … remains one of the most popular automated and inexpensive metrics."* — Wikipedia, https://en.wikipedia.org/wiki/BLEU
>
> *MMLU — "Measuring Massive Multitask Language Understanding … is a popular benchmark for evaluating the capabilities of large language models."* — Wikipedia, https://en.wikipedia.org/wiki/MMLU

### Human rubrics

Automated metrics correlate with human preference, but do not replace it. Use the rubric harness for:

- **Faithfulness** — is the response grounded in the provided context?
- **Completeness** — are all required elements present?
- **Clarity / style** — is the output fit for the target audience?
- **Safety** — policy compliance, PII leakage, toxic content

Rubric format (`rubrics/human_eval_v1.json`):
```json
{
  "dimensions": [
    {"id": "accuracy", "scale": [1, 5], "description": "…"},
    {"id": "fluency", "scale": [1, 5], "description": "…"},
    {"id": "hallucination_check", "scale": [1, 5], "description": "…"}
  ],
  "aggregation": "mean",
  "inter_rater": "krippendorff_alpha"
}
```

Score with human annotations:
```bash
python score_response.py \
  --responses outputs/batch_01.jsonl \
  --rubric rubrics/human_eval_v1.json \
  --human-labels labels/batch_01_raters.csv \
  --output scores/batch_01_human.csv
```

### When to use which

| Scenario | Recommended |
|---|---|
| CI / nightly regression | Automated metrics (BLEU, ROUGE, exact-match) — fast, deterministic |
| Prompt iteration / A/B | Automated + embedding similarity — rapid feedback loop |
| Release gate / production eval | Automated + human rubric with n≥2 raters, report IRR |
| Hallucination / safety audit | TruthfulQA-style adversarial set + human faithfulness rubric |

---

## Repository layout

```
prompt-metrics/
├── score_response.py          # Main scoring CLI
├── track_metrics.py           # Nightly asset health check
├── prompt_metrics/            # Scoring library
│   ├── metrics/               # BLEU, ROUGE, BERTScore, etc.
│   ├── rubrics/               # Human rubric schemas
│   └── benchmarks/            # MMLU, TruthfulQA loaders
├── rubrics/
│   └── human_eval_v1.json     # Accuracy / fluency / hallucination rubric
├── configs/                   # Benchmark suite YAMLs
├── datasets/                  # Reference data (gitignored large files)
├── outputs/                   # Model responses
└── scores/                    # Score exports
```

---

## Contributing

PRs and issues welcome. Please include:
1. A reproducible test case
2. Expected vs. actual scores
3. Environment details (`python --version`, dependency versions)

---

## License

MIT — see `LICENSE`.

---

## Citation

If you use prompt-metrics in research, please cite:

```
@software{prompt_metrics_2026,
  author = {Olusegun Akinbo},
  title = {prompt-metrics: LLM prompt response scoring framework},
  url = {https://github.com/olusegunakinbo167-cell/prompt-metrics},
  year = {2026}
}
```
