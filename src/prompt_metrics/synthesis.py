# src/prompt_metrics/synthesis.py
"""
Synthetic dataset generation for prompt_metrics.

Uses an LLM to expand a handful of seed examples (or just a task description)
into a full, varied test suite for prompt evaluation.

All synthesis is provider-agnostic — pass in any model_client:

    def model_client(prompt: str) -> str:
        '''Takes a prompt, returns the model's raw text response.'''
        ...

Example:
    >>> def my_openai_client(prompt: str) -> str:
    ...     return openai_client.chat.completions.create(
    ...         model="gpt-4",
    ...         messages=[{"role": "user", "content": prompt}]
    ...     ).choices[0].message.content
    ...
    >>> synth = DatasetSynthesizer(model_client=my_openai_client)
    >>> dataset = synth.generate_dataset(
    ...     seed_prompts=[
    ...         "Summarize this article in 3 bullet points.",
    ...         "Give me the key takeaways from this text.",
    ...     ],
    ...     description="A text summarization tool that produces concise bullet-point summaries.",
    ...     num_cases=20,
    ... )
    >>> dataset[0]
    {'id': 'synth_001', 'input_prompt': '...', 'expected_text': '...', 'keywords': [...]}
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

# Type alias for a provider-agnostic model client
# Takes a prompt string, returns the model's raw text response
ModelClient = Callable[[str], str]


class DatasetSynthesizer:
    """
    LLM-powered synthetic test case generator.

    Takes a task description + optional seed prompts, and expands them into
    a diverse evaluation dataset with varied inputs, expected outputs, and
    keywords.

    Args:
        model_client: Callable that takes a prompt string and returns the
            model's raw text response. Provider-agnostic — pass in OpenAI,
            Anthropic, local model, mock, etc.
        case_id_prefix: Prefix for generated case IDs (default: "synth").
            Case IDs will be "{prefix}_{000}", e.g. "synth_001".
    """

    def __init__(
        self,
        model_client: ModelClient,
        *,
        case_id_prefix: str = "synth",
    ):
        if not callable(model_client):
            raise TypeError("model_client must be callable")
        self.model_client = model_client
        self.case_id_prefix = case_id_prefix

    # ---- public API ----

    def generate_dataset(
        self,
        seed_prompts: list[str],
        description: str,
        num_cases: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Generate a synthetic evaluation dataset.

        Args:
            seed_prompts: Example user prompts / inputs. Can be empty if
                `description` is detailed enough. The synthesizer will use
                these as style/structure guides and generate variations.
            description: A description of what the application / prompt being
                tested does. E.g. "A customer support chatbot that answers
                questions about order status" or "A SQL query generator that
                converts natural language to PostgreSQL".
                Be specific — this drives the diversity and relevance of
                generated cases.
            num_cases: Number of test cases to generate (default: 10).
                If the LLM returns fewer, they are returned as-is.
                If the LLM returns more, only the first `num_cases` are kept.

        Returns:
            A list of dataset dictionaries, each with:
                id: str              – unique case ID
                input_prompt: str    – the test input / user query
                expected_text: str | None – reference answer / expected output
                keywords: list[str]  – relevant keywords for KeywordEvaluator

            Ready to pass directly to `ExperimentRunner.run_suite()` or
            save as JSON for `prompt_metrics --dataset`.

        Raises:
            RuntimeError: If the model_client raises an exception.
            ValueError: If num_cases < 1.
        """
        if num_cases < 1:
            raise ValueError("num_cases must be >= 1")

        prompt = self._build_synthesis_prompt(seed_prompts, description, num_cases)

        try:
            raw_response = self.model_client(prompt)
        except Exception as e:
            raise RuntimeError(f"model_client failed during synthesis: {e}") from e

        cases = self._parse_synthesis_output(raw_response, num_cases)

        # Ensure we have at least something
        if not cases:
            raise RuntimeError(
                "Synthesis produced no valid test cases. "
                f"Raw model response (first 500 chars): {raw_response[:500]!r}"
            )

        # Trim to requested count, assign sequential case IDs
        cases = cases[:num_cases]
        for i, case in enumerate(cases, 1):
            case["id"] = f"{self.case_id_prefix}_{i:03d}"

        return cases

    # ---- prompt construction ----

    def _build_synthesis_prompt(
        self,
        seed_prompts: list[str],
        description: str,
        num_cases: int,
    ) -> str:
        """
        Build the LLM prompt that asks for a synthetic dataset.
        """
        seed_section = ""
        if seed_prompts:
            seed_list = "\n".join(f"  - {s}" for s in seed_prompts)
            seed_section = f"""
Seed examples (use these as style/structure guides, then generate diverse variations):

{seed_list}
"""

        return f"""You are a test case generator for LLM prompt evaluation.

Task description:
{description}
{seed_section}
Generate {num_cases} diverse test cases for evaluating this task.

For EACH test case, provide ALL of the following:
  1. input_prompt    — the user input / query to test
  2. expected_text   — a high-quality reference answer / expected output
  3. keywords        — 3-7 relevant keywords that should appear in a good response

Diversity requirements — vary your test cases across ALL of these dimensions:
  • Input length: mix short queries, medium prompts, and long/complex inputs
  • Tone / style: formal, casual, technical, conversational, etc.
  • Difficulty: simple cases AND challenging edge cases
  • Failure modes: include adversarial inputs, ambiguous queries, empty or
    minimal constraints, out-of-domain requests, conflicting instructions
  • Topic coverage: cover the full breadth of the task description,
    not just one narrow sub-area

Return your response as a JSON array (list) of objects, with this exact schema:
[
  {{
    "input_prompt": "the test input / user query",
    "expected_text": "the reference answer / expected output",
    "keywords": ["keyword1", "keyword2", "keyword3"]
  }},
  ...
]

CRITICAL:
- Output ONLY the JSON array. No markdown fences, no prose, no commentary.
- Every object MUST have all three fields: input_prompt, expected_text, keywords.
- keywords MUST be a list of strings (3-7 items).
- Generate exactly {num_cases} test cases.
- Make input_prompt values genuinely diverse — do NOT repeat similar prompts.
- expected_text should be a realistic, high-quality response, not a stub.
"""

    # ---- output parsing ----

    def _parse_synthesis_output(
        self,
        raw: str,
        num_cases: int,
    ) -> list[dict[str, Any]]:
        """
        Parse the model's synthesis response into a list of dataset dicts.

        Tries, in order:
          1. Direct JSON parse
          2. Strip markdown fences, then JSON parse
          3. Extract first JSON array from text, then parse
          4. Fall back to empty list → caller raises

        Returns a list of dicts with keys: input_prompt, expected_text, keywords.
        Missing fields are filled with None / [] so cases aren't dropped silently.
        Case IDs are assigned by the caller.
        """
        # --- Try 1: direct JSON ---
        cases = self._try_parse_json_array(raw)
        if cases:
            return self._normalize_cases(cases)

        # --- Try 2: strip markdown fences ---
        stripped = raw.strip()
        # Remove ```json ... ``` or ``` ... ```
        if stripped.startswith("```"):
            # Find closing fence
            # Strip opening fence line
            lines = stripped.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            # Strip closing fence
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            stripped = "\n".join(lines)

        cases = self._try_parse_json_array(stripped)
        if cases:
            return self._normalize_cases(cases)

        # --- Try 3: extract first JSON array from text ---
        # Look for [...] — greedy to get the outermost array
        # This handles: "Here are the cases:\n[...]\nLet me know..."
        array_match = re.search(r"\[\s*\{.*?\}\s*\]", raw, re.DOTALL)
        if array_match:
            cases = self._try_parse_json_array(array_match.group(0))
            if cases:
                return self._normalize_cases(cases)

        # --- Try 4: extract array with nested braces (more robust) ---
        # Find first '[' then track bracket depth to find matching ']'
        start = raw.find("[")
        if start != -1:
            depth = 0
            end = -1
            in_string = False
            escape = False
            for i, ch in enumerate(raw[start:], start):
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"' and not escape:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end != -1:
                cases = self._try_parse_json_array(raw[start:end])
                if cases:
                    return self._normalize_cases(cases)

        # All parsing failed
        return []

    def _try_parse_json_array(self, text: str) -> list[dict[str, Any]] | None:
        """Try to parse text as a JSON array of objects. Returns None on failure."""
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                # Filter to dict items only
                return [item for item in parsed if isinstance(item, dict)]
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        return None

    def _normalize_cases(self, raw_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Normalize parsed case dicts to the runner's expected schema.

        Accepts flexible field names and fills missing fields:
          input_prompt  ← "input_prompt" | "prompt" | "input" | "query" | "question"
          expected_text ← "expected_text" | "expected" | "expected_output" |
                          "output" | "answer" | "reference"
          keywords      ← "keywords" | "keyword" (str or list)
        """
        normalized: list[dict[str, Any]] = []

        for raw in raw_cases:
            if not isinstance(raw, dict):
                continue

            # --- input_prompt ---
            input_prompt = (
                raw.get("input_prompt")
                or raw.get("prompt")
                or raw.get("input")
                or raw.get("query")
                or raw.get("question")
            )
            if not input_prompt or not isinstance(input_prompt, str):
                # Skip cases with no valid input — can't run them
                continue
            input_prompt = input_prompt.strip()

            # --- expected_text ---
            expected_text = (
                raw.get("expected_text")
                or raw.get("expected")
                or raw.get("expected_output")
                or raw.get("output")
                or raw.get("answer")
                or raw.get("reference")
            )
            if expected_text is not None and not isinstance(expected_text, str):
                expected_text = str(expected_text)
            if isinstance(expected_text, str):
                expected_text = expected_text.strip() or None

            # --- keywords ---
            keywords = raw.get("keywords") or raw.get("keyword") or []
            if isinstance(keywords, str):
                # Comma-separated string → list
                keywords = [k.strip() for k in keywords.split(",") if k.strip()]
            if not isinstance(keywords, list):
                keywords = []
            # Ensure all items are strings
            keywords = [str(k).strip() for k in keywords if str(k).strip()]

            normalized.append(
                {
                    "input_prompt": input_prompt,
                    "expected_text": expected_text,
                    "keywords": keywords,
                }
            )

        return normalized


__all__ = ["DatasetSynthesizer", "ModelClient"]
