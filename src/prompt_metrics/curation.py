# src/prompt_metrics/curation.py
"""
Interactive human-in-the-loop curation for synthetic test cases.

Allows developers to review, edit, accept, or reject generated test cases
before they are saved to disk.
"""

from __future__ import annotations

from typing import Any, Callable


class CurationReviewer:
    """
    Interactive terminal-based reviewer for synthetic test cases.

    Walks through a list of generated cases, presenting each one with
    a clean terminal UI and prompting the user to accept, reject, edit,
    or skip.

    Example:
        >>> cases = [
        ...     {"id": "synth_001", "input_prompt": "...", "expected_text": "...", "keywords": [...]},
        ...     ...
        ... ]
        >>> reviewer = CurationReviewer(cases)
        >>> curated = reviewer.review_interactively()
        >>> # curated contains only accepted/edited cases
    """

    def __init__(
        self,
        cases: list[dict[str, Any]],
        *,
        input_fn: Callable[[str], str] | None = None,
        print_fn: Callable[[str], None] | None = None,
    ):
        """
        Args:
            cases: List of case dicts to review. Each dict should have
                `input_prompt`, `expected_text`, and `keywords` keys.
                An `id` key is optional (will be preserved if present).
            input_fn: Callable used to read user input (default: builtins.input).
                Injectable for testing.
            print_fn: Callable used to print output (default: built-in print).
                Injectable for testing.
        """
        self.cases = cases
        self._input = input_fn or input
        self._print = print_fn or print

    def review_interactively(self) -> list[dict[str, Any]]:
        """
        Review cases interactively in the terminal.

        For each case, displays:
          - input_prompt
          - expected_text
          - keywords

        Then prompts: `[a]ccept, [r]eject, [e]dit, [s]kip`

        Actions:
          a / accept  — keep the case as-is
          r / reject  — discard the case
          e / edit    — interactively edit fields, then keep
          s / skip    — skip for now (case is NOT included in output;
                        use accept if you want to keep it)

        Returns:
            List of accepted/edited cases, in original order (minus rejected/skipped).
            Case IDs are preserved from input.
        """
        if not self.cases:
            self._print("No cases to review.")
            return []

        curated: list[dict[str, Any]] = []
        total = len(self.cases)

        self._print("")
        self._print("=" * 70)
        self._print("  Synthetic Test Case Curation")
        self._print("=" * 70)
        self._print(f"  Reviewing {total} case(s).")
        self._print("  Actions: [a]ccept | [r]eject | [e]dit | [s]kip")
        self._print("=" * 70)

        for idx, case in enumerate(self.cases, 1):
            accepted_case = self._review_single_case(case, idx, total)
            if accepted_case is not None:
                curated.append(accepted_case)

        # Summary
        self._print("")
        self._print("=" * 70)
        self._print(f"  Curation complete.")
        self._print(f"  Accepted: {len(curated)} / {total}")
        self._print(f"  Rejected: {total - len(curated)} / {total}")
        self._print("=" * 70)
        self._print("")

        return curated

    # ---- single case review ----

    def _review_single_case(
        self,
        case: dict[str, Any],
        idx: int,
        total: int,
    ) -> dict[str, Any] | None:
        """
        Review a single case. Returns the case dict if accepted,
        None if rejected/skipped.
        """
        case_id = case.get("id", f"case_{idx:03d}")
        input_prompt = case.get("input_prompt", "")
        expected_text = case.get("expected_text", "")
        keywords = case.get("keywords", [])

        # Display case
        self._print("")
        self._print("─" * 70)
        self._print(f"  Case {idx}/{total}  ·  {case_id}")
        self._print("─" * 70)
        self._print("")
        self._print("  Input prompt:")
        self._print(self._indent_block(str(input_prompt), "    "))
        self._print("")
        self._print("  Expected text:")
        self._print(self._indent_block(str(expected_text or "(none)"), "    "))
        self._print("")
        self._print(f"  Keywords: {', '.join(keywords) if keywords else '(none)'}")
        self._print("")

        # Action loop
        while True:
            choice = self._input("  Action [a]ccept / [r]eject / [e]dit / [s]kip: ").strip().lower()

            if choice in ("a", "accept"):
                self._print("  → Accepted.")
                return case

            elif choice in ("r", "reject"):
                self._print("  → Rejected.")
                return None

            elif choice in ("s", "skip"):
                self._print("  → Skipped.")
                return None

            elif choice in ("e", "edit"):
                edited = self._edit_case_interactive(case.copy())
                if edited is not None:
                    self._print("  → Edited and accepted.")
                    return edited
                else:
                    # Edit was cancelled, re-prompt for action
                    self._print("")
                    self._print("  Input prompt:")
                    self._print(self._indent_block(str(input_prompt), "    "))
                    self._print("")
                    self._print("  Expected text:")
                    self._print(self._indent_block(str(expected_text or "(none)"), "    "))
                    self._print("")
                    self._print(f"  Keywords: {', '.join(keywords) if keywords else '(none)'}")
                    self._print("")
                    continue

            else:
                self._print("  Invalid choice. Enter a/r/e/s.")
                continue

    def _edit_case_interactive(self, case: dict[str, Any]) -> dict[str, Any] | None:
        """
        Interactive field editor for a single case.

        Prompts for each field with the current value shown.
        Press Enter to keep the current value.
        Enter ":q" to cancel the edit (returns None).

        Returns the edited case dict, or None if cancelled.
        """
        self._print("")
        self._print("  ┌─ Edit mode ─────────────────────────────────────────────")
        self._print("  │ Enter new value, or press Enter to keep current.")
        self._print("  │ Enter :q to cancel edit.")
        self._print("  └─────────────────────────────────────────────────────────")
        self._print("")

        # --- input_prompt ---
        current_prompt = case.get("input_prompt", "")
        self._print(f"  Current input_prompt:")
        self._print(self._indent_block(str(current_prompt), "    "))
        new_prompt = self._input("  New input_prompt (Enter to keep, :q to cancel): ")
        if new_prompt.strip() == ":q":
            self._print("  Edit cancelled.")
            return None
        if new_prompt.strip():
            case["input_prompt"] = new_prompt.strip()

        # --- expected_text ---
        self._print("")
        current_expected = case.get("expected_text", "") or ""
        self._print(f"  Current expected_text:")
        self._print(self._indent_block(str(current_expected or "(none)"), "    "))
        new_expected = self._input("  New expected_text (Enter to keep, :q to cancel): ")
        if new_expected.strip() == ":q":
            self._print("  Edit cancelled.")
            return None
        if new_expected.strip():
            case["expected_text"] = new_expected.strip()
        elif new_expected == "" and current_expected:
            # Empty input → keep current
            pass

        # --- keywords ---
        self._print("")
        current_keywords = case.get("keywords", [])
        kw_str = ", ".join(current_keywords) if current_keywords else "(none)"
        self._print(f"  Current keywords: {kw_str}")
        new_kw_str = self._input("  New keywords (comma-separated, Enter to keep, :q to cancel): ")
        if new_kw_str.strip() == ":q":
            self._print("  Edit cancelled.")
            return None
        if new_kw_str.strip():
            # Parse comma-separated keywords
            new_keywords = [k.strip() for k in new_kw_str.split(",") if k.strip()]
            case["keywords"] = new_keywords

        self._print("")
        # Show summary of edited case
        self._print("  Edited case:")
        self._print(f"    input_prompt:  {case.get('input_prompt', '')[:60]}{'...' if len(str(case.get('input_prompt', ''))) > 60 else ''}")
        exp = str(case.get("expected_text", "") or "")
        self._print(f"    expected_text: {exp[:60]}{'...' if len(exp) > 60 else ''}")
        kws = case.get("keywords", [])
        self._print(f"    keywords:      {', '.join(kws) if kws else '(none)'}")
        self._print("")

        return case

    # ---- helpers ----

    @staticmethod
    def _indent_block(text: str, prefix: str = "  ") -> str:
        """Indent every line of a multi-line string."""
        if not text:
            return prefix + "(empty)"
        lines = text.splitlines() or [""]
        return "\n".join(f"{prefix}{line}" if line else prefix for line in lines)


__all__ = ["CurationReviewer"]
