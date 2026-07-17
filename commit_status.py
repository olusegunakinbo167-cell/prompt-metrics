#!/usr/bin/env python3
"""
Post a commit status check to GitHub using `gh api`.

Useful for blocking merges via branch protection rules that require
a status check to pass before merging.

Exit codes:
  0 — status posted successfully, or gh not available (warning only)
  1 — gh is available but the API call failed
"""
import argparse
import json
import shutil
import subprocess
import sys


def get_commit_sha() -> str | None:
    """Resolve the current commit SHA from git."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_repo() -> str | None:
    """Resolve owner/repo from gh cli."""
    try:
        result = subprocess.run(
            ["gh", "repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def check_gh_available() -> tuple[bool, str | None]:
    """Check if gh is installed and authenticated."""
    if not shutil.which("gh"):
        return False, "gh CLI not found in PATH"
    try:
        result = subprocess.run(
            ["gh", "auth", "status"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return False, "gh CLI is not authenticated"
    except FileNotFoundError:
        return False, "gh CLI not found in PATH"
    return True, None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Post a commit status check to GitHub."
    )
    parser.add_argument(
        "--state",
        required=True,
        choices=["pending", "success", "failure", "error"],
        help="Status state",
    )
    parser.add_argument(
        "--description",
        required=True,
        help="Status description",
    )
    parser.add_argument(
        "--context",
        default="prompt-metrics/eval",
        help="Status context / check name (default: prompt-metrics/eval)",
    )
    parser.add_argument(
        "--sha",
        default=None,
        help="Commit SHA to attach status to (default: HEAD)",
    )
    parser.add_argument(
        "--target-url",
        default=None,
        help="Optional URL to link from the status check",
    )
    args = parser.parse_args()

    # Bail out cleanly if gh isn't usable — don't break local runs
    gh_ok, gh_reason = check_gh_available()
    if not gh_ok:
        print(f"warning: skipping commit status — {gh_reason}", file=sys.stderr)
        return 0

    sha = args.sha or get_commit_sha()
    if not sha:
        print("warning: skipping commit status — could not resolve commit SHA", file=sys.stderr)
        return 0

    repo = get_repo()
    if not repo:
        print("warning: skipping commit status — could not resolve owner/repo", file=sys.stderr)
        return 0

    payload = {
        "state": args.state,
        "description": args.description,
        "context": args.context,
    }
    if args.target_url:
        payload["target_url"] = args.target_url

    # Post status via gh api
    try:
        result = subprocess.run(
            [
                "gh", "api",
                f"/repos/{repo}/statuses/{sha}",
                "--method", "POST",
                "--input", "-",
            ],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(
                f"error: gh api failed: {result.stderr.strip()}",
                file=sys.stderr,
            )
            return 1
        print(f"✓ posted status '{args.state}' for {sha[:8]} — {args.context}: {args.description}")
        return 0
    except Exception as e:
        print(f"error: failed to post commit status: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
