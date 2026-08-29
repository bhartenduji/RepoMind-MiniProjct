"""
RepoMind - Git Diff Extractor

Extracts high-quality bug-fix candidates from collected repositories.

Pipeline:

repository_manifest.json
        |
        v
Git commit history
        |
        v
Bug-fix candidate detection
        |
        v
Parent commit -> fixing commit diff
        |
        v
Quality filtering
        |
        v
data/processed/bug_fix_candidates.jsonl
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MANIFEST_PATH = Path("data/processed/repository_manifest.json")
OUTPUT_PATH = Path("data/processed/bug_fix_candidates.jsonl")

# Maximum number of commits inspected per repository.
# None means inspect the complete reachable history.
MAX_COMMITS_PER_REPO = None

# Avoid enormous patches which are usually poor training examples.
MAX_PATCH_CHARS = 80_000
MAX_CHANGED_FILES = 30
MAX_ADDED_LINES = 1_500
MAX_DELETED_LINES = 1_500

# Ignore generated / vendored / lock / documentation-heavy paths.
IGNORED_PATH_PATTERNS = [
    r"(^|/)__pycache__(/|$)",
    r"(^|/)node_modules(/|$)",
    r"(^|/)vendor(/|$)",
    r"(^|/)dist(/|$)",
    r"(^|/)build(/|$)",
    r"(^|/)target(/|$)",
    r"(^|/)coverage(/|$)",
    r"(^|/)htmlcov(/|$)",
    r"(^|/)site-packages(/|$)",
    r"(^|/)package-lock\.json$",
    r"(^|/)yarn\.lock$",
    r"(^|/)pnpm-lock\.yaml$",
    r"(^|/)poetry\.lock$",
    r"(^|/)Pipfile\.lock$",
    r"(^|/)Cargo\.lock$",
    r"(^|/)go\.sum$",
]

# Strong indicators that a commit is likely related to a bug.
STRONG_BUG_KEYWORDS = [
    "fix",
    "fixed",
    "fixes",
    "bug",
    "bugfix",
    "bug fix",
    "defect",
    "error",
    "exception",
    "crash",
    "failure",
    "fail",
    "regression",
    "incorrect",
    "wrong",
    "broken",
    "break",
    "issue",
    "patch",
    "resolve",
    "resolves",
    "resolved",
    "repair",
    "hotfix",
    "workaround",
    "prevent",
]

# Terms which often indicate non-bug changes.
NEGATIVE_KEYWORDS = [
    "merge branch",
    "merge pull request",
    "release",
    "version bump",
    "bump version",
    "changelog",
    "readme",
    "documentation",
    "docs:",
    "typo",
    "formatting",
    "lint",
    "style only",
    "refactor only",
    "rename only",
]

SOURCE_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".c",
    ".h",
    ".cpp",
    ".cc",
    ".cxx",
    ".cs",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".kts",
    ".scala",
    ".m",
    ".mm",
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def run_git(
    repo_path: Path,
    args: list[str],
    check: bool = True,
) -> str:
    """Run a git command and return stdout."""

    command = ["git", "-C", str(repo_path)] + args

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if check and result.returncode != 0:
        raise RuntimeError(
            f"Git command failed:\n"
            f"Command: {' '.join(command)}\n"
            f"Exit code: {result.returncode}\n"
            f"stderr: {result.stderr.strip()}"
        )

    return result.stdout


def load_manifest() -> list[dict[str, Any]]:
    """Load repository manifest."""

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Repository manifest must contain a JSON list.")

    return data


def normalize_text(text: str) -> str:
    """Normalize commit-message text."""

    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_ignored_path(path: str) -> bool:
    """Return True if a path should be ignored."""

    for pattern in IGNORED_PATH_PATTERNS:
        if re.search(pattern, path, flags=re.IGNORECASE):
            return True

    return False


def is_source_file(path: str) -> bool:
    """Return True for common source-code files."""

    suffix = Path(path).suffix.lower()
    return suffix in SOURCE_EXTENSIONS


# ---------------------------------------------------------------------------
# Commit analysis
# ---------------------------------------------------------------------------

def get_commits(repo_path: Path) -> list[dict[str, str]]:
    """
    Return commits from the repository.

    Format:
        hash
        parent
        author
        date
        subject
        body
    """

    format_string = (
        "%H%x1f"
        "%P%x1f"
        "%an%x1f"
        "%aI%x1f"
        "%s%x1f"
        "%b%x1e"
    )

    output = run_git(
        repo_path,
        [
            "log",
            "--all",
            "--no-show-signature",
            f"--pretty=format:{format_string}",
        ],
    )

    commits = []

    for record in output.split("\x1e"):
        record = record.strip()

        if not record:
            continue

        fields = record.split("\x1f")

        if len(fields) < 6:
            continue

        commit_hash = fields[0].strip()
        parents = fields[1].strip()
        author = fields[2].strip()
        date = fields[3].strip()
        subject = fields[4].strip()
        body = fields[5].strip()

        parent_list = parents.split() if parents else []

        commits.append(
            {
                "hash": commit_hash,
                "parent": parent_list[0] if parent_list else "",
                "parent_count": str(len(parent_list)),
                "author": author,
                "date": date,
                "subject": subject,
                "body": body,
            }
        )

    if MAX_COMMITS_PER_REPO is not None:
        commits = commits[:MAX_COMMITS_PER_REPO]

    return commits


def bug_score(subject: str, body: str) -> tuple[int, list[str]]:
    """
    Score a commit based on its message.

    This is intentionally a candidate detector, not a final label.
    """

    text = normalize_text(f"{subject} {body}")

    score = 0
    signals: list[str] = []

    for keyword in STRONG_BUG_KEYWORDS:
        if re.search(
            rf"\b{re.escape(keyword)}\b",
            text,
            flags=re.IGNORECASE,
        ):
            score += 2
            signals.append(keyword)

    for keyword in NEGATIVE_KEYWORDS:
        if keyword in text:
            score -= 3
            signals.append(f"negative:{keyword}")

    # Issue / PR references are useful evidence.
    if re.search(r"(#\d+|\bissue\s+\d+|\bpr\s+\d+)", text):
        score += 2
        signals.append("issue_or_pr_reference")

    # Explicit test-related fixes are useful.
    if "test" in text and any(
        word in text
        for word in ["fix", "bug", "regression", "failure", "error"]
    ):
        score += 2
        signals.append("bug_related_test")

    return score, sorted(set(signals))


# ---------------------------------------------------------------------------
# Diff extraction
# ---------------------------------------------------------------------------

def get_changed_files(
    repo_path: Path,
    parent: str,
    commit_hash: str,
) -> list[dict[str, Any]]:
    """Return changed files and line statistics."""

    output = run_git(
        repo_path,
        [
            "diff",
            "--numstat",
            "--find-renames",
            parent,
            commit_hash,
        ],
    )

    files = []

    for line in output.splitlines():
        parts = line.split("\t")

        if len(parts) < 3:
            continue

        added_raw, deleted_raw, path = parts[0], parts[1], parts[2]

        # Binary files show "-" instead of line counts.
        if added_raw == "-" or deleted_raw == "-":
            continue

        try:
            added = int(added_raw)
            deleted = int(deleted_raw)
        except ValueError:
            continue

        files.append(
            {
                "path": path,
                "added_lines": added,
                "deleted_lines": deleted,
                "is_source": is_source_file(path),
                "ignored": is_ignored_path(path),
            }
        )

    return files


def get_patch(
    repo_path: Path,
    parent: str,
    commit_hash: str,
) -> str:
    """Return unified diff."""

    return run_git(
        repo_path,
        [
            "diff",
            "--no-ext-diff",
            "--unified=20",
            "--find-renames",
            parent,
            commit_hash,
            "--",
        ],
    )


def filter_changed_files(
    files: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep useful source files."""

    useful = []

    for item in files:
        path = item["path"]

        if item["ignored"]:
            continue

        if not item["is_source"]:
            continue

        useful.append(item)

    return useful


def diff_quality(
    files: list[dict[str, Any]],
    patch: str,
) -> tuple[bool, list[str]]:
    """
    Apply structural quality checks.

    Returns:
        (accepted, reasons)
    """

    reasons = []

    if not files:
        reasons.append("no_source_files")
        return False, reasons

    if len(files) > MAX_CHANGED_FILES:
        reasons.append("too_many_changed_files")
        return False, reasons

    added = sum(f["added_lines"] for f in files)
    deleted = sum(f["deleted_lines"] for f in files)

    if added > MAX_ADDED_LINES:
        reasons.append("too_many_added_lines")
        return False, reasons

    if deleted > MAX_DELETED_LINES:
        reasons.append("too_many_deleted_lines")
        return False, reasons

    if len(patch) > MAX_PATCH_CHARS:
        reasons.append("patch_too_large")
        return False, reasons

    if added + deleted == 0:
        reasons.append("no_line_changes")
        return False, reasons

    return True, reasons


# ---------------------------------------------------------------------------
# Repository extraction
# ---------------------------------------------------------------------------

def extract_repository(
    repository: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Extract bug-fix candidates from one repository."""

    repo_id = repository.get("repo_id", "unknown")

    local_path = repository.get("local_path")

    if not local_path:
        raise ValueError(
            f"{repo_id}: manifest has no local_path"
        )

    repo_path = Path(local_path)

    if not repo_path.exists():
        raise FileNotFoundError(
            f"{repo_id}: repository path does not exist: {repo_path}"
        )

    if not (repo_path / ".git").exists():
        raise RuntimeError(
            f"{repo_id}: not a Git repository: {repo_path}"
        )

    commits = get_commits(repo_path)

    stats = {
        "commits_seen": len(commits),
        "bug_candidates": 0,
        "accepted": 0,
        "rejected": 0,
        "merge_commits": 0,
        "no_parent": 0,
        "no_source_diff": 0,
    }

    records = []

    for commit in commits:
        commit_hash = commit["hash"]
        parent = commit["parent"]

        if not parent:
            stats["no_parent"] += 1
            continue

        if int(commit["parent_count"]) > 1:
            stats["merge_commits"] += 1
            continue

        score, signals = bug_score(
            commit["subject"],
            commit["body"],
        )

        # Require meaningful evidence from the commit message.
        if score < 2:
            continue

        stats["bug_candidates"] += 1

        try:
            changed_files = get_changed_files(
                repo_path,
                parent,
                commit_hash,
            )

            source_files = filter_changed_files(changed_files)

            if not source_files:
                stats["no_source_diff"] += 1
                stats["rejected"] += 1
                continue

            patch = get_patch(
                repo_path,
                parent,
                commit_hash,
            )

        except Exception as exc:
            print(
                f"  Warning: could not extract "
                f"{commit_hash[:10]}: {exc}"
            )
            stats["rejected"] += 1
            continue

        accepted, quality_reasons = diff_quality(
            source_files,
            patch,
        )

        if not accepted:
            stats["rejected"] += 1
            continue

        stats["accepted"] += 1

        record = {
            "record_id": (
                f"{repo_id}:{commit_hash}"
            ),
            "repo_id": repo_id,
            "repository": repository.get("name"),
            "owner": repository.get("owner"),
            "language": repository.get("language"),

            "commit": {
                "hash": commit_hash,
                "parent": parent,
                "author": commit["author"],
                "date": commit["date"],
                "subject": commit["subject"],
                "body": commit["body"],
            },

            "bug_signal": {
                "score": score,
                "signals": signals,
            },

            "changed_files": source_files,

            "diff_stats": {
                "files_changed": len(source_files),
                "added_lines": sum(
                    f["added_lines"]
                    for f in source_files
                ),
                "deleted_lines": sum(
                    f["deleted_lines"]
                    for f in source_files
                ),
            },

            "patch": patch,

            "quality": {
                "status": "high_quality",
                "reasons": quality_reasons,
            },

            "provenance": {
                "source": "git_history",
                "repository_url": repository.get(
                    "clone_url"
                ),
                "commit_url": (
                    f"https://github.com/"
                    f"{repo_id}/commit/{commit_hash}"
                ),
            },
        }

        records.append(record)

    return records, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print("RepoMind Git Diff Extractor")
    print("============================")
    print()

    manifest = load_manifest()

    print(f"Repositories in manifest: {len(manifest)}")
    print()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_records: list[dict[str, Any]] = []

    total_stats = {
        "repositories": 0,
        "commits_seen": 0,
        "bug_candidates": 0,
        "accepted": 0,
        "rejected": 0,
        "merge_commits": 0,
        "no_parent": 0,
        "no_source_diff": 0,
    }

    for index, repository in enumerate(
        manifest,
        start=1,
    ):
        repo_id = repository.get(
            "repo_id",
            "unknown",
        )

        print("=" * 70)
        print(
            f"[{index}/{len(manifest)}] "
            f"{repo_id}"
        )
        print("=" * 70)

        try:
            records, stats = extract_repository(
                repository
            )

            all_records.extend(records)

            total_stats["repositories"] += 1

            for key in (
                "commits_seen",
                "bug_candidates",
                "accepted",
                "rejected",
                "merge_commits",
                "no_parent",
                "no_source_diff",
            ):
                total_stats[key] += stats[key]

            print(
                f"Commits inspected: "
                f"{stats['commits_seen']}"
            )
            print(
                f"Bug candidates:    "
                f"{stats['bug_candidates']}"
            )
            print(
                f"Accepted diffs:    "
                f"{stats['accepted']}"
            )
            print(
                f"Rejected diffs:    "
                f"{stats['rejected']}"
            )

        except Exception as exc:
            print(
                f"Repository failed: {exc}"
            )

        print()

    # Write JSONL.
    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        for record in all_records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)

    print(
        f"Repositories processed: "
        f"{total_stats['repositories']}"
    )

    print(
        f"Commits inspected:      "
        f"{total_stats['commits_seen']}"
    )

    print(
        f"Bug candidates:         "
        f"{total_stats['bug_candidates']}"
    )

    print(
        f"Accepted diffs:         "
        f"{total_stats['accepted']}"
    )

    print(
        f"Rejected diffs:         "
        f"{total_stats['rejected']}"
    )

    print(
        f"Merge commits skipped:  "
        f"{total_stats['merge_commits']}"
    )

    print(
        f"No-source diffs:        "
        f"{total_stats['no_source_diff']}"
    )

    print()
    print(
        f"Output records: "
        f"{len(all_records)}"
    )

    print(
        f"Output file: "
        f"{OUTPUT_PATH}"
    )

    if all_records:
        print()
        print(
            "Dataset extraction completed successfully."
        )
    else:
        print()
        print(
            "WARNING: No accepted bug-fix diffs were "
            "produced."
        )


if __name__ == "__main__":
    main()
