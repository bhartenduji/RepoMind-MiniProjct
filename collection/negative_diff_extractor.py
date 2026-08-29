"""
RepoMind - Non-Bug-Fix Diff Extractor

Collects high-quality commits that are intentionally NOT identified
as bug-fix candidates.

These records are used as negative examples (label = 0) for the
binary bug-fix classifier.

Input:
    data/processed/repository_manifest.json
    data/processed/bug_fix_candidates.jsonl

Output:
    data/processed/non_bug_fix_candidates.jsonl

Important:
    Negative examples must NOT be selected using a rule that makes
    bug_signal.score == 0 for every negative record.

    Otherwise the classifier can learn the dataset construction
    heuristic instead of learning actual bug-fix characteristics.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any


# ============================================================
# PATHS
# ============================================================

MANIFEST_PATH = Path(
    "data/processed/repository_manifest.json"
)

BUG_CANDIDATE_PATH = Path(
    "data/processed/bug_fix_candidates.jsonl"
)

OUTPUT_PATH = Path(
    "data/processed/non_bug_fix_candidates.jsonl"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Number of negative examples we want overall.
TARGET_NEGATIVES = 12000

# Maximum negatives collected from one repository.
MAX_NEGATIVES_PER_REPO = 1200

MAX_PATCH_CHARS = 80000
MAX_CHANGED_FILES = 30
MAX_ADDED_LINES = 1500
MAX_DELETED_LINES = 1500


# ============================================================
# PATH FILTERS
# ============================================================

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
    r"(^|/)package-lock.json$",
    r"(^|/)yarn.lock$",
    r"(^|/)pnpm-lock.yaml$",
    r"(^|/)poetry.lock$",
    r"(^|/)Pipfile.lock$",
    r"(^|/)Cargo.lock$",
    r"(^|/)go.sum$",
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


# ============================================================
# BUG SIGNALS
# ============================================================

# These are useful for calculating the bug_signal feature.
#
# IMPORTANT:
# We do NOT use these as an automatic "all negatives must have
# score 0" filter.
#
# The classifier should see both positive and negative records
# containing some of these words.

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


# ============================================================
# EXPLICIT BUG-FIX INTENT
# ============================================================

# These patterns represent a much stronger indication that the
# commit's PRIMARY PURPOSE is actually fixing a bug.
#
# Generic words such as "issue", "error", "test", or "fix" by
# themselves are NOT enough to reject a negative example.
#
# Examples that should generally remain eligible:
#
#   "fix typo in documentation"
#   "fix test naming"
#   "fix formatting"
#   "update test after API change"
#   "improve error message"
#   "fix example"
#
# Examples that should generally be rejected:
#
#   "fix crash when parsing malformed input"
#   "fix regression in query compiler"
#   "resolve bug causing incorrect results"
#   "repair broken transaction handling"

EXPLICIT_BUG_FIX_PATTERNS = [
    r"\bfix(?:ed|es)?\s+(?:the\s+)?(?:bug|issue|defect|regression|crash)\b",
    r"\bfix(?:ed|es)?\s+.*\b(?:crash|regression|incorrect|broken|wrong)\b",
    r"\bresolve[ds]?\s+(?:the\s+)?(?:bug|issue|defect|regression)\b",
    r"\brepair(?:ed|s)?\s+(?:the\s+)?(?:bug|issue|defect|regression)\b",
    r"\b(?:bug|issue|defect)\s+(?:fix|fixed|fixes)\b",
    r"\bfix(?:ed|es)?\s+.*\bcaus(?:e|es|ed|ing)\b",
    r"\bfix(?:ed|es)?\s+.*\bincorrect\b",
    r"\bfix(?:ed|es)?\s+.*\bwrong\b",
    r"\bfix(?:ed|es)?\s+.*\bbroken\b",
    r"\bfix(?:ed|es)?\s+.*\bfail(?:ure|ed|ing)?\b",
    r"\bfix(?:ed|es)?\s+.*\bexception\b",
    r"\bfix(?:ed|es)?\s+.*\bcrash(?:es|ed|ing)?\b",
    r"\bfix(?:ed|es)?\s+.*\bregression\b",
    r"\bhotfix\b",
]


# ============================================================
# NON-BUG / ROUTINE CHANGE SIGNALS
# ============================================================

ROUTINE_KEYWORDS = [
    "add",
    "implement",
    "feature",
    "new",
    "update",
    "upgrade",
    "refactor",
    "cleanup",
    "rename",
    "remove",
    "delete",
    "deprecate",
    "support",
    "improve",
    "optimize",
    "performance",
    "typing",
    "type hints",
    "test",
    "tests",
    "testing",
    "documentation",
    "docs",
    "readme",
    "example",
    "examples",
]


# ============================================================
# GIT UTILITY
# ============================================================

def run_git(
    repo_path: Path,
    args: list[str],
    check: bool = True,
) -> str:
    command = [
        "git",
        "-C",
        str(repo_path),
    ] + args

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


# ============================================================
# MANIFEST
# ============================================================

def load_manifest() -> list[dict[str, Any]]:
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
        raise ValueError(
            "Repository manifest must contain a JSON list."
        )

    return data


# ============================================================
# EXISTING POSITIVE RECORDS
# ============================================================

def load_positive_commit_ids() -> set[str]:
    positive_ids: set[str] = set()

    if not BUG_CANDIDATE_PATH.exists():
        raise FileNotFoundError(
            f"Bug candidate dataset not found: "
            f"{BUG_CANDIDATE_PATH}"
        )

    with BUG_CANDIDATE_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            repo_id = record.get(
                "repo_id",
                "",
            )

            commit = record.get(
                "commit",
                {},
            )

            commit_hash = commit.get(
                "hash",
                "",
            )

            if repo_id and commit_hash:
                positive_ids.add(
                    f"{repo_id}:{commit_hash}"
                )

    return positive_ids


# ============================================================
# TEXT
# ============================================================

def normalize_text(text: str) -> str:
    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# BUG SIGNAL SCORING
# ============================================================

def calculate_bug_signal(
    subject: str,
    body: str,
) -> tuple[int, list[str]]:
    """
    Calculate the original-style bug signal.

    This is a FEATURE, not the negative-label rule.

    Both positive and negative records may have a non-zero score.
    """

    text = normalize_text(
        f"{subject} {body}"
    )

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

    # Issue / PR references are useful evidence.
    if re.search(
        r"(#\d+|\bissue\s+\d+|\bpr\s+\d+)",
        text,
        flags=re.IGNORECASE,
    ):
        score += 2
        signals.append(
            "issue_or_pr_reference"
        )

    # Explicit test-related bug/failure evidence.
    if (
        re.search(
            r"\btest\b|\btests\b|\btesting\b",
            text,
            flags=re.IGNORECASE,
        )
        and any(
            word in text
            for word in [
                "fix",
                "bug",
                "regression",
                "failure",
                "error",
            ]
        )
    ):
        score += 2
        signals.append(
            "bug_related_test"
        )

    return score, sorted(
        set(signals)
    )


# ============================================================
# NEGATIVE COMMIT DETECTION
# ============================================================

def has_explicit_bug_fix_intent(
    subject: str,
    body: str,
) -> bool:
    """
    Return True only when the commit message strongly indicates
    that the PRIMARY purpose is a production bug fix.

    This intentionally does NOT reject every commit containing
    words such as "fix", "error", "issue", or "test".
    """

    text = normalize_text(
        f"{subject} {body}"
    )

    for pattern in EXPLICIT_BUG_FIX_PATTERNS:

        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):
            return True

    return False


def has_bug_signal(
    subject: str,
    body: str,
) -> bool:
    """
    Backwards-compatible wrapper.

    This now means explicit bug-fix intent rather than merely
    the presence of a generic bug-related word.
    """

    return has_explicit_bug_fix_intent(
        subject,
        body,
    )


# ============================================================
# ROUTINE SIGNAL
# ============================================================

def routine_signal_score(
    subject: str,
    body: str,
) -> int:
    text = normalize_text(
        f"{subject} {body}"
    )

    score = 0

    for keyword in ROUTINE_KEYWORDS:

        if re.search(
            rf"\b{re.escape(keyword)}\b",
            text,
            flags=re.IGNORECASE,
        ):
            score += 1

    return score


# ============================================================
# DIFF EXTRACTION
# ============================================================
# ============================================================
# COMMIT HISTORY
# ============================================================

def get_commits(
    repo_path: Path,
) -> list[dict[str, str]]:
    """
    Return commits from the repository history.

    Each commit contains:
    - hash
    - first parent
    - parent count
    - author
    - date
    - subject
    - body

    Merge commits are retained here and filtered later by
    extract_repository().
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

    commits: list[dict[str, str]] = []

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

        parent_list = (
            parents.split()
            if parents
            else []
        )

        commits.append(
            {
                "hash": commit_hash,
                "parent": (
                    parent_list[0]
                    if parent_list
                    else ""
                ),
                "parent_count": str(
                    len(parent_list)
                ),
                "author": author,
                "date": date,
                "subject": subject,
                "body": body,
            }
        )

    return commits
def get_changed_files(
    repo_path: Path,
    parent: str,
    commit_hash: str,
) -> list[dict[str, Any]]:

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

        added_raw = parts[0]
        deleted_raw = parts[1]
        path = parts[2]

        if (
            added_raw == "-"
            or deleted_raw == "-"
        ):
            continue

        try:
            added = int(
                added_raw
            )

            deleted = int(
                deleted_raw
            )

        except ValueError:
            continue

        files.append(
            {
                "path": path,
                "added_lines": added,
                "deleted_lines": deleted,
                "is_source": is_source_file(
                    path
                ),
                "ignored": is_ignored_path(
                    path
                ),
            }
        )

    return files


def filter_changed_files(
    files: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    useful = []

    for item in files:

        if item["ignored"]:
            continue

        if not item["is_source"]:
            continue

        useful.append(item)

    return useful


def get_patch(
    repo_path: Path,
    parent: str,
    commit_hash: str,
) -> str:

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


# ============================================================
# PATH HELPERS
# ============================================================

def is_ignored_path(path: str) -> bool:

    for pattern in IGNORED_PATH_PATTERNS:

        if re.search(
            pattern,
            path,
            flags=re.IGNORECASE,
        ):
            return True

    return False


def is_source_file(path: str) -> bool:

    suffix = Path(
        path
    ).suffix.lower()

    return suffix in SOURCE_EXTENSIONS


# ============================================================
# DIFF QUALITY
# ============================================================

def diff_quality(
    files: list[dict[str, Any]],
    patch: str,
) -> tuple[bool, list[str]]:

    reasons = []

    if not files:

        reasons.append(
            "no_source_files"
        )

        return False, reasons

    if len(files) > MAX_CHANGED_FILES:

        reasons.append(
            "too_many_changed_files"
        )

        return False, reasons

    added = sum(
        f["added_lines"]
        for f in files
    )

    deleted = sum(
        f["deleted_lines"]
        for f in files
    )

    if added > MAX_ADDED_LINES:

        reasons.append(
            "too_many_added_lines"
        )

        return False, reasons

    if deleted > MAX_DELETED_LINES:

        reasons.append(
            "too_many_deleted_lines"
        )

        return False, reasons

    if len(patch) > MAX_PATCH_CHARS:

        reasons.append(
            "patch_too_large"
        )

        return False, reasons

    if added + deleted == 0:

        reasons.append(
            "no_line_changes"
        )

        return False, reasons

    return True, reasons


# ============================================================
# REPOSITORY EXTRACTION
# ============================================================

def extract_repository(
    repository: dict[str, Any],
    positive_ids: set[str],
) -> tuple[
    list[dict[str, Any]],
    dict[str, int],
]:

    repo_id = repository.get(
        "repo_id",
        "unknown",
    )

    local_path = repository.get(
        "local_path"
    )

    if not local_path:
        raise ValueError(
            f"{repo_id}: manifest has no local_path"
        )

    repo_path = Path(
        local_path
    )

    if not repo_path.exists():
        raise FileNotFoundError(
            f"{repo_id}: repository path does not exist: "
            f"{repo_path}"
        )

    if not (
        repo_path / ".git"
    ).exists():
        raise RuntimeError(
            f"{repo_id}: not a Git repository: "
            f"{repo_path}"
        )

    commits = get_commits(
        repo_path
    )

    records = []

    stats = {
        "commits_seen": len(commits),
        "eligible": 0,
        "accepted": 0,
        "rejected": 0,
        "merge_commits": 0,
        "already_positive": 0,
        "bug_signal": 0,
        "no_parent": 0,
        "no_source_diff": 0,
        "routine_signal_zero": 0,
    }

    for commit in commits:

        if (
            len(records)
            >= MAX_NEGATIVES_PER_REPO
        ):
            break

        commit_hash = commit["hash"]
        parent = commit["parent"]

        if not parent:

            stats["no_parent"] += 1

            continue

        if int(
            commit["parent_count"]
        ) > 1:

            stats["merge_commits"] += 1

            continue

        record_id = (
            f"{repo_id}:{commit_hash}"
        )

        # Never use an existing positive commit.
        if record_id in positive_ids:

            stats["already_positive"] += 1

            continue

        # ----------------------------------------------------
        # Calculate bug signal as a FEATURE.
        #
        # DO NOT require this to be zero.
        # ----------------------------------------------------

        bug_score, bug_signals = (
            calculate_bug_signal(
                commit["subject"],
                commit["body"],
            )
        )

        # ----------------------------------------------------
        # Reject only clearly explicit bug-fix commits.
        #
        # This is deliberately narrower than the old
        # has_bug_signal() rule.
        # ----------------------------------------------------

        if has_explicit_bug_fix_intent(
            commit["subject"],
            commit["body"],
        ):

            stats["bug_signal"] += 1

            continue

        # ----------------------------------------------------
        # Require an ordinary development signal.
        # ----------------------------------------------------

        routine_score = (
            routine_signal_score(
                commit["subject"],
                commit["body"],
            )
        )

        if routine_score == 0:

            stats["routine_signal_zero"] += 1

            continue

        stats["eligible"] += 1

        try:

            changed_files = (
                get_changed_files(
                    repo_path,
                    parent,
                    commit_hash,
                )
            )

            source_files = (
                filter_changed_files(
                    changed_files
                )
            )

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

        accepted, quality_reasons = (
            diff_quality(
                source_files,
                patch,
            )
        )

        if not accepted:

            stats["rejected"] += 1

            continue

        stats["accepted"] += 1

        record = {

            "record_id": record_id,

            "repo_id": repo_id,

            "repository": repository.get(
                "name"
            ),

            "owner": repository.get(
                "owner"
            ),

            "language": repository.get(
                "language"
            ),

            "commit": {

                "hash": commit_hash,

                "parent": parent,

                "author": commit["author"],

                "date": commit["date"],

                "subject": commit["subject"],

                "body": commit["body"],
            },

            # IMPORTANT:
            # Preserve the ACTUAL signal instead of forcing
            # every negative record to score zero.
            "bug_signal": {

                "score": bug_score,

                "signals": bug_signals,
            },

            "changed_files": source_files,

            "diff_stats": {

                "files_changed": len(
                    source_files
                ),

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

            "label": 0,

            "quality": {

                "status": "high_quality",

                "reasons": quality_reasons,
            },

            "dataset_quality": {

                "version": "negative_v2",

                "score": 4,

                "reasons": [
                    "no_explicit_bug_fix_intent",
                    "routine_development_signal",
                    "source_code_change",
                    "structurally_valid_diff",
                ],
            },

            "provenance": {

                "source": "git_history",

                "repository_url": repository.get(
                    "clone_url"
                ),

                "commit_url": (
                    f"https://github.com/"
                    f"{repo_id}/commit/"
                    f"{commit_hash}"
                ),
            },
        }

        records.append(record)

    return records, stats


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 70)
    print(
        "RepoMind Non-Bug-Fix Diff Extractor"
    )
    print("=" * 70)
    print()

    manifest = load_manifest()

    positive_ids = (
        load_positive_commit_ids()
    )

    print(
        f"Repositories: "
        f"{len(manifest)}"
    )

    print(
        f"Existing positive commits: "
        f"{len(positive_ids)}"
    )

    print(
        f"Target negatives: "
        f"{TARGET_NEGATIVES}"
    )

    print()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_records = []

    total_stats = {
        "repositories": 0,
        "commits_seen": 0,
        "eligible": 0,
        "accepted": 0,
        "rejected": 0,
        "merge_commits": 0,
        "already_positive": 0,
        "bug_signal": 0,
        "no_parent": 0,
        "no_source_diff": 0,
        "routine_signal_zero": 0,
    }

    for index, repository in enumerate(
        manifest,
        start=1,
    ):

        if (
            len(all_records)
            >= TARGET_NEGATIVES
        ):
            break

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

            records, stats = (
                extract_repository(
                    repository,
                    positive_ids,
                )
            )

            remaining = (
                TARGET_NEGATIVES
                - len(all_records)
            )

            records = records[
                :remaining
            ]

            all_records.extend(
                records
            )

            total_stats[
                "repositories"
            ] += 1

            for key in total_stats:

                if key == "repositories":
                    continue

                total_stats[key] += (
                    stats.get(key, 0)
                )

            print(
                f"Commits inspected: "
                f"{stats['commits_seen']}"
            )

            print(
                f"Eligible negatives: "
                f"{stats['eligible']}"
            )

            print(
                f"Accepted negatives: "
                f"{len(records)}"
            )

            print(
                f"Explicit bug-fix commits skipped: "
                f"{stats['bug_signal']}"
            )

            print(
                f"No routine signal: "
                f"{stats['routine_signal_zero']}"
            )

        except Exception as exc:

            print(
                f"Repository failed: {exc}"
            )

        print()

    # --------------------------------------------------------
    # Write output
    # --------------------------------------------------------

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
    print(
        "NEGATIVE EXTRACTION SUMMARY"
    )
    print("=" * 70)

    print(
        f"Repositories processed: "
        f"{total_stats['repositories']}"
    )

    print(
        f"Commits inspected:       "
        f"{total_stats['commits_seen']}"
    )

    print(
        f"Eligible negatives:      "
        f"{total_stats['eligible']}"
    )

    print(
        f"Accepted negatives:      "
        f"{len(all_records)}"
    )

    print(
        f"Explicit bug-fix skipped: "
        f"{total_stats['bug_signal']}"
    )

    print(
        f"Routine signal missing:   "
        f"{total_stats['routine_signal_zero']}"
    )

    print(
        f"Merge commits skipped:    "
        f"{total_stats['merge_commits']}"
    )

    print(
        f"Already-positive skipped: "
        f"{total_stats['already_positive']}"
    )

    print()

    print(
        f"Output: "
        f"{OUTPUT_PATH}"
    )

    print()

    if len(all_records) >= TARGET_NEGATIVES:

        print(
            "SUCCESS: Target negative dataset "
            "size reached."
        )

    else:

        print(
            "WARNING: Target was not reached."
        )

        print(
            f"Collected only "
            f"{len(all_records)} "
            f"negative examples."
        )


if __name__ == "__main__":
    main()