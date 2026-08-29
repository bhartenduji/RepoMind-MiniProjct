from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = Path(
    "data/processed/bug_fix_candidates.jsonl"
)

FILTERED_PATH = Path(
    "data/processed/bug_fix_quality_filtered.jsonl"
)

SPLIT_DIR = Path(
    "data/processed/dataset_splits"
)

TRAIN_PATH = SPLIT_DIR / "train.jsonl"
VALIDATION_PATH = SPLIT_DIR / "validation.jsonl"
TEST_PATH = SPLIT_DIR / "test.jsonl"


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

# Keep repositories reasonably balanced.
#
# This is a CAP, not a requirement.
# Repositories with fewer records keep all of them.
MAX_RECORDS_PER_REPOSITORY = 3000

# Minimum number of records required for a repository
# to participate in the final dataset.
MIN_RECORDS_PER_REPOSITORY = 100

# Split ratios.
TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10

# Patch size limits.
MAX_CHANGED_FILES = 20
MAX_CHANGED_LINES = 1000

# Extremely tiny patches are often weak training examples.
MIN_CHANGED_LINES = 1

# Require source-code changes.
SOURCE_EXTENSIONS = {
    ".py",
}

# Weak / noisy commit-message patterns.
REJECT_PATTERNS = [
    r"^\s*merge\b",
    r"^\s*merged\b",
    r"\bmerge pull request\b",
    r"\bmerge branch\b",
    r"\bsvnmerge\b",
    r"\bcherry[- ]pick\b",
    r"\bcherry picked\b",
    r"\brelease\b",
    r"\bversion bump\b",
    r"\bbump version\b",
    r"^\s*bump\b",
    r"\bupdate changelog\b",
    r"^\s*changelog\b",
    r"^\s*docs?\b",
    r"\bdocumentation\b",
    r"^\s*readme\b",
    r"\bformatting\b",
    r"^\s*lint\b",
    r"\btypo\b",
]

# Strong positive evidence.
STRONG_POSITIVE_PATTERNS = [
    r"\bfix\b",
    r"\bfixed\b",
    r"\bfixes\b",
    r"\bbug\b",
    r"\bbugfix\b",
    r"\bbug fix\b",
    r"\bregression\b",
    r"\bcrash\b",
    r"\bexception\b",
    r"\bfailure\b",
    r"\bincorrect\b",
    r"\bwrong\b",
    r"\bbroken\b",
    r"\bdefect\b",
    r"\bresolve\b",
    r"\bresolved\b",
    r"\bresolves\b",
    r"\bprevent\b",
    r"\brepair\b",
    r"\bhotfix\b",
    r"\bworkaround\b",
]

# Additional evidence that makes a candidate stronger.
SUPPORTING_PATTERNS = [
    r"#\d+",
    r"\bissue\s+#?\d+\b",
    r"\bpr\s+#?\d+\b",
    r"\bpull request\b",
    r"\btest\b",
    r"\btest case\b",
    r"\bregression test\b",
]


# ============================================================
# HELPERS
# ============================================================

def normalize_text(text: str) -> str:
    """Normalize text for comparisons."""

    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_records() -> list[dict[str, Any]]:
    """Load the raw candidate dataset."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_PATH}"
        )

    records = []

    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    return records


def save_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """Write records as JSONL."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for record in records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def source_file_count(
    record: dict[str, Any],
) -> int:
    """Count Python files changed by a record."""

    count = 0

    for item in record.get(
        "changed_files",
        [],
    ):

        path = item.get(
            "path",
            "",
        )

        if Path(path).suffix.lower() in SOURCE_EXTENSIONS:

            count += 1

    return count


def changed_line_count(
    record: dict[str, Any],
) -> int:
    """Calculate total changed source lines."""

    stats = record.get(
        "diff_stats",
        {},
    )

    added = int(
        stats.get(
            "added_lines",
            0,
        )
        or 0
    )

    deleted = int(
        stats.get(
            "deleted_lines",
            0,
        )
        or 0
    )

    return added + deleted


def patch_hash(
    record: dict[str, Any],
) -> str:
    """
    Create a normalized patch fingerprint.

    This catches exact duplicate patches even when
    record IDs differ.
    """

    patch = record.get(
        "patch",
        "",
    )

    normalized = normalize_text(
        patch
    )

    return hashlib.sha256(
        normalized.encode(
            "utf-8"
        )
    ).hexdigest()


def commit_key(
    record: dict[str, Any],
) -> str:
    """Unique repository + commit identifier."""

    return (
        f"{record.get('repo_id', '')}:"
        f"{record.get('commit', {}).get('hash', '')}"
    )


def has_reject_pattern(
    text: str,
) -> str | None:
    """Return matching rejection pattern."""

    for pattern in REJECT_PATTERNS:

        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):

            return pattern

    return None


def count_matches(
    text: str,
    patterns: list[str],
) -> int:
    """Count distinct positive signals."""

    matches = 0

    for pattern in patterns:

        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):

            matches += 1

    return matches


# ============================================================
# QUALITY SCORING
# ============================================================

def calculate_quality_score(
    record: dict[str, Any],
) -> tuple[int, list[str]]:
    """
    Calculate a stricter quality score.

    This score is independent of the original extractor score.
    """

    commit = record.get(
        "commit",
        {}
    )

    subject = commit.get(
        "subject",
        "",
    )

    body = commit.get(
        "body",
        "",
    )

    text = normalize_text(
        f"{subject} {body}"
    )

    score = 0
    reasons = []

    # --------------------------------------------------------
    # Strong bug language
    # --------------------------------------------------------

    strong_matches = count_matches(
        text,
        STRONG_POSITIVE_PATTERNS,
    )

    if strong_matches >= 3:

        score += 5

        reasons.append(
            "multiple_strong_bug_signals"
        )

    elif strong_matches == 2:

        score += 4

        reasons.append(
            "two_strong_bug_signals"
        )

    elif strong_matches == 1:

        score += 2

        reasons.append(
            "strong_bug_signal"
        )

    # --------------------------------------------------------
    # Supporting evidence
    # --------------------------------------------------------

    support_matches = count_matches(
        text,
        SUPPORTING_PATTERNS,
    )

    if support_matches >= 2:

        score += 3

        reasons.append(
            "multiple_supporting_signals"
        )

    elif support_matches == 1:

        score += 1

        reasons.append(
            "supporting_signal"
        )

    # --------------------------------------------------------
    # Test evidence
    # --------------------------------------------------------

    if re.search(
        r"\btest\b|\btests\b|\btesting\b",
        text,
        flags=re.IGNORECASE,
    ):

        score += 1

        reasons.append(
            "test_related"
        )

    # --------------------------------------------------------
    # Original extractor evidence
    # --------------------------------------------------------

    original_score = int(
        record.get(
            "bug_signal",
            {}
        ).get(
            "score",
            0,
        )
        or 0
    )

    if original_score >= 5:

        score += 2

        reasons.append(
            "strong_original_signal"
        )

    elif original_score >= 3:

        score += 1

        reasons.append(
            "moderate_original_signal"
        )

    return score, reasons


# ============================================================
# RECORD FILTER
# ============================================================

def evaluate_record(
    record: dict[str, Any],
) -> tuple[bool, int, list[str]]:
    """
    Decide whether a raw record is strong enough.

    Returns:
        accepted
        score
        reasons
    """

    reasons = []

    # --------------------------------------------------------
    # Language
    # --------------------------------------------------------

    language = (
        record.get(
            "language"
        )
        or ""
    ).lower()

    if language != "python":

        reasons.append(
            "non_python_repository"
        )

        return False, 0, reasons

    # --------------------------------------------------------
    # Commit identity
    # --------------------------------------------------------

    if not commit_key(record).split(":")[-1]:

        reasons.append(
            "missing_commit_hash"
        )

        return False, 0, reasons

    # --------------------------------------------------------
    # Commit message
    # --------------------------------------------------------

    subject = record.get(
        "commit",
        {}
    ).get(
        "subject",
        "",
    )

    body = record.get(
        "commit",
        {}
    ).get(
        "body",
        "",
    )

    text = normalize_text(
        f"{subject} {body}"
    )

    if not text:

        reasons.append(
            "empty_commit_message"
        )

        return False, 0, reasons

    reject_pattern = has_reject_pattern(
        text
    )

    if reject_pattern:

        reasons.append(
            "reject_message_pattern"
        )

        return False, 0, reasons

    # --------------------------------------------------------
    # Source files
    # --------------------------------------------------------

    python_files = source_file_count(
        record
    )

    if python_files == 0:

        reasons.append(
            "no_python_source_change"
        )

        return False, 0, reasons

    # --------------------------------------------------------
    # Patch size
    # --------------------------------------------------------

    changed_lines = changed_line_count(
        record
    )

    if changed_lines < MIN_CHANGED_LINES:

        reasons.append(
            "empty_change"
        )

        return False, 0, reasons

    if changed_lines > MAX_CHANGED_LINES:

        reasons.append(
            "patch_too_large"
        )

        return False, 0, reasons

    changed_files = len(
        record.get(
            "changed_files",
            [],
        )
    )

    if changed_files > MAX_CHANGED_FILES:

        reasons.append(
            "too_many_changed_files"
        )

        return False, 0, reasons

    # --------------------------------------------------------
    # Patch existence
    # --------------------------------------------------------

    patch = record.get(
        "patch",
        "",
    )

    if not patch.strip():

        reasons.append(
            "empty_patch"
        )

        return False, 0, reasons

    # --------------------------------------------------------
    # Quality score
    # --------------------------------------------------------

    score, score_reasons = calculate_quality_score(
        record
    )

    reasons.extend(
        score_reasons
    )

    # Require meaningful evidence.
    #
    # A commit mentioning only "issue #123" is not enough.
    #

    if score < 4:

        reasons.append(
            "insufficient_bug_evidence"
        )

        return False, score, reasons

    # Require at least one explicit strong bug signal.
    strong_signal = count_matches(
        text,
        STRONG_POSITIVE_PATTERNS,
    )

    if strong_signal == 0:

        reasons.append(
            "no_explicit_bug_signal"
        )

        return False, score, reasons

    return True, score, reasons


# ============================================================
# DUPLICATE REMOVAL
# ============================================================

def remove_duplicates(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """
    Remove exact commit and exact patch duplicates.
    """

    seen_commits = set()
    seen_patches = set()

    unique = []
    duplicates = 0

    for record in records:

        ck = commit_key(
            record
        )

        if ck in seen_commits:

            duplicates += 1
            continue

        pk = patch_hash(
            record
        )

        if pk in seen_patches:

            duplicates += 1
            continue

        seen_commits.add(
            ck
        )

        seen_patches.add(
            pk
        )

        unique.append(
            record
        )

    return unique, duplicates


# ============================================================
# REPOSITORY BALANCING
# ============================================================

def balance_repositories(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Prevent Ansible/Django/etc. from dominating the dataset.

    Repositories with fewer than the cap keep all records.
    """

    grouped = defaultdict(list)

    for record in records:

        grouped[
            record["repo_id"]
        ].append(
            record
        )

    rng = random.Random(
        RANDOM_SEED
    )

    balanced = []

    counts = {}

    for repo_id, repo_records in sorted(
        grouped.items()
    ):

        rng.shuffle(
            repo_records
        )

        if len(repo_records) < MIN_RECORDS_PER_REPOSITORY:

            # Keep it out of the final balanced
            # dataset rather than allowing tiny
            # repositories to create unstable splits.

            counts[repo_id] = 0
            continue

        selected = repo_records[
            :MAX_RECORDS_PER_REPOSITORY
        ]

        balanced.extend(
            selected
        )

        counts[repo_id] = len(
            selected
        )

    rng.shuffle(
        balanced
    )

    return balanced, counts


# ============================================================
# REPOSITORY-AWARE SPLIT
# ============================================================

def split_by_repository(
    records: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """
    Split each repository independently.

    This prevents a single repository from having
    wildly different distributions across splits.

    Note:
        We intentionally keep repositories represented
        in all three splits, because the model should
        learn general bug-fix patterns rather than simply
        memorizing repository identity.

    Leakage prevention is handled by commit/patch
    deduplication before this split.
    """

    grouped = defaultdict(list)

    for record in records:

        grouped[
            record["repo_id"]
        ].append(
            record
        )

    rng = random.Random(
        RANDOM_SEED
    )

    train = []
    validation = []
    test = []

    for repo_id, repo_records in sorted(
        grouped.items()
    ):

        rng.shuffle(
            repo_records
        )

        n = len(
            repo_records
        )

        train_end = int(
            n * TRAIN_RATIO
        )

        validation_end = (
            train_end
            +
            int(
                n * VALIDATION_RATIO
            )
        )

        # Make sure tiny rounding issues
        # don't create empty validation/test
        # sets for usable repositories.

        if n >= 10:

            train_part = repo_records[
                :train_end
            ]

            validation_part = repo_records[
                train_end:validation_end
            ]

            test_part = repo_records[
                validation_end:
            ]

        else:

            train_part = repo_records
            validation_part = []
            test_part = []

        train.extend(
            train_part
        )

        validation.extend(
            validation_part
        )

        test.extend(
            test_part
        )

    rng.shuffle(
        train
    )

    rng.shuffle(
        validation
    )

    rng.shuffle(
        test
    )

    return train, validation, test


# ============================================================
# MAIN PIPELINE
# ============================================================

def main() -> None:

    print()
    print("=" * 70)
    print("RepoMind Dataset Quality Filter")
    print("=" * 70)
    print()

    print(
        f"Input: "
        f"{INPUT_PATH}"
    )

    records = load_records()

    print(
        f"Raw records: "
        f"{len(records)}"
    )

    # --------------------------------------------------------
    # Step 1: quality filtering
    # --------------------------------------------------------

    accepted = []
    rejection_reasons = Counter()
    quality_scores = Counter()

    for record in records:

        ok, score, reasons = evaluate_record(
            record
        )

        if ok:

            record = dict(
                record
            )

            record["dataset_quality"] = {
                "score": score,
                "reasons": reasons,
                "version": "v2",
            }

            accepted.append(
                record
            )

            quality_scores[
                score
            ] += 1

        else:

            for reason in reasons:

                rejection_reasons[
                    reason
                ] += 1

    print()
    print(
        f"After quality filtering: "
        f"{len(accepted)}"
    )

    print()
    print("Top rejection reasons:")

    for reason, count in rejection_reasons.most_common(
        15
    ):

        print(
            f"  {reason}: {count}"
        )

    # --------------------------------------------------------
    # Step 2: duplicate removal
    # --------------------------------------------------------

    unique, duplicate_count = remove_duplicates(
        accepted
    )

    print()
    print(
        f"Removed duplicates: "
        f"{duplicate_count}"
    )

    print(
        f"After deduplication: "
        f"{len(unique)}"
    )

    # --------------------------------------------------------
    # Step 3: repository balancing
    # --------------------------------------------------------

    balanced, repository_counts = balance_repositories(
        unique
    )

    print()
    print(
        "Repository-balanced dataset:"
    )

    for repo_id, count in sorted(
        repository_counts.items()
    ):

        print(
            f"  {repo_id}: {count}"
        )

    print()
    print(
        f"Balanced records: "
        f"{len(balanced)}"
    )

    # --------------------------------------------------------
    # Save filtered dataset
    # --------------------------------------------------------

    save_jsonl(
        FILTERED_PATH,
        balanced,
    )

    print()
    print(
        f"Filtered dataset written to:"
        f"\n  {FILTERED_PATH}"
    )

    # --------------------------------------------------------
    # Step 4: repository-aware split
    # --------------------------------------------------------

    train, validation, test = split_by_repository(
        balanced
    )

    save_jsonl(
        TRAIN_PATH,
        train,
    )

    save_jsonl(
        VALIDATION_PATH,
        validation,
    )

    save_jsonl(
        TEST_PATH,
        test,
    )

    print()
    print("=" * 70)
    print("DATASET SPLIT")
    print("=" * 70)

    print(
        f"Train:       {len(train)}"
    )

    print(
        f"Validation:  {len(validation)}"
    )

    print(
        f"Test:        {len(test)}"
    )

    print()
    print(
        f"Train file:"
        f"\n  {TRAIN_PATH}"
    )

    print(
        f"Validation file:"
        f"\n  {VALIDATION_PATH}"
    )

    print(
        f"Test file:"
        f"\n  {TEST_PATH}"
    )

    # --------------------------------------------------------
    # Final sanity checks
    # --------------------------------------------------------

    all_ids = set()

    duplicate_ids = 0

    for split in (
        train,
        validation,
        test,
    ):

        for record in split:

            record_id = record.get(
                "record_id"
            )

            if record_id in all_ids:

                duplicate_ids += 1

            all_ids.add(
                record_id
            )

    print()
    print(
        "Leakage sanity check:"
    )

    print(
        f"Duplicate record IDs across splits: "
        f"{duplicate_ids}"
    )

    if duplicate_ids == 0:

        print(
            "PASS"
        )

    else:

        print(
            "WARNING"
        )

    # Repository distribution.
    print()
    print(
        "Final repository distribution:"
    )

    for split_name, split_records in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):

        counts = Counter(
            r["repo_id"]
            for r in split_records
        )

        print()
        print(
            f"{split_name}:"
        )

        for repo_id, count in sorted(
            counts.items()
        ):

            print(
                f"  {repo_id}: {count}"
            )

    print()
    print("=" * 70)
    print("QUALITY FILTER COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()