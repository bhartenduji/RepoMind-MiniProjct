"""
RepoMind - Binary Classification Dataset Builder

Builds a binary classification dataset from:

Positive examples:
    data/processed/bug_fix_quality_filtered.jsonl

Negative examples:
    data/processed/non_bug_fix_candidates.jsonl

Labels:
    1 = bug-fix / positive
    0 = non-bug-fix / negative

Outputs:
    data/processed/classification_dataset.jsonl

    data/processed/classification_splits/train.jsonl
    data/processed/classification_splits/validation.jsonl
    data/processed/classification_splits/test.jsonl

Split:
    80% train
    10% validation
    10% test

Important:
    - Uses ALL available positive examples.
    - Uses ALL available negative examples.
    - Does not artificially balance or discard examples.
    - Removes duplicate record_id values.
    - Preserves the original record fields.
    - Adds a top-level "label" field.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


# ============================================================
# PATHS
# ============================================================

POSITIVE_PATH = Path(
    "data/processed/bug_fix_quality_filtered.jsonl"
)

NEGATIVE_PATH = Path(
    "data/processed/non_bug_fix_candidates.jsonl"
)

OUTPUT_PATH = Path(
    "data/processed/classification_dataset.jsonl"
)

SPLIT_DIR = Path(
    "data/processed/classification_splits"
)

TRAIN_PATH = SPLIT_DIR / "train.jsonl"
VALIDATION_PATH = SPLIT_DIR / "validation.jsonl"
TEST_PATH = SPLIT_DIR / "test.jsonl"


# ============================================================
# CONFIG
# ============================================================

SEED = 42

TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10


# ============================================================
# VALIDATION
# ============================================================

def validate_config() -> None:
    total = (
        TRAIN_RATIO
        + VALIDATION_RATIO
        + TEST_RATIO
    )

    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            "Train/validation/test ratios must sum to 1.0"
        )


# ============================================================
# JSONL
# ============================================================

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """
    Load records from a JSONL file.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}"
        )

    records: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} "
                    f"at line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object in {path} "
                    f"at line {line_number}"
                )

            records.append(record)

    return records


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """
    Write records to JSONL.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# RECORD VALIDATION
# ============================================================

def get_record_id(
    record: dict[str, Any],
    index: int,
    source: str,
) -> str:
    """
    Return a stable record identifier.

    record_id is preferred. If it is missing, create a
    source-specific fallback identifier.
    """

    record_id = record.get("record_id")

    if record_id is not None:
        record_id = str(record_id).strip()

        if record_id:
            return record_id

    return f"{source}:missing_record_id:{index}"


def prepare_records(
    records: list[dict[str, Any]],
    label: int,
    source: str,
) -> list[dict[str, Any]]:
    """
    Add the classification label to every record.

    Existing label fields are overwritten intentionally so that
    the positive/negative source file determines the ground truth.
    """

    prepared: list[dict[str, Any]] = []

    for index, original in enumerate(
        records,
        start=1,
    ):
        record = dict(original)

        record["label"] = label

        record["record_id"] = get_record_id(
            record,
            index,
            source,
        )

        prepared.append(record)

    return prepared


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """
    Remove duplicate record_id values.

    The first occurrence is retained.
    """

    seen: set[str] = set()

    unique: list[dict[str, Any]] = []

    duplicates = 0

    for record in records:

        record_id = str(
            record["record_id"]
        )

        if record_id in seen:
            duplicates += 1
            continue

        seen.add(record_id)
        unique.append(record)

    return unique, duplicates


# ============================================================
# CROSS-CLASS DUPLICATE CHECK
# ============================================================

def remove_cross_class_duplicates(
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    int,
]:
    """
    Ensure a record cannot appear in both classes.

    Positive examples take precedence if a record_id appears
    in both source datasets.
    """

    positive_ids = {
        record["record_id"]
        for record in positives
    }

    filtered_negatives: list[dict[str, Any]] = []

    conflicts = 0

    for record in negatives:

        if record["record_id"] in positive_ids:
            conflicts += 1
            continue

        filtered_negatives.append(record)

    return (
        positives,
        filtered_negatives,
        conflicts,
    )


# ============================================================
# STRATIFIED SPLIT
# ============================================================

def split_class(
    records: list[dict[str, Any]],
    rng: random.Random,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """
    Split one class into train/validation/test.

    This guarantees approximately the same class proportion
    in every split.
    """

    records = list(records)

    rng.shuffle(records)

    total = len(records)

    train_count = int(
        total * TRAIN_RATIO
    )

    validation_count = int(
        total * VALIDATION_RATIO
    )

    train = records[
        :train_count
    ]

    validation = records[
        train_count:
        train_count + validation_count
    ]

    test = records[
        train_count + validation_count:
    ]

    return (
        train,
        validation,
        test,
    )


def create_stratified_splits(
    records: list[dict[str, Any]],
    seed: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """
    Create deterministic stratified train/validation/test splits.
    """

    positives = [
        record
        for record in records
        if record["label"] == 1
    ]

    negatives = [
        record
        for record in records
        if record["label"] == 0
    ]

    rng = random.Random(seed)

    (
        positive_train,
        positive_validation,
        positive_test,
    ) = split_class(
        positives,
        rng,
    )

    (
        negative_train,
        negative_validation,
        negative_test,
    ) = split_class(
        negatives,
        rng,
    )

    train = (
        positive_train
        + negative_train
    )

    validation = (
        positive_validation
        + negative_validation
    )

    test = (
        positive_test
        + negative_test
    )

    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)

    return (
        train,
        validation,
        test,
    )


# ============================================================
# STATISTICS
# ============================================================

def print_class_distribution(
    name: str,
    records: list[dict[str, Any]],
) -> None:

    counts = Counter(
        record["label"]
        for record in records
    )

    total = len(records)

    positive = counts.get(1, 0)
    negative = counts.get(0, 0)

    print()
    print("=" * 60)
    print(name.upper())
    print("=" * 60)

    print(
        f"Total:      {total}"
    )

    print(
        f"label 0:    {negative}"
    )

    print(
        f"label 1:    {positive}"
    )

    if total:
        print(
            f"Positive %: "
            f"{positive / total * 100:.2f}%"
        )

        print(
            f"Negative %: "
            f"{negative / total * 100:.2f}%"
        )


def print_repository_distribution(
    name: str,
    records: list[dict[str, Any]],
) -> None:

    counts = Counter(
        record.get(
            "repo_id",
            "unknown",
        )
        for record in records
    )

    print()
    print(
        f"{name} repositories:"
    )

    for repo, count in counts.most_common():
        print(
            f"  {repo}: {count}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    validate_config()

    print()
    print("=" * 70)
    print("RepoMind Binary Classification Dataset Builder")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Load positive examples
    # --------------------------------------------------------

    print(
        f"Loading positive examples:\n"
        f"  {POSITIVE_PATH}"
    )

    positive_raw = load_jsonl(
        POSITIVE_PATH
    )

    print(
        f"Positive source records: "
        f"{len(positive_raw)}"
    )

    # --------------------------------------------------------
    # Load negative examples
    # --------------------------------------------------------

    print()
    print(
        f"Loading negative examples:\n"
        f"  {NEGATIVE_PATH}"
    )

    negative_raw = load_jsonl(
        NEGATIVE_PATH
    )

    print(
        f"Negative source records: "
        f"{len(negative_raw)}"
    )

    # --------------------------------------------------------
    # Prepare labels
    # --------------------------------------------------------

    positives = prepare_records(
        positive_raw,
        label=1,
        source="positive",
    )

    negatives = prepare_records(
        negative_raw,
        label=0,
        source="negative",
    )

    # --------------------------------------------------------
    # Deduplicate within each class
    # --------------------------------------------------------

    positives, positive_duplicates = (
        deduplicate_records(
            positives
        )
    )

    negatives, negative_duplicates = (
        deduplicate_records(
            negatives
        )
    )

    print()
    print(
        "Duplicate removal:"
    )

    print(
        f"  Positive duplicates removed: "
        f"{positive_duplicates}"
    )

    print(
        f"  Negative duplicates removed: "
        f"{negative_duplicates}"
    )

    # --------------------------------------------------------
    # Remove cross-class conflicts
    # --------------------------------------------------------

    (
        positives,
        negatives,
        cross_class_conflicts,
    ) = remove_cross_class_duplicates(
        positives,
        negatives,
    )

    print(
        f"  Cross-class conflicts removed: "
        f"{cross_class_conflicts}"
    )

    # --------------------------------------------------------
    # Final dataset
    # --------------------------------------------------------

    records = (
        positives
        + negatives
    )

    print()
    print(
        "=" * 70
    )

    print(
        "FINAL CLASS DISTRIBUTION"
    )

    print(
        "=" * 70
    )

    print(
        f"Positive examples: "
        f"{len(positives)}"
    )

    print(
        f"Negative examples: "
        f"{len(negatives)}"
    )

    print(
        f"Total examples:    "
        f"{len(records)}"
    )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    labels = Counter(
        record["label"]
        for record in records
    )

    if labels.get(1, 0) == 0:
        raise RuntimeError(
            "No positive examples found."
        )

    if labels.get(0, 0) == 0:
        raise RuntimeError(
            "No negative examples found."
        )

    record_ids = [
        record["record_id"]
        for record in records
    ]

    if len(record_ids) != len(
        set(record_ids)
    ):
        raise RuntimeError(
            "Duplicate record_id values remain."
        )

    # --------------------------------------------------------
    # Write complete classification dataset
    # --------------------------------------------------------

    print()
    print(
        "Writing complete classification dataset..."
    )

    write_jsonl(
        OUTPUT_PATH,
        records,
    )

    # --------------------------------------------------------
    # Create stratified splits
    # --------------------------------------------------------

    print()
    print(
        "Creating stratified train/validation/test splits..."
    )

    (
        train_records,
        validation_records,
        test_records,
    ) = create_stratified_splits(
        records,
        seed=SEED,
    )

    # --------------------------------------------------------
    # Write splits
    # --------------------------------------------------------

    write_jsonl(
        TRAIN_PATH,
        train_records,
    )

    write_jsonl(
        VALIDATION_PATH,
        validation_records,
    )

    write_jsonl(
        TEST_PATH,
        test_records,
    )

    # --------------------------------------------------------
    # Print split statistics
    # --------------------------------------------------------

    print_class_distribution(
        "Train",
        train_records,
    )

    print_class_distribution(
        "Validation",
        validation_records,
    )

    print_class_distribution(
        "Test",
        test_records,
    )

    # --------------------------------------------------------
    # Repository distributions
    # --------------------------------------------------------

    print_repository_distribution(
        "Train",
        train_records,
    )

    print_repository_distribution(
        "Validation",
        validation_records,
    )

    print_repository_distribution(
        "Test",
        test_records,
    )

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    split_total = (
        len(train_records)
        + len(validation_records)
        + len(test_records)
    )

    if split_total != len(records):
        raise RuntimeError(
            "Split sizes do not add up to the "
            "complete dataset."
        )

    train_ids = {
        record["record_id"]
        for record in train_records
    }

    validation_ids = {
        record["record_id"]
        for record in validation_records
    }

    test_ids = {
        record["record_id"]
        for record in test_records
    }

    if train_ids & validation_ids:
        raise RuntimeError(
            "Train/validation data leakage detected."
        )

    if train_ids & test_ids:
        raise RuntimeError(
            "Train/test data leakage detected."
        )

    if validation_ids & test_ids:
        raise RuntimeError(
            "Validation/test data leakage detected."
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CLASSIFICATION DATASET COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Total:      {len(records)}"
    )

    print(
        f"Train:      {len(train_records)}"
    )

    print(
        f"Validation: {len(validation_records)}"
    )

    print(
        f"Test:       {len(test_records)}"
    )

    print()
    print(
        f"Positive:   {labels.get(1, 0)}"
    )

    print(
        f"Negative:   {labels.get(0, 0)}"
    )

    print()
    print(
        f"Output:     {OUTPUT_PATH}"
    )

    print(
        f"Train:      {TRAIN_PATH}"
    )

    print(
        f"Validation: {VALIDATION_PATH}"
    )

    print(
        f"Test:       {TEST_PATH}"
    )

    print()
    print(
        "No examples were artificially discarded "
        "for class balancing."
    )

    print(
        "Stratified split seed:",
        SEED,
    )

    print()
    print(
        "Dataset build completed successfully."
    )


if __name__ == "__main__":
    main()