import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


INPUT_PATH = (
    "data/processed/"
    "change_classifier_with_history.json"
)

OUTPUT_DIR = Path(
    "data/processed/"
    "historical_bug_classifier_splits"
)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def load_dataset(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def parse_timestamp(timestamp):
    return datetime.fromisoformat(
        timestamp
    )


def group_by_commit(records):
    """
    Group all file-change records that belong
    to the same commit.

    This prevents data leakage.
    """

    commit_groups = defaultdict(list)

    for record in records:

        commit_hash = record.get(
            "commit_hash"
        )

        if not commit_hash:
            continue

        commit_groups[
            commit_hash
        ].append(record)

    return commit_groups


def get_commit_timestamp(records):
    """
    Return the timestamp of a commit group.
    """

    timestamps = []

    for record in records:

        timestamp = record.get(
            "timestamp"
        )

        if timestamp:

            timestamps.append(
                parse_timestamp(
                    timestamp
                )
            )

    if not timestamps:
        return datetime.min

    return min(
        timestamps
    )


def split_commits_temporally(
    commit_groups
):
    """
    Sort commits oldest -> newest.

    Earliest 70%   -> train
    Next 15%       -> validation
    Latest 15%     -> test
    """

    commit_items = list(
        commit_groups.items()
    )

    commit_items.sort(
        key=lambda item:
        get_commit_timestamp(
            item[1]
        )
    )

    total_commits = len(
        commit_items
    )

    train_end = int(
        total_commits
        *
        TRAIN_RATIO
    )

    validation_end = int(
        total_commits
        *
        (
            TRAIN_RATIO
            +
            VAL_RATIO
        )
    )

    train_commits = commit_items[
        :train_end
    ]

    validation_commits = commit_items[
        train_end:
        validation_end
    ]

    test_commits = commit_items[
        validation_end:
    ]

    return (
        train_commits,
        validation_commits,
        test_commits
    )


def flatten_commit_groups(
    commit_items
):
    """
    Convert grouped commits back into
    normal records.
    """

    records = []

    for _, commit_records in commit_items:

        records.extend(
            commit_records
        )

    return records


def save_json(
    data,
    path
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=2
        )


def get_label_stats(records):
    positives = sum(
        1
        for record in records
        if record.get(
            "label"
        ) == 1
    )

    negatives = sum(
        1
        for record in records
        if record.get(
            "label"
        ) == 0
    )

    return (
        positives,
        negatives
    )


def get_commit_hashes(records):
    return {
        record.get(
            "commit_hash"
        )
        for record in records
        if record.get(
            "commit_hash"
        )
    }


def verify_no_leakage(
    train_records,
    validation_records,
    test_records
):
    """
    Make sure the same commit never appears
    in multiple splits.
    """

    train_hashes = get_commit_hashes(
        train_records
    )

    validation_hashes = get_commit_hashes(
        validation_records
    )

    test_hashes = get_commit_hashes(
        test_records
    )

    train_validation_overlap = (
        train_hashes
        &
        validation_hashes
    )

    train_test_overlap = (
        train_hashes
        &
        test_hashes
    )

    validation_test_overlap = (
        validation_hashes
        &
        test_hashes
    )

    return {
        "train_validation_overlap":
            len(
                train_validation_overlap
            ),

        "train_test_overlap":
            len(
                train_test_overlap
            ),

        "validation_test_overlap":
            len(
                validation_test_overlap
            ),
    }


def verify_historical_features(records):
    """
    Check how many records actually contain
    historical_features.
    """

    with_history = sum(
        1
        for record in records
        if isinstance(
            record.get(
                "historical_features"
            ),
            dict
        )
    )

    return with_history


if __name__ == "__main__":

    print(
        "\nLoading historical classifier dataset..."
    )

    records = load_dataset(
        INPUT_PATH
    )

    print(
        f"Total records: "
        f"{len(records)}"
    )

    history_count = (
        verify_historical_features(
            records
        )
    )

    print(
        f"Records with historical features: "
        f"{history_count}"
    )

    # ==========================================
    # GROUP BY COMMIT
    # ==========================================

    commit_groups = group_by_commit(
        records
    )

    print(
        f"Unique commits: "
        f"{len(commit_groups)}"
    )

    # ==========================================
    # TEMPORAL SPLIT
    # ==========================================

    (
        train_commits,
        validation_commits,
        test_commits
    ) = split_commits_temporally(
        commit_groups
    )

    # ==========================================
    # FLATTEN
    # ==========================================

    train_records = flatten_commit_groups(
        train_commits
    )

    validation_records = (
        flatten_commit_groups(
            validation_commits
        )
    )

    test_records = flatten_commit_groups(
        test_commits
    )

    # ==========================================
    # LABEL STATS
    # ==========================================

    train_positive, train_negative = (
        get_label_stats(
            train_records
        )
    )

    (
        validation_positive,
        validation_negative
    ) = get_label_stats(
        validation_records
    )

    test_positive, test_negative = (
        get_label_stats(
            test_records
        )
    )

    # ==========================================
    # LEAKAGE CHECK
    # ==========================================

    leakage = verify_no_leakage(
        train_records,
        validation_records,
        test_records
    )

    # ==========================================
    # SAVE FILES
    # ==========================================

    save_json(
        train_records,
        OUTPUT_DIR
        /
        "train.json"
    )

    save_json(
        validation_records,
        OUTPUT_DIR
        /
        "validation.json"
    )

    save_json(
        test_records,
        OUTPUT_DIR
        /
        "test.json"
    )

    # ==========================================
    # PRINT RESULTS
    # ==========================================

    print(
        "\nRepoMind Historical Dataset Splits"
    )

    print(
        "----------------------------------"
    )

    print(
        f"Train records: "
        f"{len(train_records)}"
    )

    print(
        f"Train positives: "
        f"{train_positive}"
    )

    print(
        f"Train negatives: "
        f"{train_negative}"
    )

    print()

    print(
        f"Validation records: "
        f"{len(validation_records)}"
    )

    print(
        f"Validation positives: "
        f"{validation_positive}"
    )

    print(
        f"Validation negatives: "
        f"{validation_negative}"
    )

    print()

    print(
        f"Test records: "
        f"{len(test_records)}"
    )

    print(
        f"Test positives: "
        f"{test_positive}"
    )

    print(
        f"Test negatives: "
        f"{test_negative}"
    )

    print(
        "\nLeakage Check"
    )

    print(
        "----------------------------------"
    )

    print(
        "Train/Validation overlap: "
        f"{leakage['train_validation_overlap']}"
    )

    print(
        "Train/Test overlap: "
        f"{leakage['train_test_overlap']}"
    )

    print(
        "Validation/Test overlap: "
        f"{leakage['validation_test_overlap']}"
    )

    print(
        "\nSaved to:"
    )

    print(
        OUTPUT_DIR
    )