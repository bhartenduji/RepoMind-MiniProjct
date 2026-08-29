import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


INPUT_PATH = "data/processed/change_classifier_dataset.json"

OUTPUT_DIR = Path(
    "data/processed/bug_classifier_splits"
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
    Group all file-level records belonging
    to the same Git commit.

    This prevents records from one commit
    appearing in both train and test.
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
    Get timestamp for one commit group.
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

    return min(timestamps)


def split_commits_temporally(
    commit_groups
):

    commit_items = list(
        commit_groups.items()
    )

    # Sort oldest -> newest
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

    val_end = int(
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

    val_commits = commit_items[
        train_end:val_end
    ]

    test_commits = commit_items[
        val_end:
    ]

    return (
        train_commits,
        val_commits,
        test_commits
    )


def flatten_commit_groups(
    commit_items
):

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


def get_commit_hashes(
    records
):

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
    train,
    val,
    test
):

    train_hashes = (
        get_commit_hashes(
            train
        )
    )

    val_hashes = (
        get_commit_hashes(
            val
        )
    )

    test_hashes = (
        get_commit_hashes(
            test
        )
    )

    train_val_overlap = (
        train_hashes
        &
        val_hashes
    )

    train_test_overlap = (
        train_hashes
        &
        test_hashes
    )

    val_test_overlap = (
        val_hashes
        &
        test_hashes
    )

    return {
        "train_val_overlap":
            len(
                train_val_overlap
            ),

        "train_test_overlap":
            len(
                train_test_overlap
            ),

        "val_test_overlap":
            len(
                val_test_overlap
            ),
    }


if __name__ == "__main__":

    print(
        "\nLoading classifier dataset..."
    )

    records = load_dataset(
        INPUT_PATH
    )

    print(
        f"Total records: "
        f"{len(records)}"
    )

    commit_groups = (
        group_by_commit(
            records
        )
    )

    print(
        f"Unique commits: "
        f"{len(commit_groups)}"
    )

    (
        train_commits,
        val_commits,
        test_commits
    ) = split_commits_temporally(
        commit_groups
    )

    train_records = (
        flatten_commit_groups(
            train_commits
        )
    )

    val_records = (
        flatten_commit_groups(
            val_commits
        )
    )

    test_records = (
        flatten_commit_groups(
            test_commits
        )
    )

    train_pos, train_neg = (
        get_label_stats(
            train_records
        )
    )

    val_pos, val_neg = (
        get_label_stats(
            val_records
        )
    )

    test_pos, test_neg = (
        get_label_stats(
            test_records
        )
    )

    leakage = (
        verify_no_leakage(
            train_records,
            val_records,
            test_records
        )
    )

    save_json(
        train_records,
        OUTPUT_DIR
        /
        "train.json"
    )

    save_json(
        val_records,
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

    print(
        "\nRepoMind Bug Classifier Splits"
    )

    print(
        "------------------------------"
    )

    print(
        f"Train records: "
        f"{len(train_records)}"
    )

    print(
        f"Train positives: "
        f"{train_pos}"
    )

    print(
        f"Train negatives: "
        f"{train_neg}"
    )

    print()

    print(
        f"Validation records: "
        f"{len(val_records)}"
    )

    print(
        f"Validation positives: "
        f"{val_pos}"
    )

    print(
        f"Validation negatives: "
        f"{val_neg}"
    )

    print()

    print(
        f"Test records: "
        f"{len(test_records)}"
    )

    print(
        f"Test positives: "
        f"{test_pos}"
    )

    print(
        f"Test negatives: "
        f"{test_neg}"
    )

    print(
        "\nLeakage Check"
    )

    print(
        "------------------------------"
    )

    print(
        f"Train/Validation overlap: "
        f"{leakage['train_val_overlap']}"
    )

    print(
        f"Train/Test overlap: "
        f"{leakage['train_test_overlap']}"
    )

    print(
        f"Validation/Test overlap: "
        f"{leakage['val_test_overlap']}"
    )

    print(
        "\nSaved to:"
    )

    print(
        OUTPUT_DIR
    )