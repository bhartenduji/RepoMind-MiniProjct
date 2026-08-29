import json
import random
import re
from pathlib import Path


DIFF_HISTORY_PATH = "data/processed/diff_history.json"
OUTPUT_PATH = "data/processed/change_classifier_dataset.json"

RANDOM_SEED = 42


BUG_KEYWORDS = [
    "fix",
    "fixed",
    "bug",
    "issue",
    "error",
    "crash",
    "regression",
    "incorrect",
    "wrong",
    "fail",
    "failure",
    "broken",
    "patch",
]


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def is_bug_fix_message(message):
    if not message:
        return False

    normalized = message.lower()

    for keyword in BUG_KEYWORDS:
        pattern = rf"\b{re.escape(keyword)}\b"

        if re.search(pattern, normalized):
            return True

    return False


def build_record(record, label):
    return {
        "commit_hash": record.get("commit_hash"),
        "timestamp": record.get("timestamp"),
        "author_name": record.get("author_name"),
        "message": record.get("message"),
        "file_path": record.get("file_path"),
        "change_type": record.get("change_type"),
        "patch": record.get("patch"),
        "added_lines": record.get("added_lines", []),
        "removed_lines": record.get("removed_lines", []),
        "added_line_count": record.get(
            "added_line_count",
            0
        ),
        "removed_line_count": record.get(
            "removed_line_count",
            0
        ),
        "label": label,
    }


def build_classifier_dataset(records):
    positive_records = []
    negative_records = []

    for record in records:
        message = record.get(
            "message",
            ""
        )

        if is_bug_fix_message(message):

            positive_records.append(
                build_record(
                    record,
                    label=1
                )
            )

        else:

            negative_records.append(
                build_record(
                    record,
                    label=0
                )
            )

    print(
        f"Positive candidates: "
        f"{len(positive_records)}"
    )

    print(
        f"Negative candidates: "
        f"{len(negative_records)}"
    )

    # -----------------------------------------
    # BALANCE DATASET
    # -----------------------------------------

    random.seed(
        RANDOM_SEED
    )

    # Use the smaller class size so both
    # classes contain the same number of records.
    target_size = min(
        len(positive_records),
        len(negative_records)
    )

    positive_sample = random.sample(
        positive_records,
        target_size
    )

    negative_sample = random.sample(
        negative_records,
        target_size
    )

    dataset = (
        positive_sample
        +
        negative_sample
    )

    random.shuffle(
        dataset
    )

    return (
        dataset,
        len(positive_records),
        len(negative_records)
    )


def save_dataset(
    dataset,
    output_path
):
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            dataset,
            file,
            indent=2
        )


if __name__ == "__main__":

    print(
        "\nLoading diff history..."
    )

    records = load_json(
        DIFF_HISTORY_PATH
    )

    print(
        f"Total diff records: "
        f"{len(records)}"
    )

    dataset, positives, negatives = (
        build_classifier_dataset(
            records
        )
    )

    save_dataset(
        dataset,
        OUTPUT_PATH
    )

    positive_final = sum(
        1
        for record in dataset
        if record["label"] == 1
    )

    negative_final = sum(
        1
        for record in dataset
        if record["label"] == 0
    )

    print(
        "\nRepoMind Change Classifier Dataset"
    )

    print(
        "----------------------------------"
    )

    print(
        f"Original positives: "
        f"{positives}"
    )

    print(
        f"Original negatives: "
        f"{negatives}"
    )

    print(
        f"Final dataset size: "
        f"{len(dataset)}"
    )

    print(
        f"Positive examples: "
        f"{positive_final}"
    )

    print(
        f"Negative examples: "
        f"{negative_final}"
    )

    print(
        f"Saved to: "
        f"{OUTPUT_PATH}"
    )

    print(
        "\nFirst 10 labels:"
    )

    for record in dataset[:10]:

        print(
            f"label={record['label']} "
            f"| {record['message'][:80]}"
        )