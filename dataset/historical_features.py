import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


INPUT_PATH = "data/processed/change_classifier_dataset.json"
HISTORY_PATH = "data/processed/file_change_history.json"

OUTPUT_PATH = (
    "data/processed/change_classifier_with_history.json"
)


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def parse_timestamp(value):
    if not value:
        return None

    return datetime.fromisoformat(
        value
    )


def build_history_index(history_records):
    """
    Group historical change records by file path.

    Example:

    src/flask/app.py
        ->
    [
        commit A,
        commit B,
        commit C,
        ...
    ]
    """

    file_history = defaultdict(list)

    for record in history_records:

        file_path = record.get(
            "file_path"
        )

        timestamp = parse_timestamp(
            record.get(
                "timestamp"
            )
        )

        if not file_path or not timestamp:
            continue

        record_copy = dict(record)

        record_copy[
            "_parsed_timestamp"
        ] = timestamp

        file_history[
            file_path
        ].append(
            record_copy
        )

    # Sort every file's history
    # oldest -> newest
    for file_path in file_history:

        file_history[file_path].sort(
            key=lambda item:
            item["_parsed_timestamp"]
        )

    return file_history


def is_bug_like_message(message):
    """
    Simple historical bug-fix heuristic.

    This is intentionally similar to
    our earlier candidate labels.
    """

    if not message:
        return False

    message = message.lower()

    keywords = [
        "fix",
        "fixed",
        "bug",
        "bugfix",
        "bug fix",
        "error",
        "crash",
        "regression",
        "failure",
        "broken",
        "incorrect",
    ]

    return any(
        keyword in message
        for keyword in keywords
    )


def calculate_historical_features(
    file_path,
    current_timestamp,
    file_history
):
    """
    Compute features using ONLY changes
    before current_timestamp.
    """

    default_features = {
        "prior_commit_count": 0,
        "prior_bug_fix_count": 0,
        "prior_total_insertions": 0,
        "prior_total_deletions": 0,
        "prior_total_churn": 0,
        "prior_unique_authors": 0,
        "days_since_last_change": -1,
        "historical_bug_fix_ratio": 0.0,
        "average_prior_churn": 0.0,
    }

    if not file_path:
        return default_features

    history = file_history.get(
        file_path,
        []
    )

    if not history:
        return default_features

    previous_records = []

    for record in history:

        record_time = record[
            "_parsed_timestamp"
        ]

        if record_time < current_timestamp:

            previous_records.append(
                record
            )

        else:
            break

    if not previous_records:
        return default_features

    prior_commit_count = len(
        previous_records
    )

    total_insertions = sum(
        record.get(
            "insertions",
            0
        )
        for record in previous_records
    )

    total_deletions = sum(
        record.get(
            "deletions",
            0
        )
        for record in previous_records
    )

    total_churn = (
        total_insertions
        +
        total_deletions
    )

    authors = {
        record.get(
            "author_email"
        )
        or record.get(
            "author_name"
        )
        for record in previous_records
        if (
            record.get(
                "author_email"
            )
            or record.get(
                "author_name"
            )
        )
    }

    bug_fix_count = sum(
        1
        for record in previous_records
        if is_bug_like_message(
            record.get(
                "message",
                ""
            )
        )
    )

    last_change_time = (
        previous_records[-1][
            "_parsed_timestamp"
        ]
    )

    days_since_last_change = (
        current_timestamp
        -
        last_change_time
    ).days

    bug_fix_ratio = (
        bug_fix_count
        /
        prior_commit_count
        if prior_commit_count > 0
        else 0.0
    )

    average_prior_churn = (
        total_churn
        /
        prior_commit_count
        if prior_commit_count > 0
        else 0.0
    )

    return {
        "prior_commit_count":
            prior_commit_count,

        "prior_bug_fix_count":
            bug_fix_count,

        "prior_total_insertions":
            total_insertions,

        "prior_total_deletions":
            total_deletions,

        "prior_total_churn":
            total_churn,

        "prior_unique_authors":
            len(authors),

        "days_since_last_change":
            days_since_last_change,

        "historical_bug_fix_ratio":
            round(
                bug_fix_ratio,
                6
            ),

        "average_prior_churn":
            round(
                average_prior_churn,
                6
            ),
    }


def enrich_dataset(
    dataset,
    file_history
):

    enriched_records = []

    for record in dataset:

        timestamp = parse_timestamp(
            record.get(
                "timestamp"
            )
        )

        if not timestamp:
            continue

        file_path = record.get(
            "file_path"
        )

        historical_features = (
            calculate_historical_features(
                file_path,
                timestamp,
                file_history
            )
        )

        enriched_record = dict(
            record
        )

        enriched_record[
            "historical_features"
        ] = historical_features

        enriched_records.append(
            enriched_record
        )

    return enriched_records


def save_json(
    data,
    path
):

    path = Path(
        path
    )

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


if __name__ == "__main__":

    print(
        "\nLoading classifier dataset..."
    )

    dataset = load_json(
        INPUT_PATH
    )

    print(
        f"Classifier records: "
        f"{len(dataset)}"
    )

    print(
        "\nLoading file history..."
    )

    history_records = load_json(
        HISTORY_PATH
    )

    print(
        f"Historical records: "
        f"{len(history_records)}"
    )

    print(
        "\nBuilding file history index..."
    )

    file_history = build_history_index(
        history_records
    )

    print(
        f"Files with history: "
        f"{len(file_history)}"
    )

    print(
        "\nCalculating historical features..."
    )

    enriched = enrich_dataset(
        dataset,
        file_history
    )

    save_json(
        enriched,
        OUTPUT_PATH
    )

    print(
        "\nRepoMind Historical Features"
    )

    print(
        "-----------------------------"
    )

    print(
        f"Enriched records: "
        f"{len(enriched)}"
    )

    print(
        f"Saved to: "
        f"{OUTPUT_PATH}"
    )

    print(
        "\nFirst 5 examples:"
    )

    for record in enriched[:5]:

        print(
            "-" * 60
        )

        print(
            f"File: "
            f"{record['file_path']}"
        )

        print(
            f"Label: "
            f"{record['label']}"
        )

        print(
            "Historical features:"
        )

        for (
            key,
            value
        ) in record[
            "historical_features"
        ].items():

            print(
                f"  {key}: "
                f"{value}"
            )