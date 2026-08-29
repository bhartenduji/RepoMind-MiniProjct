import json
import re
from pathlib import Path


DIFF_HISTORY_PATH = "data/processed/diff_history.json"
OUTPUT_PATH = "data/processed/bug_fix_dataset.json"


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


def load_diff_history(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def is_bug_fix_message(message):
    """
    First-pass heuristic:
    classify a commit message as bug-fix-like
    if it contains one of our keywords.
    """

    if not message:
        return False

    normalized = message.lower()

    for keyword in BUG_KEYWORDS:

        pattern = rf"\b{re.escape(keyword)}\b"

        if re.search(pattern, normalized):
            return True

    return False


def mine_bug_fixes(records):

    bug_records = []

    for record in records:

        message = record.get(
            "message",
            ""
        )

        if not is_bug_fix_message(
            message
        ):
            continue

        bug_record = {
            "commit_hash":
                record.get("commit_hash"),

            "timestamp":
                record.get("timestamp"),

            "author_name":
                record.get("author_name"),

            "message":
                message,

            "file_path":
                record.get("file_path"),

            "change_type":
                record.get("change_type"),

            "patch":
                record.get("patch"),

            "added_lines":
                record.get("added_lines", []),

            "removed_lines":
                record.get("removed_lines", []),

            "added_line_count":
                record.get("added_line_count", 0),

            "removed_line_count":
                record.get("removed_line_count", 0),

            "label":
                "bug_fix"
        }

        bug_records.append(
            bug_record
        )

    return bug_records


def save_dataset(
    records,
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
            records,
            file,
            indent=2
        )


if __name__ == "__main__":

    records = load_diff_history(
        DIFF_HISTORY_PATH
    )

    bug_records = mine_bug_fixes(
        records
    )

    save_dataset(
        bug_records,
        OUTPUT_PATH
    )

    print(
        "\nRepoMind Bug Fix Miner"
    )

    print(
        "----------------------"
    )

    print(
        f"Total diff records: "
        f"{len(records)}"
    )

    print(
        f"Bug-fix records: "
        f"{len(bug_records)}"
    )

    print(
        f"Saved to: "
        f"{OUTPUT_PATH}"
    )

    print(
        "\nFirst 10 bug-fix examples:"
    )

    for record in bug_records[:10]:

        print(
            "-" * 60
        )

        print(
            f"Commit: "
            f"{record['commit_hash'][:10]}"
        )

        print(
            f"File: "
            f"{record['file_path']}"
        )

        print(
            f"Message: "
            f"{record['message']}"
        )

        print(
            f"Added lines: "
            f"{record['added_line_count']}"
        )

        print(
            f"Removed lines: "
            f"{record['removed_line_count']}"
        )