import json
from pathlib import Path

from git import Repo


REPO_PATH = "data/raw/test_repo"
OUTPUT_PATH = "data/processed/file_change_history.json"


def extract_file_change_history(repo_path):
    repo = Repo(repo_path)

    records = []

    for commit in repo.iter_commits():

        # Skip commits without parents for now
        if not commit.parents:
            continue

        parent = commit.parents[0]

        try:
            diff_index = parent.diff(
                commit,
                create_patch=False
            )
        except Exception:
            continue

        # GitPython gives per-file stats here
        stats_files = commit.stats.files

        for diff_item in diff_index:

            old_path = diff_item.a_path
            new_path = diff_item.b_path

            # Prefer current/new path when possible
            file_path = new_path or old_path

            if not file_path:
                continue

            file_stats = stats_files.get(
                file_path,
                {}
            )

            insertions = file_stats.get(
                "insertions",
                0
            )

            deletions = file_stats.get(
                "deletions",
                0
            )

            # Determine change type
            change_type = "modified"

            if diff_item.new_file:
                change_type = "added"

            elif diff_item.deleted_file:
                change_type = "deleted"

            elif diff_item.renamed_file:
                change_type = "renamed"

            record = {
                "commit_hash": commit.hexsha,
                "timestamp": (
                    commit.committed_datetime.isoformat()
                ),
                "author_name": commit.author.name,
                "author_email": commit.author.email,
                "message": commit.message.strip(),

                "file_path": file_path,
                "old_path": old_path,
                "new_path": new_path,

                "change_type": change_type,

                "insertions": insertions,
                "deletions": deletions,
                "total_changes": (
                    insertions + deletions
                ),
            }

            records.append(record)

    return records


def save_file_change_history(
    records,
    output_path
):
    output_path = Path(output_path)

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

    records = extract_file_change_history(
        REPO_PATH
    )

    save_file_change_history(
        records,
        OUTPUT_PATH
    )

    print("\nRepoMind File Change History")
    print("----------------------------")

    print(
        f"Total file-change records: "
        f"{len(records)}"
    )

    print(
        f"Saved to: "
        f"{OUTPUT_PATH}"
    )

    print("\nFirst 10 records:")

    for record in records[:10]:

        print("-" * 60)

        print(
            f"Commit: "
            f"{record['commit_hash'][:10]}"
        )

        print(
            f"File: "
            f"{record['file_path']}"
        )

        print(
            f"Type: "
            f"{record['change_type']}"
        )

        print(
            f"Insertions: "
            f"{record['insertions']}"
        )

        print(
            f"Deletions: "
            f"{record['deletions']}"
        )

        print(
            f"Message: "
            f"{record['message']}"
        )