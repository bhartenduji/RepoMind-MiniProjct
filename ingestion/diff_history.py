import json
from pathlib import Path

from git import Repo


REPO_PATH = "data/raw/test_repo"
OUTPUT_PATH = "data/processed/diff_history.json"

# For the first version, avoid storing extremely large patches.
MAX_PATCH_CHARACTERS = 20000


def decode_patch(diff_item):
    """
    GitPython gives patch data as bytes.
    Convert it safely into text.
    """

    try:
        return diff_item.diff.decode(
            "utf-8",
            errors="replace"
        )
    except Exception:
        return None


def get_change_type(diff_item):
    """
    Determine whether the file was:
    added, deleted, renamed, or modified.
    """

    if diff_item.new_file:
        return "added"

    if diff_item.deleted_file:
        return "deleted"

    if diff_item.renamed_file:
        return "renamed"

    return "modified"


def extract_added_removed_lines(patch):
    """
    Extract only actual added and removed source lines.

    Ignore diff metadata lines such as:
    +++ file.py
    --- file.py
    """

    added_lines = []
    removed_lines = []

    if not patch:
        return added_lines, removed_lines

    for line in patch.splitlines():

        if line.startswith("+++") or line.startswith("---"):
            continue

        if line.startswith("+"):
            added_lines.append(
                line[1:]
            )

        elif line.startswith("-"):
            removed_lines.append(
                line[1:]
            )

    return added_lines, removed_lines


def extract_diff_history(repo_path):
    repo = Repo(repo_path)

    records = []

    for commit_index, commit in enumerate(
        repo.iter_commits(),
        start=1
    ):

        # Root commits do not have a parent.
        if not commit.parents:
            continue

        # For now use the first parent.
        # This keeps merge commits manageable.
        parent = commit.parents[0]

        try:
            diffs = parent.diff(
                commit,
                create_patch=True
            )

        except Exception as error:
            print(
                f"Could not process commit "
                f"{commit.hexsha[:10]}: {error}"
            )
            continue

        for diff_item in diffs:

            old_path = diff_item.a_path
            new_path = diff_item.b_path

            file_path = (
                new_path
                or old_path
            )

            if not file_path:
                continue

            patch = decode_patch(
                diff_item
            )

            # Skip binary / unreadable patches.
            if patch is None:
                continue

            # Prevent a few giant files from making
            # the dataset unnecessarily large.
            patch_truncated = False

            if len(patch) > MAX_PATCH_CHARACTERS:

                patch = patch[
                    :MAX_PATCH_CHARACTERS
                ]

                patch_truncated = True

            added_lines, removed_lines = (
                extract_added_removed_lines(
                    patch
                )
            )

            record = {
                "commit_hash": commit.hexsha,

                "timestamp": (
                    commit
                    .committed_datetime
                    .isoformat()
                ),

                "author_name": (
                    commit.author.name
                ),

                "author_email": (
                    commit.author.email
                ),

                "message": (
                    commit.message.strip()
                ),

                "parent_hash": (
                    parent.hexsha
                ),

                "file_path": file_path,

                "old_path": old_path,

                "new_path": new_path,

                "change_type": (
                    get_change_type(
                        diff_item
                    )
                ),

                "patch": patch,

                "patch_truncated": (
                    patch_truncated
                ),

                "added_lines": (
                    added_lines
                ),

                "removed_lines": (
                    removed_lines
                ),

                "added_line_count": (
                    len(added_lines)
                ),

                "removed_line_count": (
                    len(removed_lines)
                ),
            }

            records.append(
                record
            )

        # Progress message every 500 commits
        if commit_index % 500 == 0:

            print(
                f"Processed "
                f"{commit_index} commits..."
            )

    return records


def save_diff_history(
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

    print(
        "\nExtracting Git diffs..."
    )

    records = extract_diff_history(
        REPO_PATH
    )

    save_diff_history(
        records,
        OUTPUT_PATH
    )

    print(
        "\nRepoMind Diff History"
    )

    print(
        "---------------------"
    )

    print(
        f"Diff records: "
        f"{len(records)}"
    )

    print(
        f"Saved to: "
        f"{OUTPUT_PATH}"
    )

    print(
        "\nFirst 5 diff records:"
    )

    for record in records[:5]:

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
            f"Type: "
            f"{record['change_type']}"
        )

        print(
            f"Added lines: "
            f"{record['added_line_count']}"
        )

        print(
            f"Removed lines: "
            f"{record['removed_line_count']}"
        )

        print(
            f"Message: "
            f"{record['message']}"
        )

        print(
            "\nPatch preview:"
        )

        print(
            record["patch"][:500]
        )