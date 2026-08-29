import json
from pathlib import Path

from git import Repo


REPO_PATH = "data/raw/test_repo"
OUTPUT_PATH = "data/processed/git_history.json"


def extract_git_history(repo_path):
    repo = Repo(repo_path)

    commits_data = []

    for commit in repo.iter_commits():

        stats = commit.stats.total

        commit_record = {
            "commit_hash": commit.hexsha,
            "author_name": commit.author.name,
            "author_email": commit.author.email,
            "timestamp": commit.committed_datetime.isoformat(),
            "message": commit.message.strip(),
            "parent_count": len(commit.parents),
            "files_changed": stats.get("files", 0),
            "insertions": stats.get("insertions", 0),
            "deletions": stats.get("deletions", 0),
        }

        commits_data.append(commit_record)

    return commits_data


def save_git_history(commits, output_path):

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
            commits,
            file,
            indent=2
        )


if __name__ == "__main__":

    commits = extract_git_history(
        REPO_PATH
    )

    save_git_history(
        commits,
        OUTPUT_PATH
    )

    print("\nRepoMind Git History")
    print("--------------------")

    print(
        f"Total commits: "
        f"{len(commits)}"
    )

    print(
        f"Saved to: "
        f"{OUTPUT_PATH}"
    )

    print("\nFirst 5 commits:")

    for commit in commits[:5]:

        print("-" * 60)

        print(
            f"Hash: "
            f"{commit['commit_hash'][:10]}"
        )

        print(
            f"Author: "
            f"{commit['author_name']}"
        )

        print(
            f"Timestamp: "
            f"{commit['timestamp']}"
        )

        print(
            f"Message: "
            f"{commit['message']}"
        )

        print(
            f"Files changed: "
            f"{commit['files_changed']}"
        )

        print(
            f"Insertions: "
            f"{commit['insertions']}"
        )

        print(
            f"Deletions: "
            f"{commit['deletions']}"
        )