from pathlib import Path

from git import Repo
from git.exc import GitCommandError

from collection.repository_manifest import (
    load_manifest,
    update_repository,
    mark_collection_started,
    mark_collection_complete,
    mark_collection_failed,
)


MANIFEST_PATH = (
    "data/processed/"
    "repository_manifest.json"
)


# =========================================================
# BASIC HELPERS
# =========================================================

def repository_exists(
    local_path
):
    """
    Return True if local_path already
    contains a Git repository.
    """

    local_path = Path(
        local_path
    )

    git_dir = (
        local_path
        /
        ".git"
    )

    return (
        local_path.exists()
        and
        git_dir.exists()
    )


def count_commits(
    repo
):
    """
    Count commits in repository history.
    """

    try:

        return sum(
            1
            for _ in repo.iter_commits()
        )

    except Exception:

        return 0


def count_python_files(
    local_path
):
    """
    Count Python source files.

    Ignore the .git directory.
    """

    local_path = Path(
        local_path
    )

    count = 0

    for path in local_path.rglob(
        "*.py"
    ):

        if ".git" in path.parts:
            continue

        count += 1

    return count


# =========================================================
# CLONE REPOSITORY
# =========================================================

def clone_repository(
    clone_url,
    local_path
):
    """
    Clone repository into local_path.
    """

    local_path = Path(
        local_path
    )

    local_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"Cloning:"
        f"\n  {clone_url}"
        f"\ninto:"
        f"\n  {local_path}"
    )

    repo = Repo.clone_from(
        clone_url,
        local_path
    )

    return repo


# =========================================================
# UPDATE EXISTING REPOSITORY
# =========================================================

def update_existing_repository(
    local_path
):
    """
    Pull newest changes from origin.
    """

    repo = Repo(
        local_path
    )

    if not repo.remotes:

        print(
            "Repository has no remote."
        )

        return repo

    origin = repo.remotes.origin

    print(
        "Fetching remote changes..."
    )

    origin.fetch()

    active_branch = None

    try:

        active_branch = (
            repo.active_branch.name
        )

    except Exception:

        pass

    if active_branch:

        print(
            f"Pulling branch: "
            f"{active_branch}"
        )

        try:

            origin.pull(
                active_branch
            )

        except GitCommandError as error:

            print(
                "Pull failed."
            )

            print(
                error
            )

            print(
                "Continuing with fetched "
                "local repository."
            )

    return repo


# =========================================================
# COLLECT ONE REPOSITORY
# =========================================================

def collect_repository(
    repository_record
):
    """
    Clone or update one repository.

    Then calculate simple collection stats.
    """

    repo_id = repository_record[
        "repo_id"
    ]

    clone_url = repository_record[
        "clone_url"
    ]

    local_path = repository_record[
        "local_path"
    ]

    print(
        "\n=================================="
    )

    print(
        f"Repository: "
        f"{repo_id}"
    )

    print(
        "=================================="
    )

    mark_collection_started(
        repo_id
    )

    try:

        # -----------------------------------------
        # Clone or update
        # -----------------------------------------

        if repository_exists(
            local_path
        ):

            print(
                "Repository already exists."
            )

            print(
                "Updating repository..."
            )

            repo = (
                update_existing_repository(
                    local_path
                )
            )

        else:

            repo = clone_repository(
                clone_url,
                local_path
            )

        # -----------------------------------------
        # Basic validation
        # -----------------------------------------

        if repo.bare:

            raise RuntimeError(
                "Repository is bare."
            )

        # -----------------------------------------
        # Stats
        # -----------------------------------------

        print(
            "Counting commits..."
        )

        commit_count = count_commits(
            repo
        )

        print(
            "Counting Python files..."
        )

        python_file_count = (
            count_python_files(
                local_path
            )
        )

        # -----------------------------------------
        # Complete status
        # -----------------------------------------

        mark_collection_complete(

            repo_id,

            commit_count=
                commit_count,

            source_file_count=
                python_file_count,
        )

        print(
            "\nCollection complete."
        )

        print(
            f"Commits: "
            f"{commit_count}"
        )

        print(
            f"Python files: "
            f"{python_file_count}"
        )

        return True

    except Exception as error:

        print(
            "\nCollection failed."
        )

        print(
            error
        )

        mark_collection_failed(
            repo_id,
            str(error)
        )

        return False


# =========================================================
# COLLECT ALL REPOSITORIES
# =========================================================

def collect_all_repositories():
    """
    Process every repository stored
    in the manifest.
    """

    repositories = load_manifest(
        MANIFEST_PATH
    )

    if not repositories:

        print(
            "Repository manifest is empty."
        )

        return

    print(
        "\nRepoMind Repository Collector"
    )

    print(
        "-----------------------------"
    )

    print(
        f"Repositories in manifest: "
        f"{len(repositories)}"
    )

    successful = 0

    failed = 0

    for repository in repositories:

        result = collect_repository(
            repository
        )

        if result:

            successful += 1

        else:

            failed += 1

    print(
        "\n=================================="
    )

    print(
        "Collection Summary"
    )

    print(
        "=================================="
    )

    print(
        f"Successful: "
        f"{successful}"
    )

    print(
        f"Failed: "
        f"{failed}"
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    collect_all_repositories()