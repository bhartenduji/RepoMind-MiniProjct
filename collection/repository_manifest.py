import json
from datetime import datetime, timezone
from pathlib import Path


MANIFEST_PATH = Path(
    "data/processed/repository_manifest.json"
)


# =========================================================
# DEFAULT REPOSITORY RECORD
# =========================================================

def create_repository_record(
    repo_id,
    owner,
    name,
    clone_url,
    local_path,
    language="python",
    license_name=None,
    default_branch=None,
    stars=0,
    forks=0,
    open_issues=0,
    created_at=None,
    updated_at=None,
    pushed_at=None,
    is_fork=False,
    archived=False,
):
    """
    Create one normalized repository registry record.

    repo_id should be unique.

    Recommended format:

        owner/name
    """

    now = datetime.now(
        timezone.utc
    ).isoformat()

    return {
        "repo_id": repo_id,
        "owner": owner,
        "name": name,

        "clone_url": clone_url,
        "local_path": local_path,

        "language": language,
        "license": license_name,
        "default_branch": default_branch,

        "stars": stars,
        "forks": forks,
        "open_issues": open_issues,

        "created_at": created_at,
        "updated_at": updated_at,
        "pushed_at": pushed_at,

        "is_fork": is_fork,
        "archived": archived,

        # -----------------------------------------
        # Collection metadata
        # -----------------------------------------

        "collection_status": "pending",

        "collection_error": None,

        "commit_count": None,
        "source_file_count": None,
        "diff_record_count": None,
        "bug_fix_record_count": None,

        "last_collected_at": None,

        "added_to_manifest_at": now,

        "dataset_version": "v1",

        # -----------------------------------------
        # Quality metadata
        # -----------------------------------------

        "quality_status": "unknown",

        "quality_notes": [],

        # -----------------------------------------
        # Provenance
        # -----------------------------------------

        "provenance": {
            "source": "manual",
            "source_url": clone_url,
        },
    }


# =========================================================
# LOAD MANIFEST
# =========================================================

def load_manifest(
    path=MANIFEST_PATH
):
    """
    Load repository manifest.

    If it does not exist yet,
    return an empty list.
    """

    path = Path(
        path
    )

    if not path.exists():
        return []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(
            file
        )


# =========================================================
# SAVE MANIFEST
# =========================================================

def save_manifest(
    repositories,
    path=MANIFEST_PATH
):
    """
    Save repository manifest safely.
    """

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
            repositories,
            file,
            indent=2
        )


# =========================================================
# FIND REPOSITORY
# =========================================================

def find_repository(
    repositories,
    repo_id
):
    """
    Find repository using repo_id.
    """

    for repository in repositories:

        if repository.get(
            "repo_id"
        ) == repo_id:

            return repository

    return None


# =========================================================
# ADD REPOSITORY
# =========================================================

def add_repository(
    repository,
    path=MANIFEST_PATH
):
    """
    Add repository only if repo_id
    does not already exist.
    """

    repositories = load_manifest(
        path
    )

    repo_id = repository[
        "repo_id"
    ]

    existing = find_repository(
        repositories,
        repo_id
    )

    if existing is not None:

        print(
            f"Repository already exists: "
            f"{repo_id}"
        )

        return False

    repositories.append(
        repository
    )

    save_manifest(
        repositories,
        path
    )

    print(
        f"Added repository: "
        f"{repo_id}"
    )

    return True


# =========================================================
# UPDATE REPOSITORY
# =========================================================

def update_repository(
    repo_id,
    updates,
    path=MANIFEST_PATH
):
    """
    Update selected repository fields.

    Example:

    update_repository(
        "pallets/flask",
        {
            "commit_count": 5556,
            "collection_status": "complete",
        }
    )
    """

    repositories = load_manifest(
        path
    )

    repository = find_repository(
        repositories,
        repo_id
    )

    if repository is None:

        print(
            f"Repository not found: "
            f"{repo_id}"
        )

        return False

    for key, value in updates.items():

        repository[
            key
        ] = value

    save_manifest(
        repositories,
        path
    )

    return True


# =========================================================
# MARK COLLECTION STARTED
# =========================================================

def mark_collection_started(
    repo_id,
    path=MANIFEST_PATH
):
    return update_repository(
        repo_id,
        {
            "collection_status":
                "collecting",

            "collection_error":
                None,
        },
        path
    )


# =========================================================
# MARK COLLECTION COMPLETE
# =========================================================

def mark_collection_complete(
    repo_id,
    commit_count=None,
    source_file_count=None,
    diff_record_count=None,
    bug_fix_record_count=None,
    path=MANIFEST_PATH,
):
    now = datetime.now(
        timezone.utc
    ).isoformat()

    return update_repository(
        repo_id,
        {
            "collection_status":
                "complete",

            "collection_error":
                None,

            "commit_count":
                commit_count,

            "source_file_count":
                source_file_count,

            "diff_record_count":
                diff_record_count,

            "bug_fix_record_count":
                bug_fix_record_count,

            "last_collected_at":
                now,
        },
        path
    )


# =========================================================
# MARK COLLECTION FAILED
# =========================================================

def mark_collection_failed(
    repo_id,
    error_message,
    path=MANIFEST_PATH
):
    return update_repository(
        repo_id,
        {
            "collection_status":
                "failed",

            "collection_error":
                str(
                    error_message
                ),
        },
        path
    )


# =========================================================
# QUALITY STATUS
# =========================================================

def update_quality_status(
    repo_id,
    status,
    note=None,
    path=MANIFEST_PATH
):
    repositories = load_manifest(
        path
    )

    repository = find_repository(
        repositories,
        repo_id
    )

    if repository is None:
        return False

    repository[
        "quality_status"
    ] = status

    if note:

        notes = repository.get(
            "quality_notes",
            []
        )

        notes.append(
            note
        )

        repository[
            "quality_notes"
        ] = notes

    save_manifest(
        repositories,
        path
    )

    return True


# =========================================================
# SUMMARY
# =========================================================

def print_manifest_summary(
    repositories
):
    print(
        "\nRepoMind Repository Manifest"
    )

    print(
        "----------------------------"
    )

    print(
        f"Total repositories: "
        f"{len(repositories)}"
    )

    status_counts = {}

    quality_counts = {}

    for repository in repositories:

        collection_status = repository.get(
            "collection_status",
            "unknown"
        )

        status_counts[
            collection_status
        ] = (
            status_counts.get(
                collection_status,
                0
            )
            +
            1
        )

        quality_status = repository.get(
            "quality_status",
            "unknown"
        )

        quality_counts[
            quality_status
        ] = (
            quality_counts.get(
                quality_status,
                0
            )
            +
            1
        )

    print(
        "\nCollection Status"
    )

    for status, count in status_counts.items():

        print(
            f"{status}: "
            f"{count}"
        )

    print(
        "\nQuality Status"
    )

    for status, count in quality_counts.items():

        print(
            f"{status}: "
            f"{count}"
        )


# =========================================================
# TEST / INITIAL ENTRY
# =========================================================

if __name__ == "__main__":

    flask_record = create_repository_record(

        repo_id="pallets/flask",

        owner="pallets",

        name="flask",

        clone_url=(
            "https://github.com/"
            "pallets/flask.git"
        ),

        local_path=(
            "data/raw/test_repo"
        ),

        language="python",

        license_name="BSD-3-Clause",

        default_branch="main",
    )

    flask_record[
        "provenance"
    ] = {
        "source":
            "manual",

        "source_url":
            "https://github.com/pallets/flask"
    }

    add_repository(
        flask_record
    )

    repositories = load_manifest()

    print_manifest_summary(
        repositories
    )