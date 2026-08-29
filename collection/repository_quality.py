from pathlib import Path

from collection.repository_manifest import (
    load_manifest,
    update_quality_status,
)


MANIFEST_PATH = (
    "data/processed/"
    "repository_manifest.json"
)


# =========================================================
# QUALITY THRESHOLDS
# =========================================================

MIN_COMMITS = 100

MIN_PYTHON_FILES = 20

APPROVED_LICENSES = {
    "MIT",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
}


# =========================================================
# SCORE ONE REPOSITORY
# =========================================================

def evaluate_repository(repository):
    """
    Evaluate one repository using simple,
    explainable quality rules.

    Returns:
        status
        score
        notes
    """

    score = 0
    notes = []

    repo_id = repository.get(
        "repo_id",
        "unknown"
    )

    # -----------------------------------------------------
    # 1. Collection status
    # -----------------------------------------------------

    collection_status = repository.get(
        "collection_status"
    )

    if collection_status != "complete":

        notes.append(
            "repository collection is not complete"
        )

        return (
            "rejected",
            0,
            notes
        )

    score += 1

    # -----------------------------------------------------
    # 2. Fork check
    # -----------------------------------------------------

    is_fork = repository.get(
        "is_fork",
        False
    )

    if is_fork:

        notes.append(
            "repository is a fork"
        )

    else:

        score += 1

    # -----------------------------------------------------
    # 3. Archived check
    # -----------------------------------------------------

    archived = repository.get(
        "archived",
        False
    )

    if archived:

        notes.append(
            "repository is archived"
        )

    else:

        score += 1

    # -----------------------------------------------------
    # 4. Commit history
    # -----------------------------------------------------

    commit_count = repository.get(
        "commit_count"
    )

    if commit_count is None:

        notes.append(
            "commit count unavailable"
        )

    elif commit_count < MIN_COMMITS:

        notes.append(
            f"too few commits: "
            f"{commit_count}"
        )

    else:

        score += 2

    # -----------------------------------------------------
    # 5. Python source files
    # -----------------------------------------------------

    source_file_count = repository.get(
        "source_file_count"
    )

    if source_file_count is None:

        notes.append(
            "source file count unavailable"
        )

    elif source_file_count < MIN_PYTHON_FILES:

        notes.append(
            f"too few Python files: "
            f"{source_file_count}"
        )

    else:

        score += 2

    # -----------------------------------------------------
    # 6. Language check
    # -----------------------------------------------------

    language = repository.get(
        "language"
    )

    if (
        language
        and
        language.lower()
        ==
        "python"
    ):

        score += 1

    else:

        notes.append(
            f"unexpected language: "
            f"{language}"
        )

    # -----------------------------------------------------
    # 7. License check
    # -----------------------------------------------------

    license_name = repository.get(
        "license"
    )

    if license_name in APPROVED_LICENSES:

        score += 2

    else:

        notes.append(
            f"license not approved or unknown: "
            f"{license_name}"
        )

    # -----------------------------------------------------
    # FINAL DECISION
    # -----------------------------------------------------

    if (
        is_fork
        or
        archived
        or
        commit_count is None
        or
        commit_count < MIN_COMMITS
        or
        source_file_count is None
        or
        source_file_count < MIN_PYTHON_FILES
    ):

        status = "rejected"

    elif score >= 9:

        status = "high_quality"

    elif score >= 7:

        status = "accepted"

    else:

        status = "review"

    if not notes:

        notes.append(
            "repository passed all quality checks"
        )

    print(
        f"\nRepository: {repo_id}"
    )

    print(
        f"Quality score: {score}/10"
    )

    print(
        f"Quality status: {status}"
    )

    for note in notes:

        print(
            f"  - {note}"
        )

    return (
        status,
        score,
        notes
    )


# =========================================================
# EVALUATE ALL REPOSITORIES
# =========================================================

def evaluate_all_repositories():

    repositories = load_manifest(
        MANIFEST_PATH
    )

    if not repositories:

        print(
            "Repository manifest is empty."
        )

        return

    print(
        "\nRepoMind Repository Quality"
    )

    print(
        "---------------------------"
    )

    print(
        f"Repositories: "
        f"{len(repositories)}"
    )

    accepted = 0
    rejected = 0
    review = 0

    for repository in repositories:

        (
            status,
            score,
            notes
        ) = evaluate_repository(
            repository
        )

        repo_id = repository[
            "repo_id"
        ]

        update_quality_status(
            repo_id,
            status,
            note=(
                f"quality_score={score}; "
                +
                "; ".join(notes)
            )
        )

        if status in {
            "accepted",
            "high_quality"
        }:

            accepted += 1

        elif status == "rejected":

            rejected += 1

        else:

            review += 1

    print(
        "\n=================================="
    )

    print(
        "Quality Summary"
    )

    print(
        "=================================="
    )

    print(
        f"Accepted/high quality: "
        f"{accepted}"
    )

    print(
        f"Review: "
        f"{review}"
    )

    print(
        f"Rejected: "
        f"{rejected}"
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    evaluate_all_repositories()