from pathlib import Path


SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".go": "go",
    ".rs": "rust",
}


IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "vendor",
    "build",
    "dist",
    "__pycache__",
}


def scan_repository(repo_path):
    repo_path = Path(repo_path)

    source_files = []

    for file_path in repo_path.rglob("*"):

        if not file_path.is_file():
            continue

        if any(
            ignored_dir in file_path.parts
            for ignored_dir in IGNORED_DIRECTORIES
        ):
            continue

        extension = file_path.suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            continue

        source_files.append(
            {
                "path": str(file_path),
                "relative_path": str(file_path.relative_to(repo_path)),
                "filename": file_path.name,
                "extension": extension,
                "language": SUPPORTED_EXTENSIONS[extension],
                "size_bytes": file_path.stat().st_size,
            }
        )

    return source_files


if __name__ == "__main__":
    files = scan_repository("data/raw/test_repo")

    print("\nRepoMind Repository Scanner")
    print("---------------------------")

    print(f"Total source files: {len(files)}")

    for file in files[:10]:
        print(file)

    language_counts = {}

    for file in files:
        language = file["language"]

        language_counts[language] = (
            language_counts.get(language, 0) + 1
        )

    print("\nLanguage distribution:")

    for language, count in language_counts.items():
        print(f"{language}: {count}")