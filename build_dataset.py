import json
from pathlib import Path

from ingestion.file_scanner import scan_repository
from parsing.code_reader import read_code_file
from parsing.python_parser import parse_python_code


REPO_PATH = "data/raw/test_repo"
OUTPUT_PATH = "data/processed/repository_dataset.json"


def build_dataset(repo_path):
    files = scan_repository(repo_path)

    dataset = []

    for file in files:

        if file["language"] != "python":
            continue

        code = read_code_file(file["path"])

        if code is None:
            continue

        parsed = parse_python_code(code)

        if parsed is None:
            continue

        file_record = {
            "relative_path": file["relative_path"],
            "filename": file["filename"],
            "language": file["language"],
            "size_bytes": file["size_bytes"],
            "functions": parsed["functions"],
            "classes": parsed["classes"],
            "imports": parsed["imports"],
        }

        dataset.append(file_record)

    return dataset


if __name__ == "__main__":

    dataset = build_dataset(REPO_PATH)

    output_path = Path(OUTPUT_PATH)

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

    print("Dataset created successfully")
    print(f"Files stored: {len(dataset)}")
    print(f"Output: {OUTPUT_PATH}")