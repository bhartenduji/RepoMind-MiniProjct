from ingestion.file_scanner import scan_repository
from parsing.code_reader import read_code_file
from parsing.python_parser import parse_python_code


REPO_PATH = "data/raw/test_repo"


def analyze_repository(repo_path):
    files = scan_repository(repo_path)

    total_functions = 0
    total_classes = 0
    total_imports = 0
    parsed_files = 0
    failed_files = 0

    for file in files:

        if file["language"] != "python":
            continue

        code = read_code_file(file["path"])

        if code is None:
            failed_files += 1
            continue

        result = parse_python_code(code)

        if result is None:
            failed_files += 1
            continue

        parsed_files += 1

        total_functions += len(result["functions"])
        total_classes += len(result["classes"])
        total_imports += len(result["imports"])

    print("\nRepoMind Repository Analysis")
    print("----------------------------")

    print(f"Source files found: {len(files)}")
    print(f"Python files parsed: {parsed_files}")
    print(f"Failed files: {failed_files}")

    print("\nCode Structure")
    print("--------------")

    print(f"Functions found: {total_functions}")
    print(f"Classes found: {total_classes}")
    print(f"Imports found: {total_imports}")


if __name__ == "__main__":
    analyze_repository(REPO_PATH)