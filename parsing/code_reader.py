def read_code_file(file_path):
    try:
        with open(
            file_path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as file:
            return file.read()

    except Exception as error:
        print(f"Could not read {file_path}")
        print(error)
        return None