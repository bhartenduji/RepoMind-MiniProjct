from pathlib import Path


def path_to_module(relative_path):
    """
    Convert a repository file path into a Python module path.

    Examples:

    src/flask/cli.py
        -> flask.cli

    src/flask/__init__.py
        -> flask

    tests/test_cli.py
        -> tests.test_cli
    """

    path = Path(relative_path)

    parts = list(path.parts)

    # Remove common source-root folders
    if parts and parts[0] in {"src", "lib"}:
        parts = parts[1:]

    if not parts:
        return None

    filename = parts[-1]

    # Package file
    if filename == "__init__.py":
        parts = parts[:-1]

    # Normal Python file
    elif filename.endswith(".py"):
        parts[-1] = filename[:-3]

    if not parts:
        return None

    return ".".join(parts)


def build_module_map(dataset):
    """
    Build a mapping:

        Python module name
            ->
        repository file path

    Example:

        flask.cli
            ->
        src/flask/cli.py
    """

    module_map = {}

    for file_record in dataset:

        if file_record.get("language") != "python":
            continue

        relative_path = file_record.get(
            "relative_path"
        )

        if not relative_path:
            continue

        module_name = path_to_module(
            relative_path
        )

        if module_name:
            module_map[module_name] = relative_path

    return module_map


def get_import_module(import_record):
    """
    Get the module part from the structured import.

    Example:

    {
        "type": "from",
        "module": "flask.cli",
        "name": "find_best_app",
        "alias": None,
        "level": 0
    }

    returns:

        flask.cli
    """

    if not isinstance(import_record, dict):
        return None

    return import_record.get("module")


def resolve_import(import_record, module_map):
    """
    Try to resolve an import to a file inside
    the repository.

    Returns:

        target repository file path

    or:

        None
    """

    if not isinstance(import_record, dict):
        return None

    import_type = import_record.get(
        "type"
    )

    module_name = import_record.get(
        "module"
    )

    imported_name = import_record.get(
        "name"
    )

    # ==========================================
    # Case 1:
    #
    # import flask.cli
    # ==========================================

    if import_type == "import":

        if not module_name:
            return None

        # Exact match
        if module_name in module_map:
            return module_map[module_name]

        return find_parent_module(
            module_name,
            module_map
        )

    # ==========================================
    # Case 2:
    #
    # from flask.cli import find_best_app
    # ==========================================

    if import_type == "from":

        if not module_name:
            return None

        # First check exact module
        if module_name in module_map:
            return module_map[module_name]

        # Sometimes imported name itself
        # represents a submodule:
        #
        # from flask import cli
        #
        # Try:
        #
        # flask.cli

        if imported_name:

            combined_module = (
                f"{module_name}."
                f"{imported_name}"
            )

            if combined_module in module_map:
                return module_map[
                    combined_module
                ]

        return find_parent_module(
            module_name,
            module_map
        )

    return None


def find_parent_module(
    module_name,
    module_map
):
    """
    Search progressively shorter module names.

    Example:

        flask.cli.helpers

    Try:

        flask.cli.helpers
        flask.cli
        flask
    """

    if not module_name:
        return None

    parts = module_name.split(".")

    while parts:

        candidate = ".".join(parts)

        if candidate in module_map:
            return module_map[candidate]

        parts = parts[:-1]

    return None