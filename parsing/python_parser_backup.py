import ast


def parse_python_code(code):
    try:
        tree = ast.parse(code)

    except SyntaxError:
        return None

    functions = []
    classes = []
    imports = []

    for node in ast.walk(tree):

        if isinstance(node, ast.FunctionDef):
            functions.append(
                {
                    "name": node.name,
                    "line_start": node.lineno,
                    "line_end": getattr(node, "end_lineno", None),
                }
            )

        elif isinstance(node, ast.AsyncFunctionDef):
            functions.append(
                {
                    "name": node.name,
                    "line_start": node.lineno,
                    "line_end": getattr(node, "end_lineno", None),
                }
            )

        elif isinstance(node, ast.ClassDef):
            classes.append(
                {
                    "name": node.name,
                    "line_start": node.lineno,
                    "line_end": getattr(node, "end_lineno", None),
                }
            )

        elif isinstance(node, ast.Import):
            for item in node.names:
                imports.append(item.name)

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)

    return {
        "functions": functions,
        "classes": classes,
        "imports": imports,
    }