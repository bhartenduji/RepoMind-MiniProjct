import ast


def get_function_parameters(node):
    parameters = []

    for arg in node.args.args:
        parameters.append(arg.arg)

    return parameters


def get_decorators(node):
    decorators = []

    for decorator in node.decorator_list:
        try:
            decorators.append(ast.unparse(decorator))
        except Exception:
            decorators.append("unknown")

    return decorators


def extract_source_segment(code, node):
    try:
        return ast.get_source_segment(code, node)
    except Exception:
        return None


def get_function_calls(node):
    calls = []

    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func

            if isinstance(func, ast.Name):
                calls.append(func.id)

            elif isinstance(func, ast.Attribute):
                try:
                    calls.append(ast.unparse(func))
                except Exception:
                    calls.append(func.attr)

    return calls


def parse_python_code(code):
    try:
        tree = ast.parse(code)

    except SyntaxError:
        return None

    functions = []
    classes = []
    imports = []

    def walk_node(node, parent_class=None):

        # =================================================
        # CLASS
        # =================================================

        if isinstance(node, ast.ClassDef):

            class_record = {
                "name": node.name,
                "line_start": node.lineno,
                "line_end": getattr(
                    node,
                    "end_lineno",
                    None
                ),
                "docstring": ast.get_docstring(node),
                "decorators": get_decorators(node),
            }

            classes.append(class_record)

            for child in node.body:
                walk_node(
                    child,
                    parent_class=node.name
                )

            return

        # =================================================
        # FUNCTION / METHOD
        # =================================================

        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef
            )
        ):

            qualified_name = (
                f"{parent_class}.{node.name}"
                if parent_class
                else node.name
            )

            function_record = {
                "name": node.name,
                "qualified_name": qualified_name,
                "parent_class": parent_class,
                "function_type": (
                    "method"
                    if parent_class
                    else "function"
                ),
                "line_start": node.lineno,
                "line_end": getattr(
                    node,
                    "end_lineno",
                    None
                ),
                "parameters": (
                    get_function_parameters(node)
                ),
                "is_async": isinstance(
                    node,
                    ast.AsyncFunctionDef
                ),
                "decorators": (
                    get_decorators(node)
                ),
                "docstring": (
                    ast.get_docstring(node)
                ),
                "source_code": (
                    extract_source_segment(
                        code,
                        node
                    )
                ),
                "calls": (
                    get_function_calls(node)
                ),
            }

            functions.append(
                function_record
            )

        # =================================================
        # IMPORT
        #
        # Example:
        #
        # import os
        # import flask.cli as cli
        # =================================================

        elif isinstance(
            node,
            ast.Import
        ):

            for item in node.names:

                imports.append(
                    {
                        "type": "import",
                        "module": item.name,
                        "name": None,
                        "alias": item.asname,
                        "level": 0,
                    }
                )

        # =================================================
        # FROM IMPORT
        #
        # Example:
        #
        # from flask.cli import find_best_app
        #
        # from .app import Flask
        # =================================================

        elif isinstance(
            node,
            ast.ImportFrom
        ):

            for item in node.names:

                imports.append(
                    {
                        "type": "from",
                        "module": node.module,
                        "name": item.name,
                        "alias": item.asname,
                        "level": node.level,
                    }
                )

        # =================================================
        # CONTINUE WALKING AST
        # =================================================

        for child in ast.iter_child_nodes(
            node
        ):
            walk_node(
                child,
                parent_class
            )

    walk_node(tree)

    return {
        "functions": functions,
        "classes": classes,
        "imports": imports,
    }