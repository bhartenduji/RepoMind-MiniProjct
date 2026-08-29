from graph.import_resolver import path_to_module


examples = [
    "src/flask/cli.py",
    "src/flask/app.py",
    "src/flask/__init__.py",
    "tests/test_cli.py",
]


for example in examples:
    print(
        example,
        "->",
        path_to_module(example)
    )