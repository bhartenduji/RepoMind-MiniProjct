import json
import networkx as nx


DATASET_PATH = "data/processed/repository_dataset.json"


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_call(call_name, parent_class=None):
    """
    Convert calls like:
        self.create_session
    into:
        UserService.create_session
    when parent_class is UserService.
    """

    if call_name.startswith("self.") and parent_class:
        method_name = call_name.split(".", 1)[1]
        return f"{parent_class}.{method_name}"

    return call_name


def build_function_id(file_record, function):
    return (
        f"{file_record['relative_path']}::"
        f"{function['qualified_name']}::"
        f"L{function['line_start']}"
    )

def build_call_graph(dataset):
    graph = nx.DiGraph()

    # Maps:
    # qualified_name -> [unique function IDs]
    #
    # Example:
    # create_app ->
    # [
    #   "src/app.py::create_app",
    #   "tests/test_app.py::create_app"
    # ]
    known_functions = {}

    # ----------------------------------
    # FIRST PASS
    # Create all function nodes
    # ----------------------------------

    for file_record in dataset:

        for function in file_record["functions"]:

            function_id = build_function_id(
                file_record,
                function
            )

            qualified_name = function["qualified_name"]

            graph.add_node(
                function_id,
                name=function["name"],
                qualified_name=qualified_name,
                file=file_record["relative_path"],
                function_type=function["function_type"],
                parent_class=function["parent_class"],
            )

            known_functions.setdefault(
                qualified_name,
                []
            ).append(function_id)

    # ----------------------------------
    # SECOND PASS
    # Create CALLS relationships
    # ----------------------------------

    unresolved_calls = 0
    ambiguous_calls = 0
    resolved_calls = 0

    for file_record in dataset:

        for function in file_record["functions"]:

            caller_id = build_function_id(
                file_record,
                function
            )

            calls = function.get("calls", [])

            for call in calls:

                target_name = normalize_call(
                    call,
                    function["parent_class"]
                )

                candidates = known_functions.get(
                    target_name,
                    []
                )

                # Exactly one matching function
                if len(candidates) == 1:

                    target_id = candidates[0]

                    graph.add_edge(
                        caller_id,
                        target_id,
                        relation="CALLS",
                    )

                    resolved_calls += 1

                # More than one possible function
                elif len(candidates) > 1:

                    ambiguous_calls += 1

                # No matching function found
                else:

                    unresolved_calls += 1

    stats = {
        "resolved_calls": resolved_calls,
        "ambiguous_calls": ambiguous_calls,
        "unresolved_calls": unresolved_calls,
    }

    return graph, stats


if __name__ == "__main__":

    dataset = load_dataset(DATASET_PATH)

    graph, stats = build_call_graph(dataset)

    print("\nRepoMind Call Graph")
    print("-------------------")

    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")

    print("\nCall Resolution")
    print("-------------------")

    print(
        f"Resolved calls: {stats['resolved_calls']}"
    )

    print(
        f"Ambiguous calls: {stats['ambiguous_calls']}"
    )

    print(
        f"Unresolved calls: {stats['unresolved_calls']}"
    )

    print("\nExample call relationships:")
    print("---------------------------")

    for source, target in list(graph.edges())[:20]:

        source_data = graph.nodes[source]
        target_data = graph.nodes[target]

        print(
            f"{source_data['qualified_name']}"
            f" [{source_data['file']}]"
            " -> "
            f"{target_data['qualified_name']}"
            f" [{target_data['file']}]"
        )