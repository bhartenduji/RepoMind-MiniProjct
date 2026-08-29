import json
import networkx as nx
import matplotlib.pyplot as plt

from graph.repository_graph import build_repository_graph


DATASET_PATH = "data/processed/repository_dataset.json"


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_small_subgraph(graph, center_node, depth=1):
    nodes = {center_node}

    current = {center_node}

    for _ in range(depth):
        next_nodes = set()

        for node in current:
            next_nodes.update(graph.successors(node))
            next_nodes.update(graph.predecessors(node))

        nodes.update(next_nodes)
        current = next_nodes

    return graph.subgraph(nodes).copy()


if __name__ == "__main__":

    dataset = load_dataset(DATASET_PATH)

    graph, _ = build_repository_graph(dataset)

    # Pick one function containing this text
    search_text = "find_best_app"

    matching_nodes = [
        node
        for node, data in graph.nodes(data=True)
        if search_text in str(data.get("qualified_name", ""))
    ]

    if not matching_nodes:
        print("No matching node found")
        exit()

    center_node = matching_nodes[0]

    print("Center node:")
    print(center_node)

    subgraph = get_small_subgraph(
        graph,
        center_node,
        depth=1
    )

    print(
        f"Subgraph nodes: {subgraph.number_of_nodes()}"
    )

    print(
        f"Subgraph edges: {subgraph.number_of_edges()}"
    )

    labels = {}

    for node, data in subgraph.nodes(data=True):
        labels[node] = (
            data.get("qualified_name")
            or data.get("name")
            or data.get("path")
            or node
        )

    pos = nx.spring_layout(
        subgraph,
        seed=42
    )

    nx.draw(
        subgraph,
        pos,
        with_labels=False,
        node_size=1000,
        arrows=True
    )

    nx.draw_networkx_labels(
        subgraph,
        pos,
        labels=labels,
        font_size=8
    )

    plt.title(
        "RepoMind Repository Subgraph"
    )

    plt.tight_layout()

    plt.savefig(
        "data/processed/repository_subgraph.png",
        dpi=200
    )

    print(
        "Saved visualization to "
        "data/processed/repository_subgraph.png"
    )