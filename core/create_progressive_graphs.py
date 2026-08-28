import fire
import networkx as nx
from simulate_data_parallel_hic_self_seed_partial_final_with_cascades import _make_or_load_graph
from pathlib import Path
import pickle
import os

def main(
    graph_name="tree", node_count=100, edges_per_step=10,
    connections_per_node=3, graph_path="./data/graphs/graph_tree.pkl", graph_folder="./data/graphs",
    data_seed=2605, force_graph=False, subfolder="progressive"
):

    os.makedirs(f"{graph_folder}/{subfolder}", exist_ok=True)

    base_graph: nx.Graph = _make_or_load_graph(
        graph_name=graph_name,
        node_count=node_count,
        connections_per_node=connections_per_node,
        graph_path=Path(graph_path),
        graph_folder=Path(graph_folder),
        data_seed=data_seed,
        force_graph=force_graph,
    )

    base_graph_path = f"{graph_folder}/{subfolder}_{graph_name}_base.pkl"
    pickle.dump(base_graph, open(base_graph_path, "wb"))

    max_edge_count = base_graph.number_of_nodes() * (base_graph.number_of_nodes() - 1) // 2
    edge_count = base_graph.number_of_edges()

    print(f"Base graph has {edge_count} edges and {base_graph.number_of_nodes()} nodes. Max edge count is {max_edge_count}. Adding edges in steps of {edges_per_step}.")

    for i in range(0, max_edge_count - edge_count, edges_per_step):
        new_graph = base_graph.copy()
        new_graph.add_edges_from([(i, j) for i in range(new_graph.number_of_nodes()) for j in range(i + 1, new_graph.number_of_nodes()) if not new_graph.has_edge(i, j)][:i])
        new_graph_path = f"{graph_folder}/{subfolder}/{graph_name}_progressive_{i}.pkl"
        pickle.dump(new_graph, open(new_graph_path, "wb"))
        print(f"Saved progressive graph with {new_graph.number_of_edges()} edges to {new_graph_path}.",end="\r")


    print()
    print("Done")

if __name__ == "__main__":
    fire.Fire(main)