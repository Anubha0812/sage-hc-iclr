from main_hybrid_split_v7_sensitivity_nested_fpa import main as main_train
from simulate_data_parallel_hic_self_seed_partial_final_with_cascades import main as main_simulate
from fire import Fire
import pickle
from argparse import Namespace
from pathlib import Path
import random
import multiprocessing
import itertools


def combine_args(arg_dict):
    """
    Given a dictionary where each value is a list of options, return a list of dictionaries
    with all possible combinations of those options.
    """
    keys = list(arg_dict.keys())
    values_product = itertools.product(*(arg_dict[k] for k in keys))
    return [dict(zip(keys, vals)) for vals in values_product]


def run_experiments(
    experiments, devices, processes_per_device, debug=False, call_function=main_train
):
    print(f"Starting {len(experiments)} experiments")

    experiments_per_device = [0 for _ in devices]

    experiments_to_assign = len(experiments)

    for i in range(len(experiments_per_device)):
        experiments_per_device[i] = min(experiments_to_assign, processes_per_device)
        experiments_to_assign -= experiments_per_device[i]

    while experiments_to_assign > 0:
        for i in range(len(experiments_per_device)):
            experiments_per_device[i] += 1
            experiments_to_assign -= 1

            if experiments_to_assign <= 0:
                break

    pools = [
        multiprocessing.get_context("spawn").Pool(processes_per_device) for _ in devices
    ]

    for exp_count, device, pool in zip(experiments_per_device, devices, pools):

        for i in range(exp_count):
            experiment_args, experiment_kwargs = experiments.pop(0)

            if debug:
                print("DEBUG")
                print("DEBUG")
                print("DEBUG")
                print(f"{i} Running experiment with args: {experiment_args} and kwargs: {experiment_kwargs} on device: {device}")
                call_function(
                    *experiment_args,
                    device=device,
                    **experiment_kwargs,
                )
            else:
                # pool.apply(
                pool.apply_async(
                    call_function,
                    args=(*experiment_args,),
                    kwds={"device": device, **experiment_kwargs},
                    error_callback=lambda e, args=experiment_args, kwargs=experiment_kwargs: open(
                        "./errors.log", "a"
                    ).write(
                        f"Error: {str(e)}\nArgs: {args}\nKwargs: {kwargs}\n"
                    ),
                )

    for pool in pools:
        pool.close()
        pool.join()

    print("Done")


def simulate(
    name,
    base_graph_path = "./data/graphs/progressive/tree_progressive_0.pkl",
    edges_per_step=100,
    graph_name="tree",
    graph_folder="./data/graphs",
    features_parent_folder="./data/features",
    subfolder="progressive",
    save_cascades=False,
    **simulation_params,
):
    base_graph = pickle.load(open(base_graph_path, "rb"))
    max_edge_count = base_graph.number_of_nodes() * (base_graph.number_of_nodes() - 1) // 2
    edge_count = base_graph.number_of_edges()
    
    print(f"Base graph has {edge_count} edges and {base_graph.number_of_nodes()} nodes. Max edge count is {max_edge_count}. Adding edges in steps of {edges_per_step}.")

    for i in range(0, max_edge_count - edge_count, edges_per_step):
        current_graph_path = f"{graph_folder}/{subfolder}/{graph_name}_progressive_{i}.pkl"
        print(f"Simulating for progressive graph with {i} edges: {current_graph_path}")
        features_folder = Path(f"{features_parent_folder}/{subfolder}/{graph_name}_progressive_{i}")
        params = Namespace(
            name=name + f"_progressive_{i}", graph_path=current_graph_path, graph_name=graph_name, features_folder=features_folder,
            graph_folder=Path(graph_folder),
            self_test=False, create_external_set=False,
            force_graph=False,
            num_shards=1, shard_id=0,
            feature_batch_size=0,
            consolidate_only=False,
            no_consolidate=False,
            validate_output=False,
            force=False,
            start_method="fork",
            cutoff=None,
            save_cascades=save_cascades,
            **simulation_params)
        main_simulate(args=params)



def train(
    name,
    base_graph_path = "./data/graphs/progressive/tree_progressive_0.pkl",
    edges_per_step=100,
    graph_name="tree",
    graph_folder="./data/graphs",
    features_parent_folder="./data/features",
    subfolder="progressive",
    devices=8,
    processes_per_device=2,
    debug=False,
    **training_params
):

    if not isinstance(devices, list):
        devices = [f"cuda:{d}" for d in range(devices)]

    base_graph = pickle.load(open(base_graph_path, "rb"))
    max_edge_count = base_graph.number_of_nodes() * (base_graph.number_of_nodes() - 1) // 2
    edge_count = base_graph.number_of_edges()
    
    print(f"Base graph has {edge_count} edges and {base_graph.number_of_nodes()} nodes. Max edge count is {max_edge_count}. Adding edges in steps of {edges_per_step}.")

    experiments = []

    for i in range(0, max_edge_count - edge_count, edges_per_step):
        current_graph_path = f"{graph_folder}/{subfolder}/{graph_name}_progressive_{i}.pkl"
        print(f"Training on progressive graph with {i} edges: {current_graph_path}")
        features_folder = f"{features_parent_folder}/{subfolder}/{graph_name}_progressive_{i}"

        experiments.append((
            [],
            {
                "name": name + f"_progressive_{i}",
                "graph_path": current_graph_path,
                "graph_name": graph_name,
                "features_folder": features_folder,
                **training_params
            }
        ))

    run_experiments(experiments, devices=devices, processes_per_device=processes_per_device, debug=debug, call_function=main_train)

if __name__ == "__main__":
    Fire()