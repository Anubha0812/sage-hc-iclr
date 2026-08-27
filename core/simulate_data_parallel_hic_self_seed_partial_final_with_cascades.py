#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import shortest_path


FEATURE_NAMES_7 = (
    "weighted_positive_ratio",
    "weighted_negative_ratio",
    "weighted_zero_ratio",
    "signal_entropy_bits",
    "minimum_nonzero_seed_distance",
    "mean_seed_distance",
    "node_degree",
)
FEATURE_NAMES_9 = FEATURE_NAMES_7 + (
    "mean_positive_out_neighbor_fraction",
    "mean_positive_out_neighbor_fraction_given_node_positive",
)

# These four features preserve the seed-specific information that is lost by
# the ordinary cross-seed aggregation.  The graph may be stored as a symmetric
# DiGraph internally, but these are ordinary undirected-neighbour statistics
# when the original graph is undirected.
FEATURE_NAMES_13 = FEATURE_NAMES_9 + (
    "self_seed_neighbor_positive_mean",
    "self_seed_neighbor_positive_variance",
    "self_seed_reach_mean",
    "self_seed_reach_variance",
)

# Partial seed selection needs an explicit availability mask. For nodes that
# are not selected as seeds, the four self-seed statistics remain zero and
# this indicator is zero. For selected seeds, it is one.
FEATURE_NAMES_14 = FEATURE_NAMES_13 + (
    "self_seed_observed",
)

SUPPORTED_GENERATORS = {
    "single",
    "multiple",
    "single_new_features",
    "multiple_new_features",
    "multiple_new_features_self_seed",
    "multiple_new_features_self_seed_masked",
}

# On Linux/HPC, workers are forked after this object is set.  The large arrays
# are therefore shared copy-on-write instead of being serialized per task.
_WORKER_CONTEXT: Optional["WorkerContext"] = None


@dataclass
class OutputPaths:
    graph_path: Path
    features_path: Path
    indices_path: Path
    seeds_path: Path
    feature_shard_dir: Path
    cascade_shard_dir: Path
    manifest_path: Path


@dataclass
class WorkerContext:
    graph: nx.DiGraph
    nodes: Tuple[int, ...]
    node_to_index: Dict[int, int]
    selected_seeds: Tuple[int, ...]
    seed_indices: np.ndarray
    distance_matrix: np.ndarray
    min_nonzero_distance: np.ndarray
    mean_distance: np.ndarray
    degree: np.ndarray
    adjacency: sparse.csr_matrix
    inverse_out_degree: np.ndarray
    number_of_assignments: int
    num_cascades: int
    features_per_assignment: int
    assignment_generator_type: str
    alpha: float
    p1: float
    p2: float
    q_val: float
    data_seed: int
    feature_batch_size: int
    feature_shard_dir: Path
    cascade_shard_dir: Path
    save_cascades: bool
    force: bool


def _format_probability(value: float) -> str:
    return str(value).replace(".", "")


def _dataset_stem(
    graph_name: str,
    assignment_generator_type: str,
    node_count: int,
    connections_per_node: int,
    number_of_assignments: int,
    num_cascades: int,
    alpha: float,
    p1: float,
    p2: float,
    q_val: float,
    features_per_assignment: int,
    seed_percentage: float,
) -> str:
    # Kept byte-for-byte compatible with the naming logic in simulations.py.
    suffix = (
        f"_seedpercentage_{seed_percentage}" if seed_percentage < 1.0 else ""
    )
    return (
        f"graph_{graph_name}_agt_{assignment_generator_type}"
        f"_nc{node_count}_cpn{connections_per_node}"
        f"_na{number_of_assignments}_nc{num_cascades}"
        f"_a{alpha}_p{_format_probability(p1)}_{_format_probability(p2)}"
        f"_q{q_val}_fpa{features_per_assignment}{suffix}"
    )


def build_output_paths(
    *,
    graph_name: str,
    assignment_generator_type: str,
    node_count: int,
    connections_per_node: int,
    number_of_assignments: int,
    num_cascades: int,
    alpha: float,
    p1: float,
    p2: float,
    q_val: float,
    features_per_assignment: int,
    seed_percentage: float,
    features_folder: Path,
    graph_folder: Path,
) -> OutputPaths:
    if graph_name == "random":
        graph_path = graph_folder / (
            f"graph_nc{node_count}_cpn{connections_per_node}.pkl"
        )
    else:
        graph_path = graph_folder / f"graph_{graph_name}.pkl"

    stem = _dataset_stem(
        graph_name,
        assignment_generator_type,
        node_count,
        connections_per_node,
        number_of_assignments,
        num_cascades,
        alpha,
        p1,
        p2,
        q_val,
        features_per_assignment,
        seed_percentage,
    )
    features_path = features_folder / f"features_{stem}.pkl"
    indices_path = features_folder / f"indices_{stem}.pkl"
    seeds_path = features_folder / f"selected_seeds_{stem}.pkl"
    feature_shard_dir = features_folder / f"features_{stem}"
    cascade_shard_dir = features_folder / f"cascades_{stem}"
    manifest_path = features_folder / f"manifest_{stem}.json"
    return OutputPaths(
        graph_path=graph_path,
        features_path=features_path,
        indices_path=indices_path,
        seeds_path=seeds_path,
        feature_shard_dir=feature_shard_dir,
        cascade_shard_dir=cascade_shard_dir,
        manifest_path=manifest_path,
    )


def atomic_pickle_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_npz_dump(path: Path, **arrays: Any) -> None:
    """Atomically save a NumPy archive without pickle-dependent objects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _pack_signal_tensor(signals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pack {-1,0,+1} observations into two bits per value.

    Four observations are stored in one uint8 byte. The original tensor shape
    is returned separately and is stored in the cascade shard.
    """
    array = np.asarray(signals, dtype=np.int8)
    if array.size and not np.isin(array, (-1, 0, 1)).all():
        raise ValueError("Signals must contain only -1, 0, and +1.")
    encoded = (array.reshape(-1).astype(np.int16) + 1).astype(np.uint8)
    padding = (-encoded.size) % 4
    if padding:
        encoded = np.pad(encoded, (0, padding), constant_values=1)
    grouped = encoded.reshape(-1, 4)
    packed = (
        grouped[:, 0]
        | (grouped[:, 1] << 2)
        | (grouped[:, 2] << 4)
        | (grouped[:, 3] << 6)
    ).astype(np.uint8, copy=False)
    return packed, np.asarray(array.shape, dtype=np.int64)


def unpack_signal_tensor(packed: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """Inverse of :func:`_pack_signal_tensor`; useful for benchmark loaders."""
    packed = np.asarray(packed, dtype=np.uint8).reshape(-1)
    decoded = np.empty(packed.size * 4, dtype=np.uint8)
    decoded[0::4] = packed & 0b11
    decoded[1::4] = (packed >> 2) & 0b11
    decoded[2::4] = (packed >> 4) & 0b11
    decoded[3::4] = (packed >> 6) & 0b11
    target_shape = tuple(int(value) for value in shape)
    target_size = int(np.prod(target_shape, dtype=np.int64))
    decoded = decoded[:target_size].astype(np.int8) - 1
    return decoded.reshape(target_shape)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _make_or_load_graph(
    *,
    graph_name: str,
    node_count: int,
    connections_per_node: int,
    graph_path: Path,
    graph_folder: Path,
    data_seed: int,
    force_graph: bool,
) -> nx.DiGraph:
    if graph_path.exists() and not force_graph:
        graph = _load_pickle(graph_path)
        print(f"Loaded graph: {graph_path}")
    else:
        if graph_name == "random":
            graph = nx.barabasi_albert_graph(
                node_count, connections_per_node, seed=data_seed
            )
        elif graph_name == "karate":
            graph = nx.karate_club_graph()
        elif graph_name == "facebook":
            source = graph_folder / "graph_facebook.pkl"
            if not source.exists():
                raise FileNotFoundError(
                    f"Facebook graph not found at {source}."
                )
            graph = _load_pickle(source)
        elif graph_name == "insider":
            source = graph_folder / "edges_2009.pkl"
            if not source.exists():
                raise FileNotFoundError(
                    f"Insider graph not found at {source}."
                )
            graph = _load_pickle(source)
        elif graph_name == "tree":
            source = graph_folder / "graph_tree.pkl"
            if not source.exists():
                raise FileNotFoundError(f"Tree graph not found at {source}.")
            graph = _load_pickle(source)
        else:
            raise ValueError(f"Unsupported graph_name: {graph_name!r}")

        if not graph.is_directed():
            graph = graph.to_directed()
        atomic_pickle_dump(graph, graph_path)
        print(f"Saved graph: {graph_path}")

    if not graph.is_directed():
        graph = graph.to_directed()

    # main.py treats feature row i as node i and creates edge_index directly
    # from graph node IDs.  Fail early if that implicit contract is violated.
    nodes = sorted(graph.nodes())
    expected = list(range(len(nodes)))
    if nodes != expected:
        raise ValueError(
            "Graph node IDs must be contiguous integers 0..N-1 for compatibility "
            "with main.py. Relabel the graph before generating features."
        )
    return nx.DiGraph(graph)


def _adjacency_csr(graph: nx.DiGraph, nodes: Sequence[int]) -> sparse.csr_matrix:
    adjacency = nx.to_scipy_sparse_array(
        graph,
        nodelist=list(nodes),
        weight=None,
        dtype=np.float32,
        format="csr",
    )
    adjacency = sparse.csr_matrix(adjacency)
    adjacency.sum_duplicates()
    if adjacency.nnz:
        # Neighbour fractions count a neighbour once, matching neighbor_map.
        adjacency.data.fill(1.0)
    return adjacency


def _select_seeds(graph: nx.DiGraph, seed_percentage: float) -> List[int]:
    node_order = list(graph.nodes())
    # degree_centrality has exactly the same ranking as degree and is much more
    # expensive to materialize for large graphs.
    selected_count = int(seed_percentage * len(node_order))
    if selected_count < 1:
        raise ValueError(
            "seed_percentage selects zero seeds; increase it so at least one seed is used."
        )
    return sorted(node_order, key=lambda node: graph.degree[node], reverse=True)[
        :selected_count
    ]


def _precompute_static_arrays(
    graph: nx.DiGraph,
    nodes: Sequence[int],
    selected_seeds: Sequence[int],
) -> Tuple[
    sparse.csr_matrix,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    adjacency = _adjacency_csr(graph, nodes)
    seed_indices = np.asarray(selected_seeds, dtype=np.int64)

    print(
        f"Computing directed unweighted distances for {len(selected_seeds)} seeds "
        f"and {len(nodes)} nodes..."
    )
    distances = shortest_path(
        adjacency,
        directed=True,
        unweighted=True,
        indices=seed_indices,
        return_predecessors=False,
    ).astype(np.float32, copy=False)
    if distances.ndim == 1:
        distances = distances[None, :]

    nonzero = distances.copy()
    nonzero[nonzero == 0.0] = np.inf
    min_nonzero = np.min(nonzero, axis=0).astype(np.float32, copy=False)
    mean_distance = np.mean(distances, axis=0).astype(np.float32, copy=False)
    degree = np.asarray([graph.degree[node] for node in nodes], dtype=np.float32)

    out_degree = np.asarray(adjacency.sum(axis=1)).reshape(-1).astype(np.float32)
    inverse_out_degree = np.zeros_like(out_degree)
    nonzero_degree = out_degree > 0
    inverse_out_degree[nonzero_degree] = 1.0 / out_degree[nonzero_degree]

    return (
        adjacency,
        seed_indices,
        distances,
        min_nonzero,
        mean_distance,
        degree,
        inverse_out_degree,
    )


def _effective_features_per_assignment(
    assignment_generator_type: str, features_per_assignment: int
) -> int:
    return 1 if assignment_generator_type.startswith("single") else features_per_assignment


def _feature_dimension(assignment_generator_type: str) -> int:
    if "self_seed_masked" in assignment_generator_type:
        return 14
    if "self_seed" in assignment_generator_type:
        return 13
    if "new_features" in assignment_generator_type:
        return 9
    return 7


def _feature_names(assignment_generator_type: str) -> Tuple[str, ...]:
    if "self_seed_masked" in assignment_generator_type:
        return FEATURE_NAMES_14
    if "self_seed" in assignment_generator_type:
        return FEATURE_NAMES_13
    if "new_features" in assignment_generator_type:
        return FEATURE_NAMES_9
    return FEATURE_NAMES_7


def _make_split_indices(
    *,
    number_of_assignments: int,
    features_per_assignment: int,
    train_fraction: float,
    validation_fraction: float,
    split_mode: str,
    data_seed: int,
) -> Dict[str, List[int]]:
    if train_fraction < 0 or validation_fraction < 0:
        raise ValueError("Split fractions must be non-negative.")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be < 1.")

    rng = np.random.default_rng(data_seed + 17_171)

    if split_mode == "sample":
        total = number_of_assignments * features_per_assignment
        shuffled = rng.permutation(total)
        train_size = int(train_fraction * total)
        validation_size = int(validation_fraction * total)
        train = shuffled[:train_size]
        validation = shuffled[train_size : train_size + validation_size]
        test = shuffled[train_size + validation_size :]
    elif split_mode == "assignment":
        # All feature sets from one node-p assignment remain in the same split.
        assignments = rng.permutation(number_of_assignments)
        train_size = int(train_fraction * number_of_assignments)
        validation_size = int(validation_fraction * number_of_assignments)
        groups = {
            "train": assignments[:train_size],
            "validation": assignments[train_size : train_size + validation_size],
            "test": assignments[train_size + validation_size :],
        }

        def expand(group: np.ndarray) -> np.ndarray:
            if group.size == 0:
                return np.empty(0, dtype=np.int64)
            offsets = np.arange(features_per_assignment, dtype=np.int64)
            return (group[:, None] * features_per_assignment + offsets[None, :]).reshape(-1)

        train = expand(groups["train"])
        validation = expand(groups["validation"])
        test = expand(groups["test"])
    else:
        raise ValueError("split_mode must be 'sample' or 'assignment'.")

    result = {
        "train": [int(value) for value in train],
        "validation": [int(value) for value in validation],
        "test": [int(value) for value in test],
    }
    _validate_split_indices(
        result, number_of_assignments * features_per_assignment
    )
    return result


def _validate_split_indices(indices: Dict[str, List[int]], total: int) -> None:
    sets = {name: set(values) for name, values in indices.items()}
    if sets["train"] & sets["validation"]:
        raise AssertionError("Train and validation indices overlap.")
    if sets["train"] & sets["test"]:
        raise AssertionError("Train and test indices overlap.")
    if sets["validation"] & sets["test"]:
        raise AssertionError("Validation and test indices overlap.")
    union = sets["train"] | sets["validation"] | sets["test"]
    if union != set(range(total)):
        raise AssertionError("Split indices do not cover every generated feature set.")


def _auto_feature_batch_size(
    requested: int,
    features_per_assignment: int,
    num_cascades: int,
    node_count: int,
) -> int:
    if requested > 0:
        return min(requested, features_per_assignment)
    # Keep dense working arrays around roughly tens of MB per worker.  The
    # cascade objects returned by the external package add further memory.
    target_cells = 5_000_000
    per_feature_cells = max(1, num_cascades * node_count)
    return max(1, min(features_per_assignment, target_cells // per_feature_cells))


def _assignment_path(context: WorkerContext, assignment_id: int) -> Path:
    return context.feature_shard_dir / f"assignment_{assignment_id}_features.pkl"


def _cascade_assignment_path(context: WorkerContext, assignment_id: int) -> Path:
    return context.cascade_shard_dir / f"assignment_{assignment_id}_cascades.npz"


def _initialize_worker(context: "WorkerContext") -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context
    _seed_worker_process()


def _seed_worker_process() -> None:
    # Prevent one process per assignment from each starting a full BLAS thread
    # pool.  These values are also set in the supplied SLURM launcher.
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(variable, "1")


def _import_cascade_generator():
    try:
        from cascadesimulator import pyCascadeGenerator
    except ImportError as exc:
        raise RuntimeError(
            "Could not import cascadesimulator.pyCascadeGenerator. Activate the "
            "same environment used by your original simulations before running "
            "this script."
        ) from exc
    return pyCascadeGenerator


def _signals_from_cascades(
    cascades: Sequence[Sequence[Any]],
    *,
    node_to_index: Dict[int, int],
    node_count: int,
    p1: float,
    p2: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Convert cascade observations to a dense {-1,0,+1} signal matrix.

    Non-activated nodes are sampled exactly from the same categorical
    probabilities used by cascade_utils.trades. Activated-node symptoms then
    overwrite those defaults, reproducing the old dictionary merge semantics.
    """
    cascade_count = len(cascades)
    signals = np.zeros((cascade_count, node_count), dtype=np.int8)

    if p1 > 0.0 or p2 > 0.0:
        draws = rng.random((cascade_count, node_count), dtype=np.float32)
        if p1 > 0.0:
            signals[draws < p1] = 1
        if p2 > 0.0:
            negative = (draws >= p1) & (draws < p1 + p2)
            signals[negative] = -1

    for cascade_index, cascade in enumerate(cascades):
        for observation in cascade:
            try:
                node_index = node_to_index[observation.node_id]
            except (AttributeError, KeyError) as exc:
                raise ValueError(
                    "Cascade observation has an unknown or missing node_id."
                ) from exc
            try:
                symptom = int(observation.symptom)
            except AttributeError as exc:
                raise ValueError(
                    "Cascade observation is missing the expected symptom attribute."
                ) from exc
            if symptom not in (-1, 0, 1):
                raise ValueError(
                    f"Unexpected activated-node symptom {symptom!r}; expected -1, 0, or 1."
                )
            signals[cascade_index, node_index] = symptom
    return signals


def _entropy_from_counts(raw_counts: np.ndarray) -> np.ndarray:
    totals = raw_counts.sum(axis=1)
    probabilities = np.divide(
        raw_counts,
        totals[:, None, :],
        out=np.zeros_like(raw_counts, dtype=np.float64),
        where=totals[:, None, :] > 0,
    )
    log_probabilities = np.zeros_like(probabilities)
    positive = probabilities > 0
    log_probabilities[positive] = np.log2(probabilities[positive])
    return -np.sum(probabilities * log_probabilities, axis=1)


def _accumulate_signal_batch(
    signals: np.ndarray,
    *,
    weighted_counts: np.ndarray,
    raw_counts: np.ndarray,
    weight: np.ndarray,
    feature_offset: int,
    num_features: int,
    num_cascades: int,
    include_neighbor_features: bool,
    adjacency: sparse.csr_matrix,
    inverse_out_degree: np.ndarray,
    neighbor_any_sum: Optional[np.ndarray],
    neighbor_positive_sum: Optional[np.ndarray],
    node_positive_count: Optional[np.ndarray],
) -> None:
    node_count = signals.shape[1]
    reshaped = signals.reshape(num_features, num_cascades, node_count)
    destination = slice(feature_offset, feature_offset + num_features)

    negative_count = np.count_nonzero(reshaped == -1, axis=1)
    zero_count = np.count_nonzero(reshaped == 0, axis=1)
    positive_count = np.count_nonzero(reshaped == 1, axis=1)

    raw_counts[destination, 0, :] += negative_count
    raw_counts[destination, 1, :] += zero_count
    raw_counts[destination, 2, :] += positive_count

    weighted_counts[destination, 0, :] += negative_count * weight[None, :]
    weighted_counts[destination, 1, :] += zero_count * weight[None, :]
    weighted_counts[destination, 2, :] += positive_count * weight[None, :]

    if not include_neighbor_features:
        return

    assert neighbor_any_sum is not None
    assert neighbor_positive_sum is not None
    assert node_positive_count is not None

    positive_flat = (signals == 1).astype(np.float32, copy=False)
    # A[u,v] = 1 for an out-edge u->v.  A @ positive.T therefore counts
    # positive out-neighbours for every node and cascade.
    fraction_transposed = adjacency.dot(positive_flat.T)
    fraction_transposed *= inverse_out_degree[:, None]
    fractions = np.asarray(fraction_transposed.T).reshape(
        num_features, num_cascades, node_count
    )
    positive_mask = positive_flat.reshape(num_features, num_cascades, node_count)

    neighbor_any_sum[destination, :] += fractions.sum(axis=1, dtype=np.float64)
    neighbor_positive_sum[destination, :] += (
        fractions * positive_mask
    ).sum(axis=1, dtype=np.float64)
    node_positive_count[destination, :] += positive_count


def _self_seed_batch_statistics(
    signals: np.ndarray,
    *,
    num_features: int,
    num_cascades: int,
    node_count: int,
    seed_index: int,
    adjacency: sparse.csr_matrix,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return four per-feature statistics for cascades seeded at one node.

    The source graph is undirected in the experiments.  It is represented
    internally by two directed edges per undirected edge, so one adjacency row
    still gives exactly the ordinary neighbours of the seed.
    """
    reshaped = signals.reshape(num_features, num_cascades, node_count)
    positive = reshaped == 1

    neighbor_indices = adjacency.getrow(seed_index).indices
    if neighbor_indices.size:
        neighbor_positive_fraction = positive[:, :, neighbor_indices].mean(axis=2)
    else:
        neighbor_positive_fraction = np.zeros(
            (num_features, num_cascades), dtype=np.float64
        )

    neighbor_mean = neighbor_positive_fraction.mean(axis=1)
    neighbor_variance = neighbor_positive_fraction.var(axis=1)

    positive_other_nodes = positive.sum(axis=2, dtype=np.int64) - positive[
        :, :, seed_index
    ].astype(np.int64, copy=False)
    reach_fraction = positive_other_nodes / max(1, node_count - 1)
    reach_mean = reach_fraction.mean(axis=1)
    reach_variance = reach_fraction.var(axis=1)

    return neighbor_mean, neighbor_variance, reach_mean, reach_variance


def _build_feature_sets_for_assignment(
    context: WorkerContext,
    assignment_id: int,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, np.ndarray]]]:
    pyCascadeGenerator = _import_cascade_generator()

    # Independent deterministic streams per assignment make results stable when
    # worker count or task order changes.
    seed_sequence = np.random.SeedSequence([context.data_seed, assignment_id, 941_083])
    generated = seed_sequence.generate_state(4, dtype=np.uint32)
    python_seed = int(generated[0])
    numpy_seed = int(generated[1])
    local_rng = np.random.default_rng(
        np.random.SeedSequence([int(generated[2]), int(generated[3])])
    )
    random.seed(python_seed)
    np.random.seed(numpy_seed)

    graph = context.graph.copy()
    node_count = len(context.nodes)
    feature_count = context.features_per_assignment
    include_neighbor_features = "new_features" in context.assignment_generator_type
    include_self_seed_features = "self_seed" in context.assignment_generator_type
    include_self_seed_mask = (
        "self_seed_masked" in context.assignment_generator_type
    )

    node_p = np.asarray(
        [round(random.uniform(0.0, 1.0), 2) for _ in context.nodes],
        dtype=np.float32,
    )
    for source, target in graph.edges():
        graph[source][target]["weight"] = float(node_p[context.node_to_index[source]])

    # Class order is [-1, 0, +1].
    weighted_counts = np.zeros((feature_count, 3, node_count), dtype=np.float64)
    raw_counts = np.zeros((feature_count, 3, node_count), dtype=np.int64)

    neighbor_any_sum: Optional[np.ndarray]
    neighbor_positive_sum: Optional[np.ndarray]
    node_positive_count: Optional[np.ndarray]
    if include_neighbor_features:
        neighbor_any_sum = np.zeros((feature_count, node_count), dtype=np.float64)
        neighbor_positive_sum = np.zeros(
            (feature_count, node_count), dtype=np.float64
        )
        node_positive_count = np.zeros((feature_count, node_count), dtype=np.int64)
    else:
        neighbor_any_sum = None
        neighbor_positive_sum = None
        node_positive_count = None

    self_seed_neighbor_mean: Optional[np.ndarray]
    self_seed_neighbor_variance: Optional[np.ndarray]
    self_seed_reach_mean: Optional[np.ndarray]
    self_seed_reach_variance: Optional[np.ndarray]
    self_seed_observed: Optional[np.ndarray]
    if include_self_seed_features:
        self_seed_neighbor_mean = np.zeros(
            (feature_count, node_count), dtype=np.float64
        )
        self_seed_neighbor_variance = np.zeros(
            (feature_count, node_count), dtype=np.float64
        )
        self_seed_reach_mean = np.zeros(
            (feature_count, node_count), dtype=np.float64
        )
        self_seed_reach_variance = np.zeros(
            (feature_count, node_count), dtype=np.float64
        )
        self_seed_observed = (
            np.zeros((feature_count, node_count), dtype=np.float64)
            if include_self_seed_mask
            else None
        )
    else:
        self_seed_neighbor_mean = None
        self_seed_neighbor_variance = None
        self_seed_reach_mean = None
        self_seed_reach_variance = None
        self_seed_observed = None

    q = [context.q_val] * node_count

    # Master cascade storage is assignment based. Axis order is
    # [feature realization, selected seed, cascade, node]. This permits any
    # later subset/test split to reference the same raw master data.
    cascade_signals: Optional[np.ndarray] = None
    if context.save_cascades:
        cascade_signals = np.empty(
            (
                feature_count,
                len(context.selected_seeds),
                context.num_cascades,
                node_count,
            ),
            dtype=np.int8,
        )

    for seed_position, seed_node in enumerate(context.selected_seeds):
        distance = context.distance_matrix[seed_position]
        weight = np.zeros(node_count, dtype=np.float64)
        finite = np.isfinite(distance)
        weight[finite] = np.exp(
            -context.alpha * distance[finite].astype(np.float64, copy=False)
        )

        # Constructing the generator once per seed instead of once per
        # (feature set, seed) preserves isolation while removing an F-fold cost.
        generator = pyCascadeGenerator(graph=graph, cascade_model="IC", q=q)

        for feature_start in range(0, feature_count, context.feature_batch_size):
            current_features = min(
                context.feature_batch_size, feature_count - feature_start
            )
            requested_cascades = current_features * context.num_cascades
            cascades = generator.generate([seed_node], requested_cascades)
            if len(cascades) != requested_cascades:
                raise RuntimeError(
                    f"Cascade generator returned {len(cascades)} cascades after "
                    f"requesting {requested_cascades}."
                )

            signals = _signals_from_cascades(
                cascades,
                node_to_index=context.node_to_index,
                node_count=node_count,
                p1=context.p1,
                p2=context.p2,
                rng=local_rng,
            )

            if cascade_signals is not None:
                cascade_signals[
                    feature_start : feature_start + current_features,
                    seed_position,
                    :,
                    :,
                ] = signals.reshape(
                    current_features, context.num_cascades, node_count
                )

            if include_self_seed_features:
                assert self_seed_neighbor_mean is not None
                assert self_seed_neighbor_variance is not None
                assert self_seed_reach_mean is not None
                assert self_seed_reach_variance is not None

                seed_index = context.node_to_index[seed_node]
                destination = slice(
                    feature_start, feature_start + current_features
                )
                if include_self_seed_mask:
                    assert self_seed_observed is not None
                    self_seed_observed[destination, seed_index] = 1.0
                (
                    batch_neighbor_mean,
                    batch_neighbor_variance,
                    batch_reach_mean,
                    batch_reach_variance,
                ) = _self_seed_batch_statistics(
                    signals,
                    num_features=current_features,
                    num_cascades=context.num_cascades,
                    node_count=node_count,
                    seed_index=seed_index,
                    adjacency=context.adjacency,
                )
                self_seed_neighbor_mean[destination, seed_index] = (
                    batch_neighbor_mean
                )
                self_seed_neighbor_variance[destination, seed_index] = (
                    batch_neighbor_variance
                )
                self_seed_reach_mean[destination, seed_index] = batch_reach_mean
                self_seed_reach_variance[destination, seed_index] = (
                    batch_reach_variance
                )

            _accumulate_signal_batch(
                signals,
                weighted_counts=weighted_counts,
                raw_counts=raw_counts,
                weight=weight,
                feature_offset=feature_start,
                num_features=current_features,
                num_cascades=context.num_cascades,
                include_neighbor_features=include_neighbor_features,
                adjacency=context.adjacency,
                inverse_out_degree=context.inverse_out_degree,
                neighbor_any_sum=neighbor_any_sum,
                neighbor_positive_sum=neighbor_positive_sum,
                node_positive_count=node_positive_count,
            )
            del cascades, signals

    totals = weighted_counts.sum(axis=1)
    positive_ratio = np.divide(
        weighted_counts[:, 2, :],
        totals,
        out=np.zeros_like(totals),
        where=totals > 0,
    )
    negative_ratio = np.divide(
        weighted_counts[:, 0, :],
        totals,
        out=np.zeros_like(totals),
        where=totals > 0,
    )
    zero_ratio = np.divide(
        weighted_counts[:, 1, :],
        totals,
        out=np.zeros_like(totals),
        where=totals > 0,
    )
    entropy = _entropy_from_counts(raw_counts)

    static = np.broadcast_to(
        np.stack(
            [
                context.min_nonzero_distance,
                context.mean_distance,
                context.degree,
            ],
            axis=1,
        )[None, :, :],
        (feature_count, node_count, 3),
    )

    base = np.stack(
        [positive_ratio, negative_ratio, zero_ratio, entropy], axis=2
    )
    feature_tensor = np.concatenate([base, static], axis=2)

    if include_neighbor_features:
        assert neighbor_any_sum is not None
        assert neighbor_positive_sum is not None
        assert node_positive_count is not None
        snapshot_count = context.num_cascades * len(context.selected_seeds)
        neighbor_any = neighbor_any_sum / max(1, snapshot_count)
        neighbor_given_positive = np.divide(
            neighbor_positive_sum,
            node_positive_count,
            out=np.zeros_like(neighbor_positive_sum),
            where=node_positive_count > 0,
        )
        feature_tensor = np.concatenate(
            [
                feature_tensor,
                neighbor_any[:, :, None],
                neighbor_given_positive[:, :, None],
            ],
            axis=2,
        )

    if include_self_seed_features:
        assert self_seed_neighbor_mean is not None
        assert self_seed_neighbor_variance is not None
        assert self_seed_reach_mean is not None
        assert self_seed_reach_variance is not None
        self_seed_components = [
            self_seed_neighbor_mean,
            self_seed_neighbor_variance,
            self_seed_reach_mean,
            self_seed_reach_variance,
        ]
        if include_self_seed_mask:
            assert self_seed_observed is not None
            self_seed_components.append(self_seed_observed)
        self_seed_tensor = np.stack(self_seed_components, axis=2)
        feature_tensor = np.concatenate(
            [feature_tensor, self_seed_tensor], axis=2
        )

    feature_tensor = feature_tensor.astype(np.float32, copy=False)
    labels = node_p.astype(np.float32, copy=False)
    shared_nodes = list(context.nodes)
    feature_sets = [
        {
            "features": feature_tensor[index],
            "labels": labels.copy(),
            "nodes": shared_nodes,
        }
        for index in range(feature_count)
    ]

    cascade_payload: Optional[Dict[str, np.ndarray]] = None
    if cascade_signals is not None:
        packed_signals, signal_shape = _pack_signal_tensor(cascade_signals)
        cascade_payload = {
            "packed_signals": packed_signals,
            "signal_shape": signal_shape,
            "labels": labels,
            "nodes": np.asarray(context.nodes, dtype=np.int64),
            "selected_seeds": np.asarray(context.selected_seeds, dtype=np.int64),
            "assignment_id": np.asarray([assignment_id], dtype=np.int64),
            "encoding_version": np.asarray([1], dtype=np.int16),
            "encoding_offset": np.asarray([1], dtype=np.int8),
            "bits_per_signal": np.asarray([2], dtype=np.int8),
        }
    return feature_sets, cascade_payload


def _run_assignment(assignment_id: int) -> Tuple[int, float, str]:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Worker context was not initialized.")
    context = _WORKER_CONTEXT
    _seed_worker_process()
    feature_path = _assignment_path(context, assignment_id)
    cascade_path = _cascade_assignment_path(context, assignment_id)

    need_features = context.force or not feature_path.exists()
    need_cascades = context.save_cascades and (context.force or not cascade_path.exists())
    if not need_features and not need_cascades:
        return assignment_id, 0.0, "skipped"

    started = time.perf_counter()
    feature_sets, cascade_payload = _build_feature_sets_for_assignment(
        context, assignment_id
    )
    if need_features:
        atomic_pickle_dump(feature_sets, feature_path)
    if need_cascades:
        if cascade_payload is None:
            raise RuntimeError("Cascade payload was requested but not generated.")
        atomic_npz_dump(cascade_path, **cascade_payload)

    elapsed = time.perf_counter() - started
    if need_features and need_cascades:
        status = "created_features_and_cascades"
    elif need_features:
        status = "created_features"
    else:
        status = "created_cascades_from_existing_feature_assignment"
    return assignment_id, elapsed, status


def _valid_existing_assignment(
    path: Path,
    expected_feature_sets: int,
    expected_nodes: int,
    expected_features: int,
) -> bool:
    try:
        data = _load_pickle(path)
        if not isinstance(data, list) or len(data) != expected_feature_sets:
            return False
        for item in data:
            if item["features"].shape != (expected_nodes, expected_features):
                return False
            if item["labels"].shape != (expected_nodes,):
                return False
        return True
    except (OSError, EOFError, pickle.UnpicklingError, KeyError, AttributeError):
        return False


def _valid_existing_cascade_assignment(
    path: Path,
    expected_feature_sets: int,
    expected_seeds: int,
    expected_cascades: int,
    expected_nodes: int,
) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "packed_signals",
                "signal_shape",
                "labels",
                "nodes",
                "selected_seeds",
                "assignment_id",
            }
            if not required.issubset(data.files):
                return False
            shape = tuple(int(value) for value in data["signal_shape"])
            expected_shape = (
                expected_feature_sets,
                expected_seeds,
                expected_cascades,
                expected_nodes,
            )
            if shape != expected_shape:
                return False
            expected_bytes = (int(np.prod(shape, dtype=np.int64)) + 3) // 4
            if data["packed_signals"].size != expected_bytes:
                return False
            if data["labels"].shape != (expected_nodes,):
                return False
            if data["nodes"].shape != (expected_nodes,):
                return False
            if data["selected_seeds"].shape != (expected_seeds,):
                return False
        return True
    except (OSError, EOFError, ValueError, KeyError):
        return False


def _assignment_ids_to_generate(
    *,
    context: WorkerContext,
    verify_existing: bool,
    shard_id: int,
    num_shards: int,
) -> List[int]:
    expected_features = _feature_dimension(context.assignment_generator_type)
    result: List[int] = []
    for assignment_id in range(context.number_of_assignments):
        if assignment_id % num_shards != shard_id:
            continue

        feature_path = _assignment_path(context, assignment_id)
        cascade_path = _cascade_assignment_path(context, assignment_id)
        needs_generation = context.force or not feature_path.exists()

        if (
            not needs_generation
            and verify_existing
            and not _valid_existing_assignment(
                feature_path,
                context.features_per_assignment,
                len(context.nodes),
                expected_features,
            )
        ):
            feature_path.unlink(missing_ok=True)
            needs_generation = True

        if context.save_cascades:
            cascade_missing = not cascade_path.exists()
            cascade_invalid = (
                verify_existing
                and not cascade_missing
                and not _valid_existing_cascade_assignment(
                    cascade_path,
                    context.features_per_assignment,
                    len(context.selected_seeds),
                    context.num_cascades,
                    len(context.nodes),
                )
            )
            if cascade_invalid:
                cascade_path.unlink(missing_ok=True)
            needs_generation = (
                needs_generation or context.force or cascade_missing or cascade_invalid
            )

        if needs_generation:
            result.append(assignment_id)
    return result


def _run_parallel_assignments(
    context: WorkerContext,
    assignment_ids: Sequence[int],
    workers: int,
    start_method: str,
) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context

    if not assignment_ids:
        print("No assignment shards need to be generated.")
        return

    workers = max(1, min(workers, len(assignment_ids)))
    print(
        f"Generating {len(assignment_ids)} assignments with {workers} worker(s); "
        f"feature batch size={context.feature_batch_size}."
    )

    if workers == 1:
        for completed, assignment_id in enumerate(assignment_ids, start=1):
            result = _run_assignment(assignment_id)
            print(
                f"[{completed}/{len(assignment_ids)}] assignment={result[0]} "
                f"status={result[2]} seconds={result[1]:.2f}",
                flush=True,
            )
        return

    if start_method == "fork" and "fork" not in mp.get_all_start_methods():
        raise RuntimeError("The 'fork' multiprocessing start method is unavailable.")

    multiprocessing_context = mp.get_context(start_method)
    # maxtasksperchild limits long-run memory growth from the external simulator.
    pool_kwargs: Dict[str, Any] = {
        "processes": workers,
        "maxtasksperchild": 20,
    }
    if start_method != "fork":
        # Spawned workers do not inherit module globals. This serializes the
        # context once per worker; fork remains strongly preferred on Linux.
        pool_kwargs["initializer"] = _initialize_worker
        pool_kwargs["initargs"] = (context,)
    with multiprocessing_context.Pool(**pool_kwargs) as pool:
        iterator = pool.imap_unordered(_run_assignment, assignment_ids, chunksize=1)
        for completed, result in enumerate(iterator, start=1):
            print(
                f"[{completed}/{len(assignment_ids)}] assignment={result[0]} "
                f"status={result[2]} seconds={result[1]:.2f}",
                flush=True,
            )


def _consolidate_feature_shards(
    *,
    context: WorkerContext,
    features_path: Path,
    force: bool,
) -> None:
    if features_path.exists() and not force:
        print(f"Consolidated features already exist: {features_path}")
        return

    missing = [
        assignment_id
        for assignment_id in range(context.number_of_assignments)
        if not _assignment_path(context, assignment_id).exists()
    ]
    if missing:
        preview = ", ".join(str(value) for value in missing[:10])
        raise RuntimeError(
            f"Cannot consolidate: {len(missing)} assignment shards are missing "
            f"(first: {preview})."
        )

    print("Consolidating assignment shards into the legacy feature pickle...")
    combined: List[Dict[str, Any]] = []
    shared_nodes = list(context.nodes)
    for assignment_id in range(context.number_of_assignments):
        feature_sets = _load_pickle(_assignment_path(context, assignment_id))
        for item in feature_sets:
            # Reusing one object allows pickle memoization to avoid writing the
            # identical node list thousands of times.
            item["nodes"] = shared_nodes
            combined.append(item)
    atomic_pickle_dump(combined, features_path)
    print(f"Saved {len(combined)} feature sets: {features_path}")


def _validate_consolidated_sample(
    *,
    context: WorkerContext,
    features_path: Path,
    indices: Dict[str, List[int]],
) -> Dict[str, Any]:
    feature_sets = _load_pickle(features_path)
    expected_count = context.number_of_assignments * context.features_per_assignment
    if len(feature_sets) != expected_count:
        raise AssertionError(
            f"Expected {expected_count} feature sets, found {len(feature_sets)}."
        )
    feature_dim = _feature_dimension(context.assignment_generator_type)
    first = feature_sets[0]
    if first["features"].shape != (len(context.nodes), feature_dim):
        raise AssertionError("Unexpected feature matrix shape in consolidated data.")
    if first["labels"].shape != (len(context.nodes),):
        raise AssertionError("Unexpected label vector shape in consolidated data.")
    _validate_split_indices(indices, expected_count)

    has_inf = bool(np.isinf(first["features"]).any())
    has_nan = bool(np.isnan(first["features"]).any())
    if has_nan:
        raise AssertionError("NaN values found in the first generated feature set.")
    return {
        "feature_set_count": expected_count,
        "node_count": len(context.nodes),
        "feature_dim": feature_dim,
        "first_feature_set_contains_inf": has_inf,
    }


def _reference_accumulation(
    signals_by_seed: Sequence[np.ndarray],
    distances: np.ndarray,
    alpha: float,
    adjacency: sparse.csr_matrix,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Slow reference used by --self-test; independent of cascadesimulator."""
    feature_count, cascade_count, node_count = signals_by_seed[0].shape
    weighted = np.zeros((feature_count, 3, node_count), dtype=np.float64)
    raw = np.zeros((feature_count, 3, node_count), dtype=np.int64)
    any_sum = np.zeros((feature_count, node_count), dtype=np.float64)
    pos_sum = np.zeros((feature_count, node_count), dtype=np.float64)
    pos_count = np.zeros((feature_count, node_count), dtype=np.int64)
    out_neighbors = [adjacency.getrow(node).indices for node in range(node_count)]

    class_index = {-1: 0, 0: 1, 1: 2}
    for seed_position, seed_signals in enumerate(signals_by_seed):
        for feature_index in range(feature_count):
            for cascade_index in range(cascade_count):
                row = seed_signals[feature_index, cascade_index]
                for node, signal in enumerate(row):
                    raw[feature_index, class_index[int(signal)], node] += 1
                    distance = distances[seed_position, node]
                    weight = math.exp(-alpha * distance) if np.isfinite(distance) else 0.0
                    weighted[feature_index, class_index[int(signal)], node] += weight
                    neighbors = out_neighbors[node]
                    fraction = (
                        float(np.count_nonzero(row[neighbors] == 1)) / len(neighbors)
                        if len(neighbors)
                        else 0.0
                    )
                    any_sum[feature_index, node] += fraction
                    if signal == 1:
                        pos_sum[feature_index, node] += fraction
                        pos_count[feature_index, node] += 1
    return weighted, raw, any_sum, pos_sum, pos_count


def run_self_test() -> None:
    rng = np.random.default_rng(123)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(5))
    graph.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 0)])
    adjacency = _adjacency_csr(graph, list(range(5)))
    out_degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    inverse = np.divide(
        1.0,
        out_degree,
        out=np.zeros_like(out_degree, dtype=np.float32),
        where=out_degree > 0,
    )
    feature_count, cascade_count, node_count, seed_count = 3, 7, 5, 2
    distances = np.asarray(
        [[0, 1, 1, 2, 3], [3, 2, 1, 0, 1]], dtype=np.float32
    )
    signals_by_seed = [
        rng.choice([-1, 0, 1], size=(feature_count, cascade_count, node_count)).astype(
            np.int8
        )
        for _ in range(seed_count)
    ]

    weighted = np.zeros((feature_count, 3, node_count), dtype=np.float64)
    raw = np.zeros((feature_count, 3, node_count), dtype=np.int64)
    any_sum = np.zeros((feature_count, node_count), dtype=np.float64)
    pos_sum = np.zeros((feature_count, node_count), dtype=np.float64)
    pos_count = np.zeros((feature_count, node_count), dtype=np.int64)

    for seed_position, cube in enumerate(signals_by_seed):
        distance = distances[seed_position]
        weight = np.exp(-0.8 * distance.astype(np.float64, copy=False))
        _accumulate_signal_batch(
            cube.reshape(feature_count * cascade_count, node_count),
            weighted_counts=weighted,
            raw_counts=raw,
            weight=weight,
            feature_offset=0,
            num_features=feature_count,
            num_cascades=cascade_count,
            include_neighbor_features=True,
            adjacency=adjacency,
            inverse_out_degree=inverse,
            neighbor_any_sum=any_sum,
            neighbor_positive_sum=pos_sum,
            node_positive_count=pos_count,
        )

    reference = _reference_accumulation(
        signals_by_seed, distances, 0.8, adjacency
    )
    np.testing.assert_allclose(weighted, reference[0], rtol=1e-7, atol=1e-7)
    np.testing.assert_array_equal(raw, reference[1])
    np.testing.assert_allclose(any_sum, reference[2], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(pos_sum, reference[3], rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(pos_count, reference[4])

    self_seed_signals = np.asarray(
        [
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0],
        ],
        dtype=np.int8,
    )
    (
        self_neighbor_mean,
        self_neighbor_variance,
        self_reach_mean,
        self_reach_variance,
    ) = _self_seed_batch_statistics(
        self_seed_signals,
        num_features=1,
        num_cascades=3,
        node_count=5,
        seed_index=0,
        adjacency=adjacency,
    )
    np.testing.assert_allclose(self_neighbor_mean, [0.5])
    np.testing.assert_allclose(self_neighbor_variance, [1.0 / 6.0])
    np.testing.assert_allclose(self_reach_mean, [0.25])
    np.testing.assert_allclose(self_reach_variance, [1.0 / 24.0])
    assert _feature_dimension("multiple_new_features_self_seed") == 13
    assert _feature_dimension("multiple_new_features_self_seed_masked") == 14
    assert _feature_names("multiple_new_features_self_seed_masked")[-1] == (
        "self_seed_observed"
    )

    packing_input = rng.choice(
        [-1, 0, 1], size=(3, 2, 7, 5)
    ).astype(np.int8)
    packed, packed_shape = _pack_signal_tensor(packing_input)
    np.testing.assert_array_equal(
        unpack_signal_tensor(packed, packed_shape), packing_input
    )

    # Test activated observations overwrite non-activated defaults.
    class Observation:
        def __init__(self, node_id: int, symptom: int):
            self.node_id = node_id
            self.symptom = symptom

    converted = _signals_from_cascades(
        [
            [Observation(0, 1), Observation(2, -1)],
            [Observation(1, 0), Observation(4, 1)],
        ],
        node_to_index={value: value for value in range(5)},
        node_count=5,
        p1=0.0,
        p2=0.0,
        rng=np.random.default_rng(7),
    )
    np.testing.assert_array_equal(converted[0], np.asarray([1, 0, -1, 0, 0]))
    np.testing.assert_array_equal(converted[1], np.asarray([0, 0, 0, 0, 1]))

    for mode in ("sample", "assignment"):
        split = _make_split_indices(
            number_of_assignments=11,
            features_per_assignment=4,
            train_fraction=0.7,
            validation_fraction=0.15,
            split_mode=mode,
            data_seed=9,
        )
        _validate_split_indices(split, 44)
    print("Self-test passed: vectorized accumulation matches the slow reference.")


def _default_worker_count() -> int:
    slurm_value = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_value:
        try:
            return max(1, int(slurm_value))
        except ValueError:
            pass
    return max(1, os.cpu_count() or 1)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate cascade features in parallel with resumable assignment shards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--name", default="simulation")
    parser.add_argument("--number-of-assignments", "--number_of_assignments", type=int, default=5000)
    parser.add_argument("--num-cascades", "--num_cascades", type=int, default=300)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--p1", type=float, default=0)
    parser.add_argument("--p2", type=float, default=0)
    parser.add_argument("--q-val", "--q_val", type=float, default=1.0)
    parser.add_argument("--features-per-assignment", "--features_per_assignment", type=int, default=10)
    parser.add_argument("--node-count", "--node_count", type=int, default=100)
    parser.add_argument("--connections-per-node", "--connections_per_node", type=int, default=2)
    parser.add_argument(
        "--assignment-generator-type",
        "--assignment_generator_type",
        "--assignemnt_generator_type",
        choices=sorted(SUPPORTED_GENERATORS),
        default="multiple_new_features",
    )
    parser.add_argument(
        "--graph-name",
        "--graph_name",
        choices=("random", "karate", "facebook", "insider", "tree"),
        default="random",
    )
    parser.add_argument("--seed-percentage", "--seed_percentage", type=float, default=1.0)
    parser.add_argument("--train-fraction", "--train_fraction", type=float, default=0.7)
    parser.add_argument("--validation-fraction", "--validation_fraction", type=float, default=0.15)
    parser.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="Accepted for compatibility; the uploaded legacy generator also did not use it",
    )
    parser.add_argument(
        "--create-external-set",
        "--create_external_set",
        action="store_true",
        help="Unsupported in the optimized generator because it stores huge Python signal objects",
    )
    parser.add_argument(
        "--split-mode",
        "--split_mode",
        choices=("sample", "assignment"),
        default="sample",
        help=(
            "'sample' reproduces the old feature-set-level split; 'assignment' "
            "prevents identical assignment labels from crossing splits"
        ),
    )
    parser.add_argument("--data-seed", "--data_seed", type=int, default=0)
    parser.add_argument(
        "--workers",
        "--n-jobs",
        "--n_jobs",
        type=int,
        default=0,
        help="0 uses SLURM_CPUS_PER_TASK or all visible CPUs",
    )
    parser.add_argument(
        "--feature-batch-size",
        "--feature_batch_size",
        type=int,
        default=0,
        help="0 selects a memory-aware automatic batch size",
    )
    parser.add_argument(
        "--start-method",
        choices=tuple(mp.get_all_start_methods()),
        default="fork" if "fork" in mp.get_all_start_methods() else mp.get_start_method(),
    )
    parser.add_argument(
        "--features-folder",
        "--features_folder",
        type=Path,
        default=Path("/scratch/svc_td_fincomp/vrango/hic_new/assignment_split/features"),
    )
    parser.add_argument(
        "--graph-folder", "--graph_folder", type=Path, default=Path.cwd() / "data" / "graphs"
    )
    parser.add_argument("--force", "--force-create-features", "--force_create_features", action="store_true")
    parser.add_argument("--force-graph", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument(
        "--save-cascades",
        "--save_cascades",
        action="store_true",
        help=(
            "Save one packed raw-observation cascade shard per assignment. "
            "The shard contains all feature realizations, so later subset-specific "
            "test manifests can reference it without duplicating cascade data."
        ),
    )
    parser.add_argument("--no-consolidate", action="store_true")
    parser.add_argument("--consolidate-only", action="store_true")
    parser.add_argument("--validate-output", action="store_true")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.create_external_set:
        raise ValueError(
            "create_external_set is not supported by the optimized generator. "
            "The normal model-training path uses False."
        )
    if args.number_of_assignments < 1:
        raise ValueError("number_of_assignments must be >= 1.")
    if args.num_cascades < 1:
        raise ValueError("num_cascades must be >= 1.")
    if args.features_per_assignment < 1:
        raise ValueError("features_per_assignment must be >= 1.")
    if not (0.0 <= args.p1 <= 1.0 and 0.0 <= args.p2 <= 1.0):
        raise ValueError("p1 and p2 must be in [0,1].")
    if args.p1 + args.p2 > 1.0:
        raise ValueError("p1 + p2 must be <= 1.")
    if not (0.0 <= args.q_val <= 1.0):
        raise ValueError("q_val must be in [0,1].")
    if not (0.0 < args.seed_percentage <= 1.0):
        raise ValueError("seed_percentage must be in (0,1].")
    if (
        args.assignment_generator_type == "multiple_new_features_self_seed"
        and not math.isclose(args.seed_percentage, 1.0)
    ):
        raise ValueError(
            "The 13-feature self-seed mode requires --seed-percentage=1.0. "
            "For partial seed selection, use "
            "--assignment-generator-type=multiple_new_features_self_seed_masked, "
            "which adds the self_seed_observed mask as feature 14."
        )
    if args.num_shards < 1:
        raise ValueError("num_shards must be >= 1.")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("shard_id must satisfy 0 <= shard_id < num_shards.")
    if args.graph_name == "random":
        if args.node_count < 2:
            raise ValueError("node_count must be >= 2 for a random graph.")
        if not (1 <= args.connections_per_node < args.node_count):
            raise ValueError(
                "connections_per_node must satisfy 1 <= value < node_count."
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0

    _validate_arguments(args)
    args.features_folder = args.features_folder.expanduser().resolve()
    args.graph_folder = args.graph_folder.expanduser().resolve()
    args.features_folder.mkdir(parents=True, exist_ok=True)
    args.graph_folder.mkdir(parents=True, exist_ok=True)

    fixed_node_counts = {"karate": 34, "facebook": 2887, "insider": 1502, "tree": 5}
    effective_node_count = fixed_node_counts.get(args.graph_name, args.node_count)
    effective_fpa = _effective_features_per_assignment(
        args.assignment_generator_type, args.features_per_assignment
    )

    paths = build_output_paths(
        graph_name=args.graph_name,
        assignment_generator_type=args.assignment_generator_type,
        node_count=effective_node_count,
        connections_per_node=args.connections_per_node,
        number_of_assignments=args.number_of_assignments,
        num_cascades=args.num_cascades,
        alpha=args.alpha,
        p1=args.p1,
        p2=args.p2,
        q_val=args.q_val,
        features_per_assignment=args.features_per_assignment,
        seed_percentage=args.seed_percentage,
        features_folder=args.features_folder,
        graph_folder=args.graph_folder,
    )
    paths.feature_shard_dir.mkdir(parents=True, exist_ok=True)
    if args.save_cascades:
        paths.cascade_shard_dir.mkdir(parents=True, exist_ok=True)

    graph = _make_or_load_graph(
        graph_name=args.graph_name,
        node_count=effective_node_count,
        connections_per_node=args.connections_per_node,
        graph_path=paths.graph_path,
        graph_folder=args.graph_folder,
        data_seed=args.data_seed,
        force_graph=args.force_graph,
    )
    nodes = tuple(sorted(graph.nodes()))
    node_to_index = {node: index for index, node in enumerate(nodes)}
    selected_seeds = tuple(_select_seeds(graph, args.seed_percentage))

    (
        adjacency,
        seed_indices,
        distances,
        min_nonzero,
        mean_distance,
        degree,
        inverse_out_degree,
    ) = _precompute_static_arrays(graph, nodes, selected_seeds)

    feature_batch_size = _auto_feature_batch_size(
        args.feature_batch_size,
        effective_fpa,
        args.num_cascades,
        len(nodes),
    )
    context = WorkerContext(
        graph=graph,
        nodes=nodes,
        node_to_index=node_to_index,
        selected_seeds=selected_seeds,
        seed_indices=seed_indices,
        distance_matrix=distances,
        min_nonzero_distance=min_nonzero,
        mean_distance=mean_distance,
        degree=degree,
        adjacency=adjacency,
        inverse_out_degree=inverse_out_degree,
        number_of_assignments=args.number_of_assignments,
        num_cascades=args.num_cascades,
        features_per_assignment=effective_fpa,
        assignment_generator_type=args.assignment_generator_type,
        alpha=args.alpha,
        p1=args.p1,
        p2=args.p2,
        q_val=args.q_val,
        data_seed=args.data_seed,
        feature_batch_size=feature_batch_size,
        feature_shard_dir=paths.feature_shard_dir,
        cascade_shard_dir=paths.cascade_shard_dir,
        save_cascades=args.save_cascades,
        force=args.force,
    )

    indices = _make_split_indices(
        number_of_assignments=args.number_of_assignments,
        features_per_assignment=effective_fpa,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        split_mode=args.split_mode,
        data_seed=args.data_seed,
    )
    atomic_pickle_dump(indices, paths.indices_path)
    atomic_pickle_dump(list(selected_seeds), paths.seeds_path)

    estimated_cascades = (
        args.number_of_assignments
        * effective_fpa
        * len(selected_seeds)
        * args.num_cascades
    )
    print(
        f"Planned cascade simulations: {estimated_cascades:,} "
        f"({args.number_of_assignments} assignments x {effective_fpa} feature sets "
        f"x {len(selected_seeds)} seeds x {args.num_cascades} cascades)."
    )
    if args.save_cascades:
        total_signal_values = (
            args.number_of_assignments
            * effective_fpa
            * len(selected_seeds)
            * args.num_cascades
            * len(nodes)
        )
        packed_bytes = (total_signal_values + 3) // 4
        print(
            "Packed cascade-master estimate: "
            f"{packed_bytes / (1024 ** 3):.2f} GiB before NPZ/container metadata "
            "(two bits per observed node signal)."
        )
        print(f"Cascade assignment shards: {paths.cascade_shard_dir}")

    if estimated_cascades >= 1_000_000_000:
        print(
            "WARNING: this configuration requests at least one billion cascades. "
            "Parallel/vectorized code cannot make that small; reduce assignments, "
            "num_cascades, features_per_assignment, or seed_percentage.",
            file=sys.stderr,
        )

    if not args.consolidate_only:
        assignment_ids = _assignment_ids_to_generate(
            context=context,
            verify_existing=args.verify_existing,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
        )
        workers = args.workers if args.workers > 0 else _default_worker_count()
        _run_parallel_assignments(
            context,
            assignment_ids,
            workers=workers,
            start_method=args.start_method,
        )

    validation_summary: Optional[Dict[str, Any]] = None
    should_consolidate = (
        not args.no_consolidate
        and (args.num_shards == 1 or args.consolidate_only)
    )
    if should_consolidate:
        _consolidate_feature_shards(
            context=context,
            features_path=paths.features_path,
            force=args.force or args.consolidate_only,
        )
        if args.validate_output:
            validation_summary = _validate_consolidated_sample(
                context=context,
                features_path=paths.features_path,
                indices=indices,
            )
            print(f"Validation: {validation_summary}")

    manifest = {
        "generator": "simulate_data_parallel.py",
        "name": args.name,
        "parameters": {
            "number_of_assignments": args.number_of_assignments,
            "num_cascades": args.num_cascades,
            "alpha": args.alpha,
            "p1": args.p1,
            "p2": args.p2,
            "q_val": args.q_val,
            "features_per_assignment_requested": args.features_per_assignment,
            "features_per_assignment_effective": effective_fpa,
            "assignment_generator_type": args.assignment_generator_type,
            "graph_name": args.graph_name,
            "node_count": len(nodes),
            "connections_per_node": args.connections_per_node,
            "seed_percentage": args.seed_percentage,
            "selected_seed_count": len(selected_seeds),
            "split_mode": args.split_mode,
            "train_fraction": args.train_fraction,
            "validation_fraction": args.validation_fraction,
            "cutoff_accepted_but_unused": args.cutoff,
            "data_seed": args.data_seed,
            "feature_batch_size": feature_batch_size,
            "estimated_cascade_count": estimated_cascades,
            "save_cascades": args.save_cascades,
            "cascade_encoding": (
                "2-bit packed {-1,0,+1}; shape=(feature,seed,cascade,node)"
                if args.save_cascades
                else None
            ),
        },
        "feature_names": list(_feature_names(args.assignment_generator_type)),
        "split_counts": {key: len(value) for key, value in indices.items()},
        "paths": {
            "graph": str(paths.graph_path),
            "features": str(paths.features_path),
            "indices": str(paths.indices_path),
            "selected_seeds": str(paths.seeds_path),
            "assignment_shards": str(paths.feature_shard_dir),
            "cascade_assignment_shards": (
                str(paths.cascade_shard_dir) if args.save_cascades else None
            ),
        },
        "validation": validation_summary,
    }
    atomic_json_dump(manifest, paths.manifest_path)
    print(f"Manifest: {paths.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
