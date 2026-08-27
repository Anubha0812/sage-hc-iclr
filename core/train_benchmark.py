"""Train saved graph-feature data with random seen/unseen validation and test sets.

By default this script prints logs only. Optional uncertainty JSON output can be enabled.

Split structure
---------------
1. Randomly divide assignment IDs into:
   - development assignments
   - unseen-validation assignments
   - unseen-test assignments
2. For every development assignment, randomly divide its realizations into:
   - train
   - validation_seen
   - test_seen
3. Every realization of an unseen-validation assignment goes to validation_unseen.
4. Every realization of an unseen-test assignment goes to test_unseen.

Checkpoint selection uses:
    selection_loss = seen_validation_weight * validation_seen_loss
                   + (1 - seen_validation_weight) * validation_unseen_loss

The default weight favors performance on new realizations of represented
assignments while still requiring performance on completely new assignments.
"""

from __future__ import annotations

import copy
import json
import os
import pickle
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from fire import Fire
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from core.model_utils import create_model, evaluate_model, make_criterion, set_seed

SCRIPT_VERSION = "hybrid-split-v6-normalized-master-subset-uncertainty-test-manifest"


def _normalise_node_count(graph_name: str, node_count: int) -> int:
    if graph_name == "random":
        return node_count

    known_sizes = {
        "karate": 34,
        "facebook": 2887,
        "insider": 1502,
        "tree": 5,
    }
    if graph_name not in known_sizes:
        raise ValueError(
            f"Unknown graph_name={graph_name!r}. Pass --graph_path explicitly."
        )
    return known_sizes[graph_name]


def build_saved_paths(
    *,
    graph_name: str,
    node_count: int,
    connections_per_node: Optional[int],
    number_of_assignments: int,
    num_cascades: int,
    alpha: float,
    p1: float,
    p2: float,
    q_val: float,
    features_per_assignment: int,
    assignemnt_generator_type: str,
    seed_percentage: float,
    features_folder: str,
    graph_folder: str,
    features_path: Optional[str],
    graph_path: Optional[str],
) -> Tuple[str, str]:
    """Construct the graph/feature paths used by simulations.py."""
    node_count = _normalise_node_count(graph_name, node_count)

    if graph_path is None:
        if graph_name == "random":
            graph_path = os.path.join(
                graph_folder,
                f"graph_nc{node_count}_cpn{connections_per_node}.pkl",
            )
        else:
            graph_path = os.path.join(graph_folder, f"graph_{graph_name}.pkl")

    if features_path is None:
        seed_suffix = (
            f"_seedpercentage_{seed_percentage}" if seed_percentage < 1.0 else ""
        )
        filename = (
            f"features_graph_{graph_name}"
            f"_agt_{assignemnt_generator_type}"
            f"_nc{node_count}"
            f"_cpn{connections_per_node}"
            f"_na{number_of_assignments}"
            f"_nc{num_cascades}"
            f"_a{alpha}"
            f"_p{str(p1).replace('.', '')}_{str(p2).replace('.', '')}"
            f"_q{q_val}"
            f"_fpa{features_per_assignment}"
            f"{seed_suffix}.pkl"
        )
        features_path = os.path.join(features_folder, filename)

    return os.path.abspath(graph_path), os.path.abspath(features_path)


def load_saved_graph_and_features(
    graph_path: str,
    features_path: str,
) -> Tuple[nx.Graph, List[Dict[str, Any]]]:
    if not os.path.isfile(graph_path):
        raise FileNotFoundError(f"Saved graph not found: {graph_path}")
    if not os.path.isfile(features_path):
        raise FileNotFoundError(f"Saved aggregate features not found: {features_path}")

    with open(graph_path, "rb") as handle:
        graph = pickle.load(handle)
    with open(features_path, "rb") as handle:
        features = pickle.load(handle)

    if not isinstance(features, list) or not features:
        raise ValueError("The aggregate feature pickle must contain a non-empty list.")

    required_keys = {"features", "labels", "nodes"}
    missing = required_keys.difference(features[0])
    if missing:
        raise ValueError(
            f"Feature dictionaries are missing required keys: {sorted(missing)}"
        )

    print(f"Graph loaded from:    {graph_path}")
    print(f"Features loaded from: {features_path}")
    print(f"Total graph samples:  {len(features)}")
    return graph, features



def _nested_assignment_ids(
    source_number_of_assignments: int,
    number_of_assignments: int,
    subset_seed: int,
) -> List[int]:
    """Return a reproducible nested subset of master assignment IDs."""
    if source_number_of_assignments < 1:
        raise ValueError("source_number_of_assignments must be at least 1.")
    if number_of_assignments < 1:
        raise ValueError("number_of_assignments must be at least 1.")
    if number_of_assignments > source_number_of_assignments:
        raise ValueError(
            "number_of_assignments cannot exceed source_number_of_assignments."
        )
    rng = np.random.default_rng(subset_seed)
    order = rng.permutation(source_number_of_assignments)
    return order[:number_of_assignments].astype(int).tolist()


def load_master_assignment_subset(
    graph_path: str,
    features_path: str,
    *,
    source_number_of_assignments: int,
    number_of_assignments: int,
    features_per_assignment: int,
    subset_seed: int,
) -> Tuple[nx.Graph, List[Dict[str, Any]], List[int]]:
    """Load a nested assignment subset from a master simulated dataset."""
    if not os.path.isfile(graph_path):
        raise FileNotFoundError(f"Saved graph not found: {graph_path}")

    with open(graph_path, "rb") as handle:
        graph = pickle.load(handle)

    selected_ids = _nested_assignment_ids(
        source_number_of_assignments,
        number_of_assignments,
        subset_seed,
    )

    shard_dir = os.path.splitext(features_path)[0]
    features: List[Dict[str, Any]] = []

    if os.path.isdir(shard_dir):
        print(f"Loading selected assignments from shards: {shard_dir}")
        for source_assignment_id in tqdm(
            selected_ids,
            desc="Loading master assignment subset",
            leave=False,
        ):
            shard_path = os.path.join(
                shard_dir,
                f"assignment_{source_assignment_id}_features.pkl",
            )
            if not os.path.isfile(shard_path):
                raise FileNotFoundError(
                    "Required master assignment shard is missing: " + shard_path
                )
            with open(shard_path, "rb") as handle:
                assignment_features = pickle.load(handle)
            if len(assignment_features) != features_per_assignment:
                raise ValueError(
                    f"Assignment shard {source_assignment_id} contains "
                    f"{len(assignment_features)} samples; expected "
                    f"{features_per_assignment}."
                )
            features.extend(assignment_features)
    else:
        if not os.path.isfile(features_path):
            raise FileNotFoundError(
                f"Saved aggregate features not found: {features_path}"
            )
        print(f"Loading master aggregate features: {features_path}")
        with open(features_path, "rb") as handle:
            master_features = pickle.load(handle)

        expected_master = source_number_of_assignments * features_per_assignment
        if len(master_features) != expected_master:
            raise ValueError(
                "Master feature count does not match the requested source layout: "
                f"expected {expected_master}, found {len(master_features)}."
            )

        for source_assignment_id in selected_ids:
            start = source_assignment_id * features_per_assignment
            stop = start + features_per_assignment
            features.extend(master_features[start:stop])
        del master_features

    if not features:
        raise ValueError("The selected master assignment subset is empty.")

    required_keys = {"features", "labels", "nodes"}
    missing = required_keys.difference(features[0])
    if missing:
        raise ValueError(
            f"Feature dictionaries are missing required keys: {sorted(missing)}"
        )

    print(f"Graph loaded from:             {graph_path}")
    print(f"Master feature source:         {features_path}")
    print(f"Source assignments available:  {source_number_of_assignments}")
    print(f"Assignments selected:          {number_of_assignments}")
    print(f"Assignment subset seed:        {subset_seed}")
    print(f"Selected graph samples:        {len(features)}")
    print(f"First selected source IDs:     {selected_ids[:10]}")
    return graph, features, selected_ids

def validate_assignment_layout(
    features: Sequence[Mapping[str, Any]],
    number_of_assignments: int,
    features_per_assignment: int,
    *,
    verify_labels: bool,
) -> None:
    expected = number_of_assignments * features_per_assignment
    actual = len(features)
    if actual != expected:
        raise ValueError(
            "Saved feature count does not match the requested layout: "
            f"expected {expected}, found {actual}."
        )

    if not verify_labels:
        print("Assignment count verified; label-layout verification skipped.")
        return

    for assignment_id in range(number_of_assignments):
        start = assignment_id * features_per_assignment
        reference = np.asarray(features[start]["labels"])

        for local_id in range(1, features_per_assignment):
            candidate = np.asarray(features[start + local_id]["labels"])
            if reference.shape != candidate.shape or not np.allclose(
                reference, candidate, rtol=0.0, atol=1e-8
            ):
                raise ValueError(
                    "Unexpected saved-data layout: labels differ inside "
                    f"assignment {assignment_id}, realization {local_id}."
                )

    print(
        "Assignment layout verified: all realizations inside an assignment "
        "share the same label vector."
    )


def _expand_assignments(
    assignment_ids: Iterable[int],
    features_per_assignment: int,
) -> List[int]:
    return [
        assignment_id * features_per_assignment + local_id
        for assignment_id in assignment_ids
        for local_id in range(features_per_assignment)
    ]


def create_hybrid_split(
    *,
    number_of_assignments: int,
    features_per_assignment: int,
    development_fraction: float,
    unseen_validation_fraction: float,
    seen_validation_fraction: float,
    seen_test_fraction: float,
    split_seed: int,
) -> Dict[str, Any]:
    """Create a reproducible five-way random split in memory."""
    if number_of_assignments < 3:
        raise ValueError("At least 3 assignments are required.")
    if features_per_assignment < 3:
        raise ValueError("At least 3 realizations per assignment are required.")


    for name, value in (
        ("development_fraction", development_fraction),
        ("unseen_validation_fraction", unseen_validation_fraction),
        ("seen_validation_fraction", seen_validation_fraction),
    ):
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")

    if not 0.0 <= seen_test_fraction < 1.0:
        raise ValueError("seen_test_fraction must be between 0 and 1, inclusive of 0.")


    if development_fraction + unseen_validation_fraction >= 1.0:
        raise ValueError(
            "development_fraction + unseen_validation_fraction must be less than 1."
        )
    if seen_validation_fraction + seen_test_fraction >= 1.0:
        raise ValueError(
            "seen_validation_fraction + seen_test_fraction must be less than 1."
        )

    rng = np.random.default_rng(split_seed)
    assignment_ids = rng.permutation(number_of_assignments).tolist()

    n_development = int(development_fraction * number_of_assignments)
    n_unseen_validation = int(unseen_validation_fraction * number_of_assignments)
    n_unseen_test = number_of_assignments - n_development - n_unseen_validation

    if min(n_development, n_unseen_validation, n_unseen_test) < 1:
        raise ValueError("The requested assignment fractions produce an empty set.")

    development_assignments = assignment_ids[:n_development]
    unseen_validation_assignments = assignment_ids[
        n_development : n_development + n_unseen_validation
    ]
    unseen_test_assignments = assignment_ids[n_development + n_unseen_validation :]

    n_seen_validation = max(
        1, int(round(seen_validation_fraction * features_per_assignment))
    )
    #n_seen_test = max(1, int(round(seen_test_fraction * features_per_assignment)))
    n_seen_test = int(round(seen_test_fraction * features_per_assignment))
    
    n_train = features_per_assignment - n_seen_validation - n_seen_test
    if n_train < 1:
        raise ValueError(
            "The seen validation/test fractions leave no training realizations."
        )

    train_indices: List[int] = []
    validation_seen_indices: List[int] = []
    test_seen_indices: List[int] = []
    local_split: Dict[int, Dict[str, List[int]]] = {}

    for assignment_id in development_assignments:
        local_ids = rng.permutation(features_per_assignment).tolist()
        validation_seen_local = sorted(local_ids[:n_seen_validation])
        test_seen_local = sorted(
            local_ids[n_seen_validation : n_seen_validation + n_seen_test]
        )
        train_local = sorted(local_ids[n_seen_validation + n_seen_test :])

        local_split[assignment_id] = {
            "train": train_local,
            "validation_seen": validation_seen_local,
            "test_seen": test_seen_local,
        }

        train_indices.extend(
            assignment_id * features_per_assignment + local_id
            for local_id in train_local
        )
        validation_seen_indices.extend(
            assignment_id * features_per_assignment + local_id
            for local_id in validation_seen_local
        )
        test_seen_indices.extend(
            assignment_id * features_per_assignment + local_id
            for local_id in test_seen_local
        )

    validation_unseen_indices = _expand_assignments(
        unseen_validation_assignments, features_per_assignment
    )
    test_unseen_indices = _expand_assignments(
        unseen_test_assignments, features_per_assignment
    )

    for indices in (
        train_indices,
        validation_seen_indices,
        validation_unseen_indices,
        test_seen_indices,
        test_unseen_indices,
    ):
        rng.shuffle(indices)

    split: Dict[str, Any] = {
        "split_seed": split_seed,
        "features_per_assignment": features_per_assignment,
        "development_assignments": development_assignments,
        "unseen_validation_assignments": unseen_validation_assignments,
        "unseen_test_assignments": unseen_test_assignments,
        "local_split": local_split,
        "train": train_indices,
        "validation_seen": validation_seen_indices,
        "validation_unseen": validation_unseen_indices,
        "test_seen": test_seen_indices,
        "test_unseen": test_unseen_indices,
    }
    validate_hybrid_split(split)
    return split


def validate_hybrid_split(split: Mapping[str, Any]) -> None:
    sample_keys = (
        "train",
        "validation_seen",
        "validation_unseen",
        "test_seen",
        "test_unseen",
    )
    sample_sets = {key: set(split[key]) for key in sample_keys}

    # test_seen is allowed to be empty when seen_test_fraction == 0.
    required_nonempty = (
        "train",
        "validation_seen",
        "validation_unseen",
        "test_unseen",
    )

    for key in required_nonempty:
        if not sample_sets[key]:
            raise ValueError(f"Split partition {key!r} is empty.")

    for i, left in enumerate(sample_keys):
        for right in sample_keys[i + 1 :]:
            overlap = sample_sets[left] & sample_sets[right]
            if overlap:
                raise ValueError(
                    f"Partitions {left!r} and {right!r} overlap: "
                    f"{sorted(overlap)[:5]}"
                )

    development = set(split["development_assignments"])
    unseen_validation = set(split["unseen_validation_assignments"])
    unseen_test = set(split["unseen_test_assignments"])
    if (
        development & unseen_validation
        or development & unseen_test
        or unseen_validation & unseen_test
    ):
        raise ValueError("Assignment partitions overlap.")

    fpa = int(split["features_per_assignment"])
    for key in ("train", "validation_seen"):
        assignment_membership = {sample_id // fpa for sample_id in split[key]}
        if assignment_membership != development:
            raise ValueError(f"{key} does not match development assignments.")

    # test_seen is optional. If present, it must come from development assignments.
    if split["test_seen"]:
        assignment_membership = {
            sample_id // fpa for sample_id in split["test_seen"]
        }
        if assignment_membership != development:
            raise ValueError("test_seen does not match development assignments.")

    validation_membership = {
        sample_id // fpa for sample_id in split["validation_unseen"]
    }
    test_membership = {sample_id // fpa for sample_id in split["test_unseen"]}
    if validation_membership != unseen_validation:
        raise ValueError("Unseen validation membership is inconsistent.")
    if test_membership != unseen_test:
        raise ValueError("Unseen test membership is inconsistent.")


def print_split_summary(split: Mapping[str, Any]) -> None:
    print("\n========== RANDOM HYBRID SPLIT ==========")
    print(f"Split seed:                    {split['split_seed']}")
    print(f"Development assignments:       {len(split['development_assignments'])}")
    print(
        "Unseen-validation assignments: "
        f"{len(split['unseen_validation_assignments'])}"
    )
    print(f"Unseen-test assignments:       {len(split['unseen_test_assignments'])}")
    print(f"Train graph samples:            {len(split['train'])}")
    print(f"Seen-validation graph samples:  {len(split['validation_seen'])}")
    print(f"Unseen-validation graph samples:{len(split['validation_unseen'])}")
    print(f"Seen-test graph samples:        {len(split['test_seen'])}")
    print(f"Unseen-test graph samples:      {len(split['test_unseen'])}")

    example_assignment = split["development_assignments"][0]
    example_local = split["local_split"][example_assignment]
    print("\nExample development assignment:")
    print(f"  assignment ID:              {example_assignment}")
    print(f"  train realizations:         {example_local['train']}")
    print(
        f"  validation_seen realizations:{example_local['validation_seen']}"
    )
    print(f"  test_seen realizations:     {example_local['test_seen']}")

    print("\nExample unseen-validation assignment:")
    print(f"  assignment ID:              {split['unseen_validation_assignments'][0]}")
    print("  all realizations go only to validation_unseen")

    print("\nExample unseen-test assignment:")
    print(f"  assignment ID:              {split['unseen_test_assignments'][0]}")
    print("  all realizations go only to test_unseen")


def resolve_input_feature_count(
    features: Sequence[Mapping[str, Any]],
    input_feature_count: Optional[int],
) -> Tuple[int, int]:
    """Return (selected_count, available_count) for dynamic feature inputs.

    Omitting ``input_feature_count`` uses every stored column. Passing 9 for a
    14-column dataset selects the original first nine features.
    """
    if not features:
        raise ValueError("Cannot resolve feature count from an empty dataset.")

    first = np.asarray(features[0]["features"])
    if first.ndim != 2:
        raise ValueError(
            f"Each feature matrix must have shape (nodes, features); found {first.shape}."
        )

    available = int(first.shape[1])
    if available < 1:
        raise ValueError("The saved feature matrices contain zero feature columns.")

    selected = available if input_feature_count is None else int(input_feature_count)
    if selected < 1:
        raise ValueError("input_feature_count must be at least 1 when supplied.")
    if selected > available:
        raise ValueError(
            f"Requested {selected} input features, but the dataset contains only "
            f"{available}."
        )

    if selected == available:
        print(f"Input feature columns: using all {available} stored columns.")
    else:
        print(
            f"Input feature columns: using the first {selected} of {available} "
            "stored columns."
        )
    return selected, available


def compute_training_feature_normalization(
    features: Sequence[Mapping[str, Any]],
    train_indices: Sequence[int],
    *,
    input_feature_count: int,
    available_feature_count: int,
    epsilon: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-column mean/std from training nodes only.

    Only the selected leading feature columns are used. Non-finite entries are
    excluded from the statistics. When normalization is applied, a non-finite
    value is replaced with its training-column mean, so its normalized value
    becomes zero.
    """
    if not train_indices:
        raise ValueError("Cannot normalize features from an empty training split.")
    if epsilon <= 0.0:
        raise ValueError("normalization_epsilon must be positive.")

    num_features = int(input_feature_count)
    sums = np.zeros(num_features, dtype=np.float64)
    sums_squared = np.zeros(num_features, dtype=np.float64)
    counts = np.zeros(num_features, dtype=np.int64)

    for sample_index in tqdm(
        train_indices,
        desc="Computing train feature statistics",
        leave=False,
    ):
        full_values = np.asarray(
            features[sample_index]["features"], dtype=np.float64
        )
        if full_values.ndim != 2 or full_values.shape[1] != available_feature_count:
            raise ValueError(
                f"Inconsistent feature shape at sample {sample_index}: expected "
                f"(*, {available_feature_count}), found {full_values.shape}."
            )
        values = full_values[:, :num_features]

        finite = np.isfinite(values)
        safe_values = np.where(finite, values, 0.0)
        sums += safe_values.sum(axis=0)
        sums_squared += np.square(safe_values).sum(axis=0)
        counts += finite.sum(axis=0)

    if np.any(counts == 0):
        bad_columns = np.where(counts == 0)[0].tolist()
        raise ValueError(
            f"Training data has no finite values in feature columns {bad_columns}."
        )

    means = sums / counts
    variances = np.maximum(sums_squared / counts - np.square(means), 0.0)
    stds = np.sqrt(variances)

    constant_columns = stds < epsilon
    if np.any(constant_columns):
        print(
            "Warning: constant/nearly constant training feature columns "
            f"{np.where(constant_columns)[0].tolist()} will use std=1."
        )
        stds[constant_columns] = 1.0

    print("Feature normalization fitted on training samples only.")
    print("  mean:", np.array2string(means, precision=6, separator=", "))
    print("  std: ", np.array2string(stds, precision=6, separator=", "))
    return means.astype(np.float32), stds.astype(np.float32)


def create_pyg_data(
    features: Sequence[Mapping[str, Any]],
    graph: nx.Graph,
    *,
    input_feature_count: int,
    available_feature_count: int,
    features_per_assignment: int,
    selected_source_assignment_ids: Sequence[int],
    feature_mean: Optional[np.ndarray] = None,
    feature_std: Optional[np.ndarray] = None,
) -> Tuple[List[Data], int]:
    edge_array = np.asarray(list(graph.edges()), dtype=np.int64)
    if edge_array.size == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.as_tensor(edge_array, dtype=torch.long).t().contiguous()

    if (feature_mean is None) != (feature_std is None):
        raise ValueError("feature_mean and feature_std must be provided together.")
    if features_per_assignment < 1:
        raise ValueError("features_per_assignment must be at least 1.")

    expected_assignments = (len(features) + features_per_assignment - 1) // features_per_assignment
    if expected_assignments != len(selected_source_assignment_ids):
        raise ValueError(
            "Selected source-assignment IDs do not match the loaded feature layout: "
            f"expected {expected_assignments}, found "
            f"{len(selected_source_assignment_ids)}."
        )

    all_data: List[Data] = []
    for sample_index, feature_dict in enumerate(features):
        full_feature_array = np.asarray(
            feature_dict["features"], dtype=np.float32
        )
        if (
            full_feature_array.ndim != 2
            or full_feature_array.shape[1] != available_feature_count
        ):
            raise ValueError(
                f"Inconsistent feature shape at sample {sample_index}: expected "
                f"(*, {available_feature_count}), found {full_feature_array.shape}."
            )
        feature_array = full_feature_array[:, :input_feature_count]

        if feature_mean is not None and feature_std is not None:
            if feature_array.shape[1] != feature_mean.shape[0]:
                raise ValueError(
                    "Feature-normalization dimensions do not match the selected data."
                )
            feature_array = np.where(
                np.isfinite(feature_array), feature_array, feature_mean
            )
            feature_array = (feature_array - feature_mean) / feature_std
        elif not np.all(np.isfinite(feature_array)):
            raise ValueError(
                "Non-finite feature values found. Enable --normalize_features=True "
                "to impute them with training-column means before normalization."
            )

        local_assignment_id = sample_index // features_per_assignment
        realization_id = sample_index % features_per_assignment
        source_assignment_id = int(
            selected_source_assignment_ids[local_assignment_id]
        )

        x = torch.as_tensor(feature_array, dtype=torch.float32)
        y = torch.as_tensor(feature_dict["labels"], dtype=torch.float32)
        all_data.append(
            Data(
                x=x,
                edge_index=edge_index,
                y=y,
                assignment_id=torch.tensor(
                    [local_assignment_id], dtype=torch.long
                ),
                source_assignment_id=torch.tensor(
                    [source_assignment_id], dtype=torch.long
                ),
                realization_id=torch.tensor([realization_id], dtype=torch.long),
            )
        )

    return all_data, int(input_feature_count)


def create_loaders(
    all_data: Sequence[Data],
    split: Mapping[str, Any],
    *,
    batch_size: int,
    train_seed: int,
) -> Dict[str, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(train_seed)

    keys = (
        "train",
        "validation_seen",
        "validation_unseen",
        "test_seen",
        "test_unseen",
    )
    datasets = {key: [all_data[index] for index in split[key]] for key in keys}

    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        ),
        "validation_seen": DataLoader(
            datasets["validation_seen"], batch_size=batch_size, shuffle=False
        ),
        "validation_unseen": DataLoader(
            datasets["validation_unseen"], batch_size=batch_size, shuffle=False
        ),
        "test_seen": DataLoader(
            datasets["test_seen"], batch_size=batch_size, shuffle=False
        ),
        "test_unseen": DataLoader(
            datasets["test_unseen"], batch_size=batch_size, shuffle=False
        ),
        # Batch size 1 preserves one graph realization and its assignment ID,
        # which is required for grouping empirical uncertainty by assignment.
        "test_seen_uncertainty": DataLoader(
            datasets["test_seen"], batch_size=1, shuffle=False
        ),
        "test_unseen_uncertainty": DataLoader(
            datasets["test_unseen"], batch_size=1, shuffle=False
        ),
    }


def _validation_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    model.eval()
    total_loss = 0.0
    total_values = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            labels = batch.y.unsqueeze(1)
            predictions = model(batch.x, batch.edge_index).clamp(0.0, 1.0)
            loss = criterion(predictions, labels)
            count = labels.numel()
            total_loss += float(loss.item()) * count
            total_values += count

    if total_values == 0:
        raise ValueError("Validation loader is empty.")
    return total_loss / total_values


def train_model_dual_validation(
    model: nn.Module,
    criterion: nn.Module,
    train_loader: DataLoader,
    validation_seen_loader: DataLoader,
    validation_unseen_loader: DataLoader,
    *,
    num_epochs: int,
    patience: int,
    min_epochs: int,
    device: torch.device,
    lr: float,
    weight_decay: float,
    seen_validation_weight: float,
    min_delta: float = 0.0,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """Train and select a checkpoint using seen + unseen validation losses."""
    if num_epochs < 1:
        raise ValueError("num_epochs must be at least 1.")
    if min_epochs < 1 or min_epochs > num_epochs:
        raise ValueError("min_epochs must be between 1 and num_epochs.")
    if patience < 1:
        raise ValueError("patience must be at least 1.")
    if not 0.0 <= seen_validation_weight <= 1.0:
        raise ValueError("seen_validation_weight must be between 0 and 1.")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    best_score = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    best_seen_loss = float("inf")
    best_unseen_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        total_train_values = 0

        for batch in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            leave=False,
        ):
            batch = batch.to(device)
            labels = batch.y.unsqueeze(1)

            optimizer.zero_grad()
            predictions = model(batch.x, batch.edge_index)
            loss = criterion(predictions, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            count = labels.numel()
            total_train_loss += float(loss.item()) * count
            total_train_values += count

        train_loss = total_train_loss / max(1, total_train_values)
        validation_seen_loss = _validation_loss(
            model, validation_seen_loader, device, criterion
        )
        validation_unseen_loss = _validation_loss(
            model, validation_unseen_loader, device, criterion
        )
        selection_loss = (
            seen_validation_weight * validation_seen_loss
            + (1.0 - seen_validation_weight) * validation_unseen_loss
        )

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Seen: {validation_seen_loss:.5f} | "
            f"Val Unseen: {validation_unseen_loss:.5f} | "
            f"Selection: {selection_loss:.5f}"
        )

        if selection_loss < best_score - min_delta:
            best_score = selection_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            best_seen_loss = validation_seen_loss
            best_unseen_loss = validation_unseen_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch + 1 >= min_epochs and patience_counter >= patience:
            print(
                f"Early stopping at epoch {epoch + 1}; "
                f"best epoch was {best_epoch}."
            )
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a best model state.")

    info = {
        "best_epoch": float(best_epoch),
        "best_selection_loss": float(best_score),
        "best_validation_seen_loss": float(best_seen_loss),
        "best_validation_unseen_loss": float(best_unseen_loss),
    }
    return best_state, info


def evaluation_to_dict(result: Tuple[Any, ...]) -> Dict[str, Any]:
    loss, l1, acc_01, acc_02, predictions, labels = result
    return {
        "loss": float(loss),
        "l1": float(l1),
        "acc@0.1": float(acc_01),
        "acc@0.2": float(acc_02),
        "predictions": np.asarray(predictions).reshape(-1),
        "labels": np.asarray(labels).reshape(-1),
    }


def combine_evaluations(
    evaluations: Sequence[Mapping[str, Any]],
    criterion: nn.Module,
) -> Dict[str, float]:
    predictions = np.concatenate([item["predictions"] for item in evaluations])
    labels = np.concatenate([item["labels"] for item in evaluations])

    pred_tensor = torch.as_tensor(predictions, dtype=torch.float32).unsqueeze(1)
    label_tensor = torch.as_tensor(labels, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        loss = float(criterion(pred_tensor, label_tensor).item())
        l1 = float(nn.L1Loss()(pred_tensor, label_tensor).item())

    error = np.abs(predictions - labels)
    return {
        "loss": loss,
        "l1": l1,
        "acc@0.1": float(np.mean(error < 0.1)),
        "acc@0.2": float(np.mean(error < 0.2)),
    }


def predict_uncertainty_by_assignment(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    confidence: float = 1.96,
) -> Dict[str, Dict[str, Any]]:
    """Compute empirical prediction-stability intervals by assignment.

    For each source assignment and node, predictions are grouped over the test
    realizations belonging to that assignment. The interval is
    ``mean +/- confidence * std / sqrt(n)``. This measures empirical stability
    across simulated feature realizations; it is not a formal coverage guarantee.
    """
    if confidence <= 0.0:
        raise ValueError("uncertainty_confidence must be positive.")

    model.eval()
    grouped_predictions: Dict[int, List[np.ndarray]] = {}
    grouped_local_ids: Dict[int, int] = {}

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions = model(batch.x, batch.edge_index).clamp(0.0, 1.0)
            prediction_array = predictions.reshape(-1).detach().cpu().numpy()

            source_assignment_id = int(
                batch.source_assignment_id.reshape(-1)[0].item()
            )
            local_assignment_id = int(batch.assignment_id.reshape(-1)[0].item())
            grouped_predictions.setdefault(source_assignment_id, []).append(
                prediction_array
            )
            grouped_local_ids[source_assignment_id] = local_assignment_id

    results: Dict[str, Dict[str, Any]] = {}
    for source_assignment_id, prediction_list in grouped_predictions.items():
        prediction_matrix = np.stack(prediction_list, axis=0)
        mean_prediction = np.mean(prediction_matrix, axis=0)

        if prediction_matrix.shape[0] > 1:
            prediction_std = np.std(prediction_matrix, axis=0, ddof=1)
        else:
            prediction_std = np.zeros_like(mean_prediction)

        n_realizations = int(prediction_matrix.shape[0])
        standard_error = prediction_std / np.sqrt(max(1, n_realizations))
        ci_lower = np.clip(
            mean_prediction - confidence * standard_error, 0.0, 1.0
        )
        ci_upper = np.clip(
            mean_prediction + confidence * standard_error, 0.0, 1.0
        )

        results[str(source_assignment_id)] = {
            "source_assignment_id": int(source_assignment_id),
            "local_assignment_id": int(
                grouped_local_ids[source_assignment_id]
            ),
            "n_features": n_realizations,
            "mean": mean_prediction.tolist(),
            "std": prediction_std.tolist(),
            "standard_error": standard_error.tolist(),
            "ci_lower": ci_lower.tolist(),
            "ci_upper": ci_upper.tolist(),
        }

    return results


def combine_uncertainty_assignments(
    *groups: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    combined: Dict[str, Dict[str, Any]] = {}
    for group in groups:
        overlap = set(combined).intersection(group)
        if overlap:
            raise ValueError(
                "Seen and unseen uncertainty groups unexpectedly share source "
                f"assignment IDs: {sorted(overlap)[:5]}"
            )
        combined.update({str(key): dict(value) for key, value in group.items()})
    return combined


def summarize_uncertainty(
    uncertainty_by_assignment: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Summarize interval widths and prediction standard deviations."""
    all_widths: List[float] = []
    all_stds: List[float] = []

    for assignment_result in uncertainty_by_assignment.values():
        lower = np.asarray(assignment_result["ci_lower"], dtype=float)
        upper = np.asarray(assignment_result["ci_upper"], dtype=float)
        std = np.asarray(assignment_result["std"], dtype=float)
        all_widths.extend((upper - lower).tolist())
        all_stds.extend(std.tolist())

    if not all_widths:
        return {
            "n_assignments": int(len(uncertainty_by_assignment)),
            "mean_ci_width": None,
            "median_ci_width": None,
            "mean_prediction_std": None,
            "median_prediction_std": None,
        }

    return {
        "n_assignments": int(len(uncertainty_by_assignment)),
        "mean_ci_width": float(np.mean(all_widths)),
        "median_ci_width": float(np.median(all_widths)),
        "mean_prediction_std": float(np.mean(all_stds)),
        "median_prediction_std": float(np.median(all_stds)),
    }


def print_uncertainty_summary(
    label: str, summary: Mapping[str, Any]
) -> None:
    def display(value: Any) -> str:
        return "N/A" if value is None else f"{float(value):.5f}"

    print(
        f"{label:<18} | "
        f"Assignments: {int(summary['n_assignments']):5d}; "
        f"Mean CI width: {display(summary['mean_ci_width'])}; "
        f"Median CI width: {display(summary['median_ci_width'])}; "
        f"Mean pred std: {display(summary['mean_prediction_std'])}; "
        f"Median pred std: {display(summary['median_prediction_std'])}"
    )


def save_json_atomic(payload: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _safe_filename_component(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in value
    )


def infer_cascade_shard_directory(features_path: str) -> str:
    """Infer the sibling cascade-shard directory used by the simulator."""
    absolute_features_path = os.path.abspath(features_path)
    feature_stem = os.path.splitext(os.path.basename(absolute_features_path))[0]
    if feature_stem.startswith("features_"):
        dataset_stem = feature_stem[len("features_") :]
    else:
        dataset_stem = feature_stem
    return os.path.join(
        os.path.dirname(absolute_features_path),
        f"cascades_{dataset_stem}",
    )


def save_combined_test_manifest(
    *,
    name: str,
    split: Mapping[str, Any],
    selected_source_assignment_ids: Sequence[int],
    source_number_of_assignments: int,
    number_of_assignments: int,
    features_per_assignment: int,
    subset_seed: int,
    split_seed: int,
    development_fraction: float,
    unseen_validation_fraction: float,
    seen_validation_fraction: float,
    seen_test_fraction: float,
    graph_name: str,
    node_count: int,
    connections_per_node: Optional[int],
    assignemnt_generator_type: str,
    seed_percentage: float,
    num_cascades: int,
    alpha: float,
    p1: float,
    p2: float,
    q_val: float,
    features_path: str,
    test_manifest_folder: str,
    cascade_shard_folder: Optional[str],
    verify_cascade_shards: bool,
) -> str:
    """Save IDs linking this subset's combined test samples to master cascades.

    The simulator stores every realization for each source assignment in one
    cascade shard.  This manifest does not copy cascade arrays; it records the
    source assignment and local realization needed to retrieve each exact test
    sample.  The combined ordering matches evaluation: test_seen followed by
    test_unseen.
    """
    if features_per_assignment < 1:
        raise ValueError("features_per_assignment must be at least 1.")
    if len(selected_source_assignment_ids) != number_of_assignments:
        raise ValueError(
            "selected_source_assignment_ids does not match number_of_assignments."
        )

    feature_shard_directory = os.path.splitext(os.path.abspath(features_path))[0]
    if cascade_shard_folder is None:
        cascade_directory = infer_cascade_shard_directory(features_path)
    else:
        cascade_directory = os.path.abspath(cascade_shard_folder)

    combined_test_indices = list(split["test_seen"]) + list(split["test_unseen"])
    samples: List[Dict[str, Any]] = []
    unique_source_assignments = set()

    for benchmark_sample_id, selected_sample_index in enumerate(
        combined_test_indices
    ):
        selected_sample_index = int(selected_sample_index)
        local_assignment_id = selected_sample_index // features_per_assignment
        local_feature_id = selected_sample_index % features_per_assignment
        source_assignment_id = int(
            selected_source_assignment_ids[local_assignment_id]
        )
        unique_source_assignments.add(source_assignment_id)

        samples.append(
            {
                "benchmark_sample_id": int(benchmark_sample_id),
                "selected_subset_sample_index": selected_sample_index,
                "local_assignment_id": int(local_assignment_id),
                "source_assignment_id": source_assignment_id,
                "local_feature_id": int(local_feature_id),
                "cascade_realization_index": int(local_feature_id),
                "feature_shard": os.path.join(
                    feature_shard_directory,
                    f"assignment_{source_assignment_id}_features.pkl",
                ),
                "cascade_shard": os.path.join(
                    cascade_directory,
                    f"assignment_{source_assignment_id}_cascades.npz",
                ),
            }
        )

    missing_cascade_shards: List[str] = []
    if verify_cascade_shards:
        for source_assignment_id in sorted(unique_source_assignments):
            shard_path = os.path.join(
                cascade_directory,
                f"assignment_{source_assignment_id}_cascades.npz",
            )
            if not os.path.isfile(shard_path):
                missing_cascade_shards.append(shard_path)

        if missing_cascade_shards:
            preview = "\n".join(missing_cascade_shards[:10])
            raise FileNotFoundError(
                "Cascade verification failed. Missing test-assignment shards "
                f"({len(missing_cascade_shards)} total). First paths:\n{preview}"
            )

    payload = {
        "format_version": 1,
        "experiment": name,
        "purpose": (
            "Combined benchmark test IDs for retrieving exact raw cascades "
            "from the reusable master simulation."
        ),
        "master_dataset": {
            "source_number_of_assignments": int(source_number_of_assignments),
            "features_per_assignment": int(features_per_assignment),
            "num_cascades_per_seed": int(num_cascades),
            "graph_name": graph_name,
            "node_count_argument": int(node_count),
            "connections_per_node": (
                None
                if connections_per_node is None
                else int(connections_per_node)
            ),
            "assignment_generator_type": assignemnt_generator_type,
            "seed_percentage": float(seed_percentage),
            "alpha": float(alpha),
            "p1": float(p1),
            "p2": float(p2),
            "q_val": float(q_val),
            "features_path": os.path.abspath(features_path),
            "feature_shard_directory": feature_shard_directory,
            "cascade_shard_directory": cascade_directory,
            "cascade_shard_directory_exists": os.path.isdir(cascade_directory),
        },
        "subset_and_split": {
            "number_of_assignments": int(number_of_assignments),
            "subset_seed": int(subset_seed),
            "split_seed": int(split_seed),
            "development_fraction": float(development_fraction),
            "unseen_validation_fraction": float(unseen_validation_fraction),
            "seen_validation_fraction": float(seen_validation_fraction),
            "seen_test_fraction": float(seen_test_fraction),
            "combined_test_order": (
                "test_seen evaluation order followed by test_unseen "
                "evaluation order"
            ),
        },
        "test_sample_count": int(len(samples)),
        "test_source_assignment_count": int(len(unique_source_assignments)),
        "samples": samples,
    }

    safe_name = _safe_filename_component(name)
    output_path = os.path.abspath(
        os.path.join(test_manifest_folder, f"{safe_name}_test_manifest.json")
    )
    save_json_atomic(payload, output_path)

    print("\n========== COMBINED TEST MANIFEST ==========")
    print(f"Manifest:                    {output_path}")
    print(f"Combined test samples:       {len(samples)}")
    print(f"Source assignments involved: {len(unique_source_assignments)}")
    print(f"Cascade shard directory:     {cascade_directory}")
    print(f"Cascade verification:        {verify_cascade_shards}")
    print("================================================")
    return output_path


def print_metrics(label: str, metrics: Mapping[str, Any]) -> None:
    print(
        f"{label:<18} | "
        f"Loss: {metrics['loss']:.5f}; "
        f"L1: {metrics['l1']:.5f}; "
        f"Acc@0.1: {metrics['acc@0.1']:.4f}; "
        f"Acc@0.2: {metrics['acc@0.2']:.4f}"
    )


def compact_metrics(metrics: Mapping[str, Any]) -> Dict[str, float]:
    return {
        "loss": float(metrics["loss"]),
        "l1": float(metrics["l1"]),
        "acc@0.1": float(metrics["acc@0.1"]),
        "acc@0.2": float(metrics["acc@0.2"]),
    }


def build_k_run_uncertainty_summary(
    all_results: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    partitions = (
        "uncertainty_test_unseen",
    )
    metrics = (
        "mean_ci_width",
        "median_ci_width",
        "mean_prediction_std",
        "median_prediction_std",
    )

    for partition in partitions:
        if not all_results or partition not in all_results[0]:
            continue
        partition_summary: Dict[str, Any] = {
            "n_assignments": int(all_results[0][partition]["n_assignments"])
        }
        for metric in metrics:
            values = np.asarray(
                [result_item[partition][metric] for result_item in all_results],
                dtype=float,
            )
            partition_summary[f"mean_{metric}"] = float(np.mean(values))
            partition_summary[f"std_{metric}"] = float(np.std(values))
        result[partition] = partition_summary
    return result


def print_k_run_summary(all_results: Sequence[Mapping[str, Any]]) -> None:
    print("\n========== K-RUN SUMMARY ==========")
    partitions = (
        "train",
        "validation_seen",
        "validation_unseen",
        "test_unseen",
    )
    for partition in partitions:
        print(f"\n{partition}")
        for metric in ("loss", "l1", "acc@0.1", "acc@0.2"):
            values = np.asarray(
                [result[partition][metric] for result in all_results], dtype=float
            )
            print(f"  {metric:<8}: {np.mean(values):.5f} +- {np.std(values):.5f}")

    uncertainty_summary = build_k_run_uncertainty_summary(all_results)
    if uncertainty_summary:
        print("\n========== K-RUN UNCERTAINTY SUMMARY ==========")
        for partition, summary in uncertainty_summary.items():
            print(f"\n{partition}")
            print(f"  assignments: {summary['n_assignments']}")
            for metric in (
                "mean_ci_width",
                "median_ci_width",
                "mean_prediction_std",
                "median_prediction_std",
            ):
                print(
                    f"  {metric:<24}: "
                    f"{summary[f'mean_{metric}']:.5f} +- "
                    f"{summary[f'std_{metric}']:.5f}"
                )


def main(
    name: str = "hybrid_random_split",
    k_runs: int = 5,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    split_seed: int = 0,
    train_seed_start: int = 1000,
    number_of_assignments: int = 5000,
    source_number_of_assignments: Optional[int] = None,
    subset_seed: int = 0,
    num_cascades: int = 300,
    alpha: float = 0.8,
    p1: float = 0.0,
    p2: float = 0.0,
    q_val: float = 1.0,
    features_per_assignment: int = 10,
    node_count: int = 100,
    connections_per_node: Optional[int] = 2,
    batch_size: int = 8,
    num_epochs: int = 200,
    patience: int = 20,
    min_epochs: int = 50,
    model_name: str = "rggcn",
    criterion_name: str = "smooth_l1",
    device: str = "cuda:0",
    assignemnt_generator_type: str = "multiple_new_features",
    graph_name: str = "random",
    seed_percentage: float = 1.0,
    development_fraction: float = 0.80,
    unseen_validation_fraction: float = 0.10,
    seen_validation_fraction: float = 0.10,
    seen_test_fraction: float = 0.10,
    seen_validation_weight: float = 0.80,
    min_delta: float = 0.0,
    # Compatibility aliases from earlier commands.
    train_assignment_fraction: Optional[float] = None,
    validation_assignment_fraction: Optional[float] = None,
    validation_fraction: Optional[float] = None,
    within_assignment_train_fraction: Optional[float] = None,
    features_folder: str = "/scratch/svc_td_fincomp/vrango/hic_new/assignment_split/features",
    graph_folder: str = "data/graphs",
    features_path: Optional[str] = None,
    graph_path: Optional[str] = None,
    verify_assignment_layout: bool = True,
    normalize_features: bool = False,
    normalization_epsilon: float = 1e-8,
    input_feature_count: Optional[int] = None,
    compute_uncertainty: bool = True,
    uncertainty_confidence: float = 1.96,
    save_uncertainty_results: bool = False,
    uncertainty_results_folder: str = "results",
    save_test_manifest: bool = True,
    test_manifest_folder: str = "benchmark_test_manifests",
    cascade_shard_folder: Optional[str] = None,
    verify_cascade_shards: bool = False,
    manifest_only: bool = False,
    **model_kwargs: Any,
) -> None:
    """Load saved data, create five splits, train, and print metrics only."""
    if k_runs < 1:
        raise ValueError("k_runs must be at least 1.")
    if source_number_of_assignments is None:
        source_number_of_assignments = number_of_assignments
    source_number_of_assignments = int(source_number_of_assignments)
    if not 0.0 < seed_percentage <= 1.0:
        raise ValueError("seed_percentage must be in (0, 1].")
    if number_of_assignments > source_number_of_assignments:
        raise ValueError(
            "number_of_assignments cannot exceed source_number_of_assignments."
        )
    if uncertainty_confidence <= 0.0:
        raise ValueError("uncertainty_confidence must be positive.")
    if manifest_only and not save_test_manifest:
        raise ValueError(
            "manifest_only=True requires save_test_manifest=True."
        )

    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Seed percentage:             {seed_percentage:.6g}")
    print(f"Master assignment count:     {source_number_of_assignments}")
    print(f"Training assignment count:   {number_of_assignments}")
    print(f"Nested subset seed:          {subset_seed}")

    # Never forward split/training-control arguments to the model constructor.
    protected_keys = (
        "train_assignment_fraction",
        "validation_assignment_fraction",
        "validation_fraction",
        "within_assignment_train_fraction",
        "development_fraction",
        "unseen_validation_fraction",
        "seen_validation_fraction",
        "seen_test_fraction",
        "seen_validation_weight",
        "min_epochs",
        "min_delta",
        "normalize_features",
        "normalization_epsilon",
        "save_test_manifest",
        "test_manifest_folder",
        "cascade_shard_folder",
        "verify_cascade_shards",
        "manifest_only",
    )
    leaked = {key: model_kwargs.pop(key, None) for key in protected_keys}

    if train_assignment_fraction is None:
        train_assignment_fraction = leaked["train_assignment_fraction"]
    if validation_assignment_fraction is None:
        validation_assignment_fraction = leaked["validation_assignment_fraction"]
    if validation_fraction is None:
        validation_fraction = leaked["validation_fraction"]
    if within_assignment_train_fraction is None:
        within_assignment_train_fraction = leaked["within_assignment_train_fraction"]

    if train_assignment_fraction is not None:
        development_fraction = float(train_assignment_fraction)
    if validation_assignment_fraction is not None:
        unseen_validation_fraction = float(validation_assignment_fraction)
    if validation_fraction is not None:
        unseen_validation_fraction = float(validation_fraction)

    if within_assignment_train_fraction is not None:
        within_train = float(within_assignment_train_fraction)
        if not 0.0 < within_train < 1.0:
            raise ValueError(
                "within_assignment_train_fraction must be between 0 and 1."
            )
        held_out = 1.0 - within_train
        current_total = seen_validation_fraction + seen_test_fraction
        if current_total <= 0.0:
            seen_validation_fraction = held_out / 2.0
            seen_test_fraction = held_out / 2.0
        else:
            seen_validation_fraction = (
                held_out * seen_validation_fraction / current_total
            )
            seen_test_fraction = held_out * seen_test_fraction / current_total

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Experiment: {name}")
    print(f"Device:     {device_obj}")
    output_items: List[str] = []
    if save_uncertainty_results:
        output_items.append(
            "uncertainty JSON files in "
            f"{os.path.abspath(uncertainty_results_folder)}"
        )
    if save_test_manifest:
        output_items.append(
            "combined-test manifest in "
            f"{os.path.abspath(test_manifest_folder)}"
        )
    if output_items:
        print("Output mode: console plus " + " and ".join(output_items))
    else:
        print("Output mode: console only; no files are written.")
    print(
        "Checkpoint selection: "
        f"{seen_validation_weight:.2f} * val_seen + "
        f"{1.0 - seen_validation_weight:.2f} * val_unseen"
    )

    graph_path, features_path = build_saved_paths(
        graph_name=graph_name,
        node_count=node_count,
        connections_per_node=connections_per_node,
        number_of_assignments=source_number_of_assignments,
        num_cascades=num_cascades,
        alpha=alpha,
        p1=p1,
        p2=p2,
        q_val=q_val,
        features_per_assignment=features_per_assignment,
        assignemnt_generator_type=assignemnt_generator_type,
        seed_percentage=seed_percentage,
        features_folder=features_folder,
        graph_folder=graph_folder,
        features_path=features_path,
        graph_path=graph_path,
    )

    if manifest_only:
        selected_source_assignment_ids = _nested_assignment_ids(
            source_number_of_assignments,
            number_of_assignments,
            subset_seed,
        )
        split = create_hybrid_split(
            number_of_assignments=number_of_assignments,
            features_per_assignment=features_per_assignment,
            development_fraction=development_fraction,
            unseen_validation_fraction=unseen_validation_fraction,
            seen_validation_fraction=seen_validation_fraction,
            seen_test_fraction=seen_test_fraction,
            split_seed=split_seed,
        )
        save_combined_test_manifest(
            name=name,
            split=split,
            selected_source_assignment_ids=selected_source_assignment_ids,
            source_number_of_assignments=source_number_of_assignments,
            number_of_assignments=number_of_assignments,
            features_per_assignment=features_per_assignment,
            subset_seed=subset_seed,
            split_seed=split_seed,
            development_fraction=development_fraction,
            unseen_validation_fraction=unseen_validation_fraction,
            seen_validation_fraction=seen_validation_fraction,
            seen_test_fraction=seen_test_fraction,
            graph_name=graph_name,
            node_count=node_count,
            connections_per_node=connections_per_node,
            assignemnt_generator_type=assignemnt_generator_type,
            seed_percentage=seed_percentage,
            num_cascades=num_cascades,
            alpha=alpha,
            p1=p1,
            p2=p2,
            q_val=q_val,
            features_path=features_path,
            test_manifest_folder=test_manifest_folder,
            cascade_shard_folder=cascade_shard_folder,
            verify_cascade_shards=verify_cascade_shards,
        )
        print("Manifest-only mode finished; no feature loading or training was run.")
        return

    graph, features, selected_source_assignment_ids = load_master_assignment_subset(
        graph_path,
        features_path,
        source_number_of_assignments=source_number_of_assignments,
        number_of_assignments=number_of_assignments,
        features_per_assignment=features_per_assignment,
        subset_seed=subset_seed,
    )
    validate_assignment_layout(
        features,
        number_of_assignments,
        features_per_assignment,
        verify_labels=verify_assignment_layout,
    )

    selected_feature_count, available_feature_count = resolve_input_feature_count(
        features, input_feature_count
    )

    split = create_hybrid_split(
        number_of_assignments=number_of_assignments,
        features_per_assignment=features_per_assignment,
        development_fraction=development_fraction,
        unseen_validation_fraction=unseen_validation_fraction,
        seen_validation_fraction=seen_validation_fraction,
        seen_test_fraction=seen_test_fraction,
        split_seed=split_seed,
    )
    print_split_summary(split)

    if save_test_manifest:
        save_combined_test_manifest(
            name=name,
            split=split,
            selected_source_assignment_ids=selected_source_assignment_ids,
            source_number_of_assignments=source_number_of_assignments,
            number_of_assignments=number_of_assignments,
            features_per_assignment=features_per_assignment,
            subset_seed=subset_seed,
            split_seed=split_seed,
            development_fraction=development_fraction,
            unseen_validation_fraction=unseen_validation_fraction,
            seen_validation_fraction=seen_validation_fraction,
            seen_test_fraction=seen_test_fraction,
            graph_name=graph_name,
            node_count=node_count,
            connections_per_node=connections_per_node,
            assignemnt_generator_type=assignemnt_generator_type,
            seed_percentage=seed_percentage,
            num_cascades=num_cascades,
            alpha=alpha,
            p1=p1,
            p2=p2,
            q_val=q_val,
            features_path=features_path,
            test_manifest_folder=test_manifest_folder,
            cascade_shard_folder=cascade_shard_folder,
            verify_cascade_shards=verify_cascade_shards,
        )

    feature_mean: Optional[np.ndarray] = None
    feature_std: Optional[np.ndarray] = None
    if normalize_features:
        feature_mean, feature_std = compute_training_feature_normalization(
            features,
            split["train"],
            input_feature_count=selected_feature_count,
            available_feature_count=available_feature_count,
            epsilon=normalization_epsilon,
        )
    else:
        print("Feature normalization: disabled")

    all_data, num_features = create_pyg_data(
        features,
        graph,
        input_feature_count=selected_feature_count,
        available_feature_count=available_feature_count,
        features_per_assignment=features_per_assignment,
        selected_source_assignment_ids=selected_source_assignment_ids,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    print(f"Model input dimension:          {num_features}")
    criterion = make_criterion(criterion_name, beta=0.1)
    eval_l1 = nn.L1Loss()
    all_results: List[Dict[str, Any]] = []

    for k in range(k_runs):
        print(f"\n========== RUN {k + 1}/{k_runs} ==========")
        train_seed = train_seed_start + k
        print(f"Training seed: {train_seed}")
        set_seed(train_seed)

        loaders = create_loaders(
            all_data, split, batch_size=batch_size, train_seed=train_seed
        )
        model = create_model(model_name, num_features, device_obj, model_kwargs)

        best_model_state, best_info = train_model_dual_validation(
            model,
            criterion,
            loaders["train"],
            loaders["validation_seen"],
            loaders["validation_unseen"],
            num_epochs=num_epochs,
            patience=patience,
            min_epochs=min_epochs,
            device=device_obj,
            lr=lr,
            weight_decay=weight_decay,
            seen_validation_weight=seen_validation_weight,
            min_delta=min_delta,
        )
        model.load_state_dict(best_model_state)
        print(
            "Selected checkpoint | "
            f"epoch: {int(best_info['best_epoch'])}; "
            f"score: {best_info['best_selection_loss']:.5f}; "
            f"val seen: {best_info['best_validation_seen_loss']:.5f}; "
            f"val unseen: {best_info['best_validation_unseen_loss']:.5f}"
        )

        train_eval = evaluation_to_dict(
            evaluate_model(model, loaders["train"], device_obj, criterion, eval_l1)
        )
        validation_seen_eval = evaluation_to_dict(
            evaluate_model(
                model,
                loaders["validation_seen"],
                device_obj,
                criterion,
                eval_l1,
            )
        )
        validation_unseen_eval = evaluation_to_dict(
            evaluate_model(
                model,
                loaders["validation_unseen"],
                device_obj,
                criterion,
                eval_l1,
            )
        )
        
        test_unseen_eval = evaluation_to_dict(
            evaluate_model(
                model, loaders["test_unseen"], device_obj, criterion, eval_l1
            )
        )

        # ---------- Prediction-collapse diagnostic ----------
        pred = test_unseen_eval["predictions"]
        true = test_unseen_eval["labels"]

        correlation = (
            np.corrcoef(true, pred)[0, 1]
            if np.std(true) > 0 and np.std(pred) > 0
            else float("nan")
        )

        print("\n========== TEST-UNSEEN PREDICTION DIAGNOSTICS ==========")
        print(f"True mean:       {np.mean(true):.5f}")
        print(f"True std:        {np.std(true):.5f}")
        print(f"Prediction mean: {np.mean(pred):.5f}")
        print(f"Prediction std:  {np.std(pred):.5f}")
        print(f"Correlation:     {correlation:.5f}")
        print(f"Prediction min:  {np.min(pred):.5f}")
        print(f"Prediction max:  {np.max(pred):.5f}")
        # --------------------------------------------------------

        # ---------- Seed versus non-seed diagnostics ----------
        node_order = list(features[0]["nodes"])
        num_nodes = len(node_order)

        # Same seed-selection rule used during simulation:
        # highest-degree nodes, with stable graph-order tie breaking.
        number_of_seeds = int(seed_percentage * num_nodes)
        selected_seeds = sorted(
            list(graph.nodes()),
            key=lambda node: graph.degree[node],
            reverse=True,
        )[:number_of_seeds]

        seed_set = set(selected_seeds)
        seed_mask = np.asarray(
            [node in seed_set for node in node_order],
            dtype=bool,
        )
        nonseed_mask = ~seed_mask

        # Convert flattened arrays into:
        # number of test samples × number of nodes
        pred_matrix = np.asarray(pred).reshape(-1, num_nodes)
        true_matrix = np.asarray(true).reshape(-1, num_nodes)
        absolute_error = np.abs(pred_matrix - true_matrix)


        def print_group_results(group_name, mask):
            node_total = int(mask.sum())
            if node_total == 0:
                print(
                    f"{group_name:<15} | Nodes: {node_total:3d} | "
                    "L1: N/A | Acc@0.1: N/A | Acc@0.2: N/A"
                )
                return

            group_error = absolute_error[:, mask]
            print(
                f"{group_name:<15} | "
                f"Nodes: {node_total:3d} | "
                f"L1: {group_error.mean():.5f} | "
                f"Acc@0.1: {np.mean(group_error < 0.1):.4f} | "
                f"Acc@0.2: {np.mean(group_error < 0.2):.4f}"
            )


        print("\n========== SEED VERSUS NON-SEED DIAGNOSTICS ==========")
        print(f"Selected seed nodes: {selected_seeds}")

        print_group_results("Seed nodes", seed_mask)
        print_group_results("Non-seed nodes", nonseed_mask)
        # ------------------------------------------------------

        uncertainty_summaries: Dict[str, Dict[str, Any]] = {}

        if compute_uncertainty:
            test_unseen_uncertainty = predict_uncertainty_by_assignment(
                model,
                loaders["test_unseen_uncertainty"],
                device_obj,
                confidence=uncertainty_confidence,
            )

            uncertainty_summaries = {
                "uncertainty_test_unseen": summarize_uncertainty(
                    test_unseen_uncertainty
                ),
            }

            print("\n========== TEST UNCERTAINTY ==========")
            print(
                "Empirical intervals use mean +/- "
                f"{uncertainty_confidence:.4g} * std / sqrt(n)."
            )

            print_uncertainty_summary(
                "Test unseen",
                uncertainty_summaries["uncertainty_test_unseen"],
            )

            if save_uncertainty_results:
                run_payload = {
                    "script_version": SCRIPT_VERSION,
                    "experiment": name,
                    "run_index": int(k),
                    "train_seed": int(train_seed),
                    "confidence_multiplier": float(uncertainty_confidence),
                    "input_feature_count": int(num_features),
                    "available_feature_count": int(available_feature_count),
                    "test_unseen": {
                        "summary": uncertainty_summaries[
                            "uncertainty_test_unseen"
                        ],
                        "by_assignment": test_unseen_uncertainty,
                    },
                }

                run_path = os.path.join(
                    uncertainty_results_folder,
                    f"{model_name}_{name}_k{k}_hybrid_uncertainty.json",
                )
                save_json_atomic(run_payload, run_path)
                print(f"Saved uncertainty results: {run_path}")

        
        print_metrics("Train", train_eval)
        print_metrics("Validation seen", validation_seen_eval)
        print_metrics("Validation unseen", validation_unseen_eval)
        print_metrics("Test unseen", test_unseen_eval)

        run_result: Dict[str, Any] = {
            "train": compact_metrics(train_eval),
            "validation_seen": compact_metrics(validation_seen_eval),
            "validation_unseen": compact_metrics(validation_unseen_eval),
            "test_unseen": compact_metrics(test_unseen_eval),
        }
        run_result.update(uncertainty_summaries)
        all_results.append(run_result)

        del model
        if device_obj.type == "cuda":
            torch.cuda.empty_cache()

    print_k_run_summary(all_results)

    if compute_uncertainty and save_uncertainty_results:
        summary_payload = {
            "script_version": SCRIPT_VERSION,
            "experiment": name,
            "model_name": model_name,
            "k_runs": int(k_runs),
            "confidence_multiplier": float(uncertainty_confidence),
            "input_feature_count": int(num_features),
            "available_feature_count": int(available_feature_count),
            "uncertainty": build_k_run_uncertainty_summary(all_results),
            "runs": all_results,
        }
        summary_path = os.path.join(
            uncertainty_results_folder,
            f"{model_name}_{name}_K{k_runs}_hybrid_uncertainty_summary.json",
        )
        save_json_atomic(summary_payload, summary_path)
        print(f"Saved K-run uncertainty summary: {summary_path}")
        print("\nFinished. Uncertainty JSON files were written.")
    else:
        print("\nFinished. No files were written.")


if __name__ == "__main__":
    Fire(main)
