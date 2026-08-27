#!/usr/bin/env python3
"""Train SAGE-HC at varying FPA and evaluate on a fixed FPA=2 unseen set.

This diagnostic preserves the benchmark split used by the supplied FPA launcher:
  * 80% development assignments
  * 10% unseen-validation assignments
  * 10% unseen-test assignments
  * 10% of each development assignment's TRAIN-FPA realizations for seen validation
  * 0% seen test
  * checkpoint selection = 0.80 * seen validation + 0.20 * unseen validation

The original training/validation protocol is preserved at every TRAIN_FPA: unseen
validation uses all TRAIN_FPA realizations, while seen validation uses 10% of each
development assignment and checkpoint selection remains 0.80 seen + 0.20 unseen.

Only the strict unseen test set is changed for this diagnostic: it always uses exactly
FIXED_EVAL_FPA source realizations (default 2), independent of TRAIN_FPA. The same
unseen-test assignment IDs and same two source realization IDs are reused for every
training-FPA setting. Consequently, cross-training-FPA changes in test accuracy,
prediction SD, or empirical interval width are not caused by changing test FPA.
The interval is still an empirical prediction-stability interval, not a calibrated
coverage interval.

The script reuses the existing FPA=100, C=50 assignment shards and saves resumable
checkpoints plus per-run CSV and JSON summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from core import train_sensitivity as core


SCRIPT_VERSION = "fixed-test-fpa2-original-validation-v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train with one FPA using the benchmark seen/unseen validation split "
            "and evaluate strict unseen assignments at a fixed test FPA."
        )
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--train-fpa", type=int, required=True)
    parser.add_argument("--fixed-eval-fpa", type=int, default=2)
    parser.add_argument("--source-fpa", type=int, default=100)
    parser.add_argument("--number-of-assignments", type=int, default=5000)
    parser.add_argument("--source-number-of-assignments", type=int, required=True)
    parser.add_argument("--subset-seed", type=int, default=0)
    parser.add_argument("--realization-subset-seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)

    parser.add_argument("--num-cascades", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--p1", type=float, default=0.0)
    parser.add_argument("--p2", type=float, default=0.0)
    parser.add_argument("--q-val", type=float, default=1.0)
    parser.add_argument("--node-count", type=int, default=100)
    parser.add_argument("--connections-per-node", type=int, default=2)
    parser.add_argument("--graph-name", default="random")
    parser.add_argument(
        "--assignment-generator-type",
        default="multiple_new_features_self_seed_masked",
    )
    parser.add_argument("--seed-percentage", type=float, default=1.0)
    parser.add_argument("--input-feature-count", type=int, default=14)

    parser.add_argument("--development-fraction", type=float, default=0.80)
    parser.add_argument("--unseen-validation-fraction", type=float, default=0.10)
    parser.add_argument("--seen-validation-fraction", type=float, default=0.10)
    parser.add_argument("--seen-test-fraction", type=float, default=0.00)
    parser.add_argument("--seen-validation-weight", type=float, default=0.80)

    parser.add_argument("--model-name", default="rggcn")
    parser.add_argument("--criterion-name", default="smooth_l1")
    parser.add_argument("--k-runs", type=int, default=2)
    parser.add_argument("--train-seed-start", type=int, default=1000)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--min-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--uncertainty-confidence", type=float, default=1.96)

    parser.add_argument("--features-folder", type=Path, required=True)
    parser.add_argument("--graph-folder", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    # FPA=1 is intentionally excluded: a non-empty seen-validation split would
    # leave no realization for training, exactly as in the supplied launcher.
    if args.train_fpa < 2:
        raise ValueError("--train-fpa must be >= 2 when seen validation is non-empty")
    if args.fixed_eval_fpa < 2:
        raise ValueError("--fixed-eval-fpa must be >= 2 so prediction SD is estimable")
    if args.source_fpa < max(args.train_fpa, args.fixed_eval_fpa):
        raise ValueError("--source-fpa must be >= train FPA and fixed evaluation FPA")
    if args.number_of_assignments < 3:
        raise ValueError("--number-of-assignments must be >= 3")
    if args.number_of_assignments > args.source_number_of_assignments:
        raise ValueError("selected assignments cannot exceed source assignments")
    if args.k_runs < 1:
        raise ValueError("--k-runs must be >= 1")
    if args.num_epochs < 1 or args.min_epochs < 1 or args.min_epochs > args.num_epochs:
        raise ValueError("invalid epoch/min-epoch configuration")
    if args.uncertainty_confidence <= 0:
        raise ValueError("--uncertainty-confidence must be positive")
    if not 0.0 <= args.seen_validation_weight <= 1.0:
        raise ValueError("--seen-validation-weight must be in [0,1]")


def nested_realization_ids(source_fpa: int, target_fpa: int, seed: int) -> List[int]:
    rng = np.random.default_rng(seed)
    return rng.permutation(source_fpa)[:target_fpa].astype(int).tolist()


def load_graph(graph_path: Path):
    if not graph_path.is_file():
        raise FileNotFoundError(f"Graph not found: {graph_path}")
    with graph_path.open("rb") as handle:
        return pickle.load(handle)


def require_master_shards(features_path: Path) -> Path:
    shard_dir = features_path.with_suffix("")
    if not shard_dir.is_dir():
        raise FileNotFoundError(
            "This diagnostic requires assignment shards. Missing directory: "
            f"{shard_dir}"
        )
    return shard_dir


def load_group_features(
    shard_dir: Path,
    source_assignment_ids: Sequence[int],
    realization_ids: Sequence[int],
    source_fpa: int,
    label: str,
) -> List[Dict[str, Any]]:
    """Load requested source assignments and source realization IDs in order."""
    if not realization_ids:
        raise ValueError(f"No realization IDs supplied for {label}")
    if min(realization_ids) < 0 or max(realization_ids) >= source_fpa:
        raise ValueError(f"Realization IDs for {label} fall outside source FPA={source_fpa}")

    loaded: List[Dict[str, Any]] = []
    for source_id in tqdm(source_assignment_ids, desc=f"Loading {label}", leave=False):
        shard_path = shard_dir / f"assignment_{int(source_id)}_features.pkl"
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing master assignment shard: {shard_path}")
        with shard_path.open("rb") as handle:
            assignment_features = pickle.load(handle)
        if len(assignment_features) != source_fpa:
            raise ValueError(
                f"{shard_path} has {len(assignment_features)} realizations; "
                f"expected {source_fpa}"
            )

        reference_labels = np.asarray(assignment_features[int(realization_ids[0])]["labels"])
        for source_realization_id in realization_ids:
            item = assignment_features[int(source_realization_id)]
            labels = np.asarray(item["labels"])
            if labels.shape != reference_labels.shape or not np.allclose(
                labels, reference_labels, rtol=0.0, atol=1e-8
            ):
                raise ValueError(
                    f"Labels differ within source assignment {source_id} in {label}"
                )
            loaded.append(item)
    return loaded


def make_loader(data, batch_size: int, shuffle: bool, seed: int | None = None):
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(0 if seed is None else seed))
        return DataLoader(data, batch_size=batch_size, shuffle=True, generator=generator)
    return DataLoader(data, batch_size=batch_size, shuffle=False)


def metrics_from_arrays(
    predictions: np.ndarray,
    labels: np.ndarray,
    criterion: nn.Module,
) -> Dict[str, float]:
    predictions = np.asarray(predictions, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)
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


def assignment_mean_metrics(
    uncertainty_by_assignment: Mapping[str, Mapping[str, Any]],
    source_assignment_ids: Sequence[int],
    test_features: Sequence[Mapping[str, Any]],
    fixed_eval_fpa: int,
    criterion: nn.Module,
) -> Dict[str, float]:
    predictions: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    for local_id, source_id in enumerate(source_assignment_ids):
        key = str(int(source_id))
        if key not in uncertainty_by_assignment:
            raise ValueError(f"Missing grouped predictions for source assignment {source_id}")
        predictions.append(np.asarray(uncertainty_by_assignment[key]["mean"], dtype=float))
        labels.append(np.asarray(test_features[local_id * fixed_eval_fpa]["labels"], dtype=float))
    return metrics_from_arrays(np.concatenate(predictions), np.concatenate(labels), criterion)


def checkpoint_compatible(
    payload: Mapping[str, Any],
    args: argparse.Namespace,
    train_realization_ids: Sequence[int],
    eval_realization_ids: Sequence[int],
    development_source_ids: Sequence[int],
    validation_source_ids: Sequence[int],
    test_source_ids: Sequence[int],
) -> bool:
    expected = {
        "script_version": SCRIPT_VERSION,
        "train_fpa": int(args.train_fpa),
        "fixed_eval_fpa": int(args.fixed_eval_fpa),
        "validation_unseen_fpa": int(args.train_fpa),
        "source_fpa": int(args.source_fpa),
        "number_of_assignments": int(args.number_of_assignments),
        "source_number_of_assignments": int(args.source_number_of_assignments),
        "subset_seed": int(args.subset_seed),
        "split_seed": int(args.split_seed),
        "realization_subset_seed": int(args.realization_subset_seed),
        "num_cascades": int(args.num_cascades),
        "p1": float(args.p1),
        "p2": float(args.p2),
        "q_val": float(args.q_val),
        "model_name": str(args.model_name),
        "input_feature_count": int(args.input_feature_count),
        "development_fraction": float(args.development_fraction),
        "unseen_validation_fraction": float(args.unseen_validation_fraction),
        "seen_validation_fraction": float(args.seen_validation_fraction),
        "seen_test_fraction": float(args.seen_test_fraction),
        "seen_validation_weight": float(args.seen_validation_weight),
        "train_realization_ids": list(map(int, train_realization_ids)),
        "eval_realization_ids": list(map(int, eval_realization_ids)),
        "development_source_ids": list(map(int, development_source_ids)),
        "validation_source_ids": list(map(int, validation_source_ids)),
        "test_source_ids": list(map(int, test_source_ids)),
    }
    return all(payload.get(key) == value for key, value in expected.items())


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    args: argparse.Namespace,
    run_index: int,
    train_seed: int,
    best_info: Mapping[str, Any],
    train_realization_ids: Sequence[int],
    eval_realization_ids: Sequence[int],
    development_source_ids: Sequence[int],
    validation_source_ids: Sequence[int],
    test_source_ids: Sequence[int],
    seen_validation_source_realizations: Mapping[str, Sequence[int]],
) -> None:
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    payload = {
        "script_version": SCRIPT_VERSION,
        "name": args.name,
        "run_index": int(run_index),
        "train_seed": int(train_seed),
        "train_fpa": int(args.train_fpa),
        "fixed_eval_fpa": int(args.fixed_eval_fpa),
        "validation_unseen_fpa": int(args.train_fpa),
        "source_fpa": int(args.source_fpa),
        "number_of_assignments": int(args.number_of_assignments),
        "source_number_of_assignments": int(args.source_number_of_assignments),
        "subset_seed": int(args.subset_seed),
        "split_seed": int(args.split_seed),
        "realization_subset_seed": int(args.realization_subset_seed),
        "num_cascades": int(args.num_cascades),
        "p1": float(args.p1),
        "p2": float(args.p2),
        "q_val": float(args.q_val),
        "model_name": str(args.model_name),
        "input_feature_count": int(args.input_feature_count),
        "development_fraction": float(args.development_fraction),
        "unseen_validation_fraction": float(args.unseen_validation_fraction),
        "seen_validation_fraction": float(args.seen_validation_fraction),
        "seen_test_fraction": float(args.seen_test_fraction),
        "seen_validation_weight": float(args.seen_validation_weight),
        "train_realization_ids": list(map(int, train_realization_ids)),
        "eval_realization_ids": list(map(int, eval_realization_ids)),
        "development_source_ids": list(map(int, development_source_ids)),
        "validation_source_ids": list(map(int, validation_source_ids)),
        "test_source_ids": list(map(int, test_source_ids)),
        "seen_validation_source_realizations": {
            str(key): list(map(int, value))
            for key, value in seen_validation_source_realizations.items()
        },
        "feature_mean": np.asarray(feature_mean, dtype=np.float32),
        "feature_std": np.asarray(feature_std, dtype=np.float32),
        "best_info": dict(best_info),
        "state_dict": state_dict,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


def write_run_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"k_runs": len(rows)}
    metric_keys = [
        "validation_seen_loss",
        "validation_unseen_loss",
        "selection_loss",
        "test_loss",
        "test_l1",
        "test_acc01",
        "test_acc02",
        "assignment_mean_loss",
        "assignment_mean_l1",
        "assignment_mean_acc01",
        "assignment_mean_acc02",
        "mean_prediction_std",
        "median_prediction_std",
        "mean_ci_width",
        "median_ci_width",
    ]
    for key in metric_keys:
        mean, std = mean_std(float(row[key]) for row in rows)
        summary[key] = {"mean": mean, "std": std}
    return summary


def main() -> None:
    args = parse_args()
    validate_args(args)

    output_dir = args.output_dir.expanduser().resolve()
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    graph_path_str, features_path_str = core.build_saved_paths(
        graph_name=args.graph_name,
        node_count=args.node_count,
        connections_per_node=args.connections_per_node,
        number_of_assignments=args.source_number_of_assignments,
        num_cascades=args.num_cascades,
        alpha=args.alpha,
        p1=args.p1,
        p2=args.p2,
        q_val=args.q_val,
        features_per_assignment=args.source_fpa,
        assignemnt_generator_type=args.assignment_generator_type,
        seed_percentage=args.seed_percentage,
        features_folder=str(args.features_folder.expanduser().resolve()),
        graph_folder=str(args.graph_folder.expanduser().resolve()),
        features_path=None,
        graph_path=None,
    )
    graph_path = Path(graph_path_str)
    features_path = Path(features_path_str)
    shard_dir = require_master_shards(features_path)
    graph = load_graph(graph_path)

    selected_source_ids = core._nested_assignment_ids(
        args.source_number_of_assignments,
        args.number_of_assignments,
        args.subset_seed,
    )

    # Use the exact v7 hybrid split logic at TRAIN_FPA. This preserves the user's
    # benchmark protocol, including max(1, round(0.10 * FPA)) seen-validation
    # realizations and zero seen-test realizations.
    split = core.create_hybrid_split(
        number_of_assignments=args.number_of_assignments,
        features_per_assignment=args.train_fpa,
        development_fraction=args.development_fraction,
        unseen_validation_fraction=args.unseen_validation_fraction,
        seen_validation_fraction=args.seen_validation_fraction,
        seen_test_fraction=args.seen_test_fraction,
        split_seed=args.split_seed,
    )

    development_local_ids = list(map(int, split["development_assignments"]))
    validation_local_ids = list(map(int, split["unseen_validation_assignments"]))
    test_local_ids = list(map(int, split["unseen_test_assignments"]))
    development_source_ids = [selected_source_ids[index] for index in development_local_ids]
    validation_source_ids = [selected_source_ids[index] for index in validation_local_ids]
    test_source_ids = [selected_source_ids[index] for index in test_local_ids]

    train_realization_ids = nested_realization_ids(
        args.source_fpa, args.train_fpa, args.realization_subset_seed
    )
    eval_realization_ids = nested_realization_ids(
        args.source_fpa, args.fixed_eval_fpa, args.realization_subset_seed
    )

    # Preserve the original training/validation protocol: development and unseen
    # validation use TRAIN_FPA. Only strict unseen testing is forced to FPA=2.
    development_features = load_group_features(
        shard_dir,
        development_source_ids,
        train_realization_ids,
        args.source_fpa,
        "development train-FPA pool",
    )
    validation_features = load_group_features(
        shard_dir,
        validation_source_ids,
        train_realization_ids,
        args.source_fpa,
        "original-protocol unseen validation",
    )
    test_features = load_group_features(
        shard_dir,
        test_source_ids,
        eval_realization_ids,
        args.source_fpa,
        "fixed-FPA unseen test",
    )

    # Translate the v7 local split from original local assignment IDs into the
    # compact development_features layout (development position x train FPA).
    train_indices: List[int] = []
    seen_validation_indices: List[int] = []
    seen_validation_source_realizations: Dict[str, List[int]] = {}
    for dev_position, original_local_assignment_id in enumerate(development_local_ids):
        local = split["local_split"][original_local_assignment_id]
        for local_realization_id in local["train"]:
            train_indices.append(dev_position * args.train_fpa + int(local_realization_id))
        seen_src_ids: List[int] = []
        for local_realization_id in local["validation_seen"]:
            seen_validation_indices.append(
                dev_position * args.train_fpa + int(local_realization_id)
            )
            seen_src_ids.append(train_realization_ids[int(local_realization_id)])
        seen_validation_source_realizations[
            str(development_source_ids[dev_position])
        ] = seen_src_ids

    if not train_indices:
        raise ValueError("Training split is empty")
    if args.seen_validation_weight > 0.0 and not seen_validation_indices:
        raise ValueError("Seen validation is empty while seen-validation weight is positive")

    selected_feature_count, available_feature_count = core.resolve_input_feature_count(
        development_features, args.input_feature_count
    )
    feature_mean, feature_std = core.compute_training_feature_normalization(
        development_features,
        train_indices,
        input_feature_count=selected_feature_count,
        available_feature_count=available_feature_count,
        epsilon=1e-8,
    )

    development_data, num_features = core.create_pyg_data(
        development_features,
        graph,
        input_feature_count=selected_feature_count,
        available_feature_count=available_feature_count,
        features_per_assignment=args.train_fpa,
        selected_source_assignment_ids=development_source_ids,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    validation_data, _ = core.create_pyg_data(
        validation_features,
        graph,
        input_feature_count=selected_feature_count,
        available_feature_count=available_feature_count,
        features_per_assignment=args.train_fpa,
        selected_source_assignment_ids=validation_source_ids,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    test_data, _ = core.create_pyg_data(
        test_features,
        graph,
        input_feature_count=selected_feature_count,
        available_feature_count=available_feature_count,
        features_per_assignment=args.fixed_eval_fpa,
        selected_source_assignment_ids=test_source_ids,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )

    train_data = [development_data[index] for index in train_indices]
    seen_validation_data = [development_data[index] for index in seen_validation_indices]

    n_seen_per_dev = len(split["local_split"][development_local_ids[0]]["validation_seen"])
    n_train_per_dev = len(split["local_split"][development_local_ids[0]]["train"])

    print("========== FIXED TEST-FPA DIAGNOSTIC ==========")
    print(f"Script version:               {SCRIPT_VERSION}")
    print(f"Experiment:                   {args.name}")
    print(f"Training FPA:                 {args.train_fpa}")
    print(f"Unseen-validation FPA:        {args.train_fpa} (original protocol)")
    print(f"Fixed strict-test FPA:         {args.fixed_eval_fpa}")
    print(f"Source/master FPA:            {args.source_fpa}")
    print(f"Selected assignments:         {args.number_of_assignments}")
    print(f"Master assignments:           {args.source_number_of_assignments}")
    print(f"Development assignments:      {len(development_source_ids)}")
    print(f"Unseen-validation assignments:{len(validation_source_ids)}")
    print(f"Unseen-test assignments:      {len(test_source_ids)}")
    print(f"Training realizations/dev:    {n_train_per_dev}")
    print(f"Seen-validation/dev:          {n_seen_per_dev}")
    print(f"Seen-test/dev:                0")
    print(f"Checkpoint weighting:         {args.seen_validation_weight:.2f} seen + "
          f"{1.0 - args.seen_validation_weight:.2f} unseen")
    print(f"Cascades per seed:            {args.num_cascades}")
    print(f"Noise:                        p1={args.p1}, p2={args.p2}, q={args.q_val}")
    print(f"Master shard directory:       {shard_dir}")
    print(f"Train-FPA source IDs:         {train_realization_ids[:20]}"
          + (" ..." if len(train_realization_ids) > 20 else ""))
    print(f"Fixed eval source IDs:        {eval_realization_ids}")
    print(f"Train graph samples:          {len(train_data)}")
    print(f"Seen-validation samples:      {len(seen_validation_data)}")
    print(f"Unseen-validation samples:    {len(validation_data)}")
    print(f"Fixed unseen-test samples:    {len(test_data)}")
    print("Normalization is fitted on training samples only.")
    print("The original seen/unseen validation protocol is preserved at TRAIN_FPA.")
    print("Only strict unseen testing is fixed to the same two source realizations")
    print("and the same unseen-test assignment IDs at every training FPA.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    criterion = core.make_criterion(args.criterion_name, beta=0.1)
    eval_l1 = nn.L1Loss()

    run_rows: List[Dict[str, Any]] = []
    best_checkpoint: Path | None = None
    best_run_selection_loss = float("inf")

    for run_index in range(args.k_runs):
        train_seed = args.train_seed_start + run_index
        core.set_seed(train_seed)
        checkpoint_path = checkpoint_dir / (
            f"trainfpa{args.train_fpa}_testfpa{args.fixed_eval_fpa}_"
            f"k{run_index}_{args.model_name}.pt"
        )

        train_loader = make_loader(train_data, args.batch_size, shuffle=True, seed=train_seed)
        seen_validation_loader = make_loader(
            seen_validation_data, args.batch_size, shuffle=False
        )
        unseen_validation_loader = make_loader(validation_data, args.batch_size, shuffle=False)
        test_loader = make_loader(test_data, args.batch_size, shuffle=False)
        test_uncertainty_loader = make_loader(test_data, 1, shuffle=False)

        model = core.create_model(args.model_name, num_features, device, {})
        best_info: Dict[str, Any]

        resumed = False
        if args.resume and checkpoint_path.is_file():
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if checkpoint_compatible(
                payload,
                args,
                train_realization_ids,
                eval_realization_ids,
                development_source_ids,
                validation_source_ids,
                test_source_ids,
            ):
                model.load_state_dict(payload["state_dict"])
                best_info = dict(payload.get("best_info", {}))
                resumed = True
                print(f"\nRUN {run_index + 1}/{args.k_runs}: resumed {checkpoint_path.name}")
            else:
                print(
                    f"\nRUN {run_index + 1}/{args.k_runs}: checkpoint metadata mismatch; retraining"
                )

        if not resumed:
            print(f"\n========== RUN {run_index + 1}/{args.k_runs} ==========")
            print(f"Training seed: {train_seed}")
            best_state, best_info = core.train_model_dual_validation(
                model,
                criterion,
                train_loader,
                seen_validation_loader,
                unseen_validation_loader,
                num_epochs=args.num_epochs,
                patience=args.patience,
                min_epochs=args.min_epochs,
                device=device,
                lr=args.lr,
                weight_decay=args.weight_decay,
                seen_validation_weight=args.seen_validation_weight,
                min_delta=args.min_delta,
            )
            model.load_state_dict(best_state)
            save_checkpoint(
                checkpoint_path,
                model=model,
                feature_mean=feature_mean,
                feature_std=feature_std,
                args=args,
                run_index=run_index,
                train_seed=train_seed,
                best_info=best_info,
                train_realization_ids=train_realization_ids,
                eval_realization_ids=eval_realization_ids,
                development_source_ids=development_source_ids,
                validation_source_ids=validation_source_ids,
                test_source_ids=test_source_ids,
                seen_validation_source_realizations=seen_validation_source_realizations,
            )
            print(f"Saved checkpoint: {checkpoint_path}")

        seen_validation_eval = core.evaluation_to_dict(
            core.evaluate_model(
                model, seen_validation_loader, device, criterion, eval_l1
            )
        )
        unseen_validation_eval = core.evaluation_to_dict(
            core.evaluate_model(
                model, unseen_validation_loader, device, criterion, eval_l1
            )
        )
        test_eval = core.evaluation_to_dict(
            core.evaluate_model(model, test_loader, device, criterion, eval_l1)
        )
        uncertainty = core.predict_uncertainty_by_assignment(
            model,
            test_uncertainty_loader,
            device,
            confidence=args.uncertainty_confidence,
        )
        uncertainty_summary = core.summarize_uncertainty(uncertainty)
        mean_metrics = assignment_mean_metrics(
            uncertainty,
            test_source_ids,
            test_features,
            args.fixed_eval_fpa,
            criterion,
        )

        selection_loss = (
            args.seen_validation_weight * float(seen_validation_eval["loss"])
            + (1.0 - args.seen_validation_weight)
            * float(unseen_validation_eval["loss"])
        )
        if selection_loss < best_run_selection_loss:
            best_run_selection_loss = selection_loss
            best_checkpoint = checkpoint_path

        row: Dict[str, Any] = {
            "script_version": SCRIPT_VERSION,
            "name": args.name,
            "number_of_assignments": int(args.number_of_assignments),
            "train_fpa": int(args.train_fpa),
            "fixed_eval_fpa": int(args.fixed_eval_fpa),
            "run": int(run_index + 1),
            "train_seed": int(train_seed),
            "best_epoch": int(round(float(best_info.get("best_epoch", -1)))),
            "validation_seen_loss": float(seen_validation_eval["loss"]),
            "validation_unseen_loss": float(unseen_validation_eval["loss"]),
            "selection_loss": float(selection_loss),
            "test_loss": float(test_eval["loss"]),
            "test_l1": float(test_eval["l1"]),
            "test_acc01": float(test_eval["acc@0.1"]),
            "test_acc02": float(test_eval["acc@0.2"]),
            "assignment_mean_loss": float(mean_metrics["loss"]),
            "assignment_mean_l1": float(mean_metrics["l1"]),
            "assignment_mean_acc01": float(mean_metrics["acc@0.1"]),
            "assignment_mean_acc02": float(mean_metrics["acc@0.2"]),
            "mean_prediction_std": float(uncertainty_summary["mean_prediction_std"]),
            "median_prediction_std": float(uncertainty_summary["median_prediction_std"]),
            "mean_ci_width": float(uncertainty_summary["mean_ci_width"]),
            "median_ci_width": float(uncertainty_summary["median_ci_width"]),
        }
        run_rows.append(row)

        print(
            "Validation          | "
            f"seen loss {row['validation_seen_loss']:.5f} | "
            f"unseen loss {row['validation_unseen_loss']:.5f} | "
            f"selection {row['selection_loss']:.5f}"
        )
        print(
            "Test per realization| "
            f"Loss {row['test_loss']:.5f} | L1 {row['test_l1']:.5f} | "
            f"Acc@0.1 {row['test_acc01']:.4f} | Acc@0.2 {row['test_acc02']:.4f}"
        )
        print(
            "Test assignment mean| "
            f"Loss {row['assignment_mean_loss']:.5f} | "
            f"L1 {row['assignment_mean_l1']:.5f} | "
            f"Acc@0.1 {row['assignment_mean_acc01']:.4f} | "
            f"Acc@0.2 {row['assignment_mean_acc02']:.4f}"
        )
        print(
            "Fixed-FPA stability | "
            f"mean pred SD {row['mean_prediction_std']:.5f} | "
            f"mean interval width {row['mean_ci_width']:.5f}"
        )

        if args.save_predictions:
            pred_path = output_dir / f"predictions_k{run_index}.npz"
            np.savez_compressed(
                pred_path,
                predictions=np.asarray(test_eval["predictions"], dtype=np.float32),
                labels=np.asarray(test_eval["labels"], dtype=np.float32),
                test_source_assignment_ids=np.asarray(test_source_ids, dtype=np.int64),
                eval_source_realization_ids=np.asarray(eval_realization_ids, dtype=np.int64),
            )
            print(f"Saved predictions: {pred_path}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    run_csv = output_dir / "fixed_test_fpa2_run_metrics.csv"
    write_run_csv(run_rows, run_csv)
    summary = build_summary(run_rows)

    summary_payload = {
        "script_version": SCRIPT_VERSION,
        "name": args.name,
        "number_of_assignments": int(args.number_of_assignments),
        "source_number_of_assignments": int(args.source_number_of_assignments),
        "train_fpa": int(args.train_fpa),
        "fixed_eval_fpa": int(args.fixed_eval_fpa),
        "validation_unseen_fpa": int(args.train_fpa),
        "source_fpa": int(args.source_fpa),
        "num_cascades": int(args.num_cascades),
        "noise": {"p1": args.p1, "p2": args.p2, "q": args.q_val},
        "split": {
            "development_fraction": args.development_fraction,
            "unseen_validation_fraction": args.unseen_validation_fraction,
            "unseen_test_fraction": 1.0
            - args.development_fraction
            - args.unseen_validation_fraction,
            "seen_validation_fraction": args.seen_validation_fraction,
            "seen_test_fraction": args.seen_test_fraction,
            "seen_validation_weight": args.seen_validation_weight,
            "subset_seed": args.subset_seed,
            "split_seed": args.split_seed,
            "development_source_ids": development_source_ids,
            "validation_source_ids": validation_source_ids,
            "test_source_ids": test_source_ids,
        },
        "train_realization_ids": train_realization_ids,
        "fixed_eval_realization_ids": eval_realization_ids,
        "seen_validation_source_realizations": seen_validation_source_realizations,
        "normalization": {
            "fit_on_training_only": True,
            "feature_mean": np.asarray(feature_mean).tolist(),
            "feature_std": np.asarray(feature_std).tolist(),
        },
        "summary": summary,
        "runs": run_rows,
    }
    summary_json = output_dir / "fixed_test_fpa2_summary.json"
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    if best_checkpoint is not None:
        best_alias = checkpoint_dir / (
            f"trainfpa{args.train_fpa}_testfpa{args.fixed_eval_fpa}_"
            f"BEST_{args.model_name}.pt"
        )
        shutil.copy2(best_checkpoint, best_alias)
        print(f"Best checkpoint alias: {best_alias}")

    print("\n========== K-RUN FIXED TEST-FPA SUMMARY ==========")
    for key in (
        "test_loss",
        "test_acc01",
        "assignment_mean_loss",
        "assignment_mean_acc01",
        "mean_prediction_std",
        "mean_ci_width",
    ):
        item = summary[key]
        print(f"{key:<26}: {item['mean']:.6f} +- {item['std']:.6f}")
    print(f"Saved run CSV:  {run_csv}")
    print(f"Saved summary:  {summary_json}")


if __name__ == "__main__":
    main()
