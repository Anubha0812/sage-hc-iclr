#!/usr/bin/env python3
"""Train BA(2) at C=1000/FPA=2 and test the frozen models across cascade budgets.

This script is intentionally specialized for the final cross-cascade generalization
experiment.  It reuses the existing FPA=10 cascade-sensitivity feature masters and
selects a deterministic nested FPA=2 subset for both training and testing.

Primary evaluation is paired: exactly the same unseen-test assignments and exactly
the same two realization IDs are evaluated at every test cascade budget.  This
isolates test-time observation budget from assignment difficulty.

The C=1000 training normalization statistics are frozen and reused for every test
cascade budget.  They are never re-fitted on test data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from core import train_sensitivity as core


SCRIPT_VERSION = "cross-cascade-generalization-fpa2-v1"
DEFAULT_TEST_CASCADES = (5, 10, 50, 100, 500, 700, 1000)


def parse_int_list(value: str) -> List[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    result = [int(item) for item in items]
    if any(item < 1 for item in result):
        raise argparse.ArgumentTypeError("Cascade counts must be >= 1.")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train K BA(2) models on 5000 assignments using a nested FPA=2 subset "
            "of the C=1000/FPA=10 master, save checkpoints plus normalization, then "
            "evaluate the frozen models on the same unseen assignments at multiple "
            "test-time cascade budgets."
        )
    )
    parser.add_argument("--name", default="ba2_trainc1000_fpa2_crosscascade")
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--graph-folder", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--noise-tag", choices=("nop", "p020_q060"), required=True)
    parser.add_argument("--p1", type=float, required=True)
    parser.add_argument("--p2", type=float, required=True)
    parser.add_argument("--q-val", type=float, required=True)

    parser.add_argument("--number-of-assignments", type=int, default=5000)
    parser.add_argument("--source-number-of-assignments", type=int, default=5000)
    parser.add_argument("--target-fpa", type=int, default=2)
    parser.add_argument("--source-fpa", type=int, default=10)
    parser.add_argument("--train-cascades", type=int, default=1000)
    parser.add_argument(
        "--test-cascades",
        type=parse_int_list,
        default=list(DEFAULT_TEST_CASCADES),
        help="Comma-separated test cascade budgets.",
    )

    parser.add_argument("--node-count", type=int, default=100)
    parser.add_argument("--connections-per-node", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--seed-percentage", type=float, default=1.0)
    parser.add_argument(
        "--feature-mode", default="multiple_new_features_self_seed_masked"
    )
    parser.add_argument("--input-feature-count", type=int, default=14)

    parser.add_argument("--subset-seed", type=int, default=0)
    parser.add_argument("--realization-subset-seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--development-fraction", type=float, default=0.80)
    parser.add_argument("--unseen-validation-fraction", type=float, default=0.10)
    parser.add_argument("--seen-validation-fraction", type=float, default=0.10)
    parser.add_argument("--seen-test-fraction", type=float, default=0.00)
    parser.add_argument("--seen-validation-weight", type=float, default=0.80)

    parser.add_argument("--model-name", default="rggcn")
    parser.add_argument("--k-runs", type=int, default=5)
    parser.add_argument("--train-seed-start", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--min-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--normalization-epsilon", type=float, default=1e-8)

    parser.add_argument(
        "--mixed-test",
        action="store_true",
        help=(
            "Also construct one heterogeneous test set by assigning each unseen "
            "assignment to exactly one cascade budget.  The paired per-budget curve "
            "remains the primary result."
        ),
    )
    parser.add_argument("--mixed-seed", type=int, default=12345)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse compatible saved checkpoints instead of retraining them.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save flattened predictions/labels for each run and cascade budget.",
    )
    return parser.parse_args()


def selected_realization_ids(source_fpa: int, target_fpa: int, seed: int) -> List[int]:
    if not (1 <= target_fpa <= source_fpa):
        raise ValueError("target_fpa must satisfy 1 <= target_fpa <= source_fpa.")
    rng = np.random.default_rng(seed)
    return rng.permutation(source_fpa)[:target_fpa].astype(int).tolist()


def build_paths(args: argparse.Namespace, cascades: int) -> Tuple[str, str]:
    graph_path, features_path = core.build_saved_paths(
        graph_name="random",
        node_count=args.node_count,
        connections_per_node=args.connections_per_node,
        number_of_assignments=args.source_number_of_assignments,
        num_cascades=cascades,
        alpha=args.alpha,
        p1=args.p1,
        p2=args.p2,
        q_val=args.q_val,
        features_per_assignment=args.source_fpa,
        assignemnt_generator_type=args.feature_mode,
        seed_percentage=args.seed_percentage,
        features_folder=str(args.feature_root),
        graph_folder=str(args.graph_folder),
        features_path=None,
        graph_path=None,
    )
    return graph_path, features_path


def require_shards(
    features_path: str,
    source_assignment_ids: Iterable[int],
) -> Path:
    shard_dir = Path(os.path.splitext(features_path)[0])
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"Feature shard directory not found: {shard_dir}")
    missing: List[Path] = []
    for source_id in source_assignment_ids:
        path = shard_dir / f"assignment_{int(source_id)}_features.pkl"
        if not path.is_file():
            missing.append(path)
            if len(missing) >= 10:
                break
    if missing:
        preview = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Required assignment feature shards are missing. First missing paths:\n"
            + preview
        )
    return shard_dir


def load_selected_test_features(
    *,
    features_path: str,
    source_assignment_ids: Sequence[int],
    source_fpa: int,
    local_realization_ids: Sequence[int],
) -> List[Dict[str, Any]]:
    shard_dir = require_shards(features_path, source_assignment_ids)
    result: List[Dict[str, Any]] = []
    for source_id in source_assignment_ids:
        shard_path = shard_dir / f"assignment_{int(source_id)}_features.pkl"
        with shard_path.open("rb") as handle:
            assignment_features = pickle.load(handle)
        if len(assignment_features) != source_fpa:
            raise ValueError(
                f"{shard_path} contains {len(assignment_features)} realizations; "
                f"expected source_fpa={source_fpa}."
            )
        result.extend(assignment_features[int(local_id)] for local_id in local_realization_ids)
    return result


def labels_by_assignment(
    features: Sequence[Mapping[str, Any]],
    source_assignment_ids: Sequence[int],
    target_fpa: int,
) -> Dict[int, np.ndarray]:
    labels: Dict[int, np.ndarray] = {}
    expected = len(source_assignment_ids) * target_fpa
    if len(features) != expected:
        raise ValueError(f"Expected {expected} selected test graphs, found {len(features)}.")
    for local_assignment, source_id in enumerate(source_assignment_ids):
        start = local_assignment * target_fpa
        reference = np.asarray(features[start]["labels"], dtype=np.float32)
        for offset in range(1, target_fpa):
            candidate = np.asarray(features[start + offset]["labels"], dtype=np.float32)
            if reference.shape != candidate.shape or not np.allclose(
                reference, candidate, rtol=0.0, atol=1e-8
            ):
                raise ValueError(
                    f"Labels differ across realizations for source assignment {source_id}."
                )
        labels[int(source_id)] = reference
    return labels


def verify_same_labels(
    reference: Mapping[int, np.ndarray],
    candidate: Mapping[int, np.ndarray],
    cascades: int,
) -> None:
    if set(reference) != set(candidate):
        raise ValueError(f"Test assignment IDs changed for C={cascades}.")
    for source_id, ref in reference.items():
        other = candidate[source_id]
        if ref.shape != other.shape or not np.allclose(ref, other, rtol=0.0, atol=1e-8):
            raise ValueError(
                f"Ground-truth labels changed across cascade budgets for source "
                f"assignment {source_id} at C={cascades}."
            )


def cpu_state_dict(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def checkpoint_compatible(payload: Mapping[str, Any], args: argparse.Namespace, run: int) -> bool:
    expected = {
        "script_version": SCRIPT_VERSION,
        "noise_tag": args.noise_tag,
        "train_cascades": args.train_cascades,
        "target_fpa": args.target_fpa,
        "source_fpa": args.source_fpa,
        "number_of_assignments": args.number_of_assignments,
        "source_number_of_assignments": args.source_number_of_assignments,
        "input_feature_count": args.input_feature_count,
        "model_name": args.model_name,
        "run_index": run,
        "train_seed": args.train_seed_start + run,
    }
    return all(payload.get(key) == value for key, value in expected.items())


def metric_dict(result: Tuple[Any, ...]) -> Dict[str, Any]:
    return core.evaluation_to_dict(result)


def compact(metrics: Mapping[str, Any]) -> Dict[str, float]:
    return {
        "loss": float(metrics["loss"]),
        "l1": float(metrics["l1"]),
        "acc@0.1": float(metrics["acc@0.1"]),
        "acc@0.2": float(metrics["acc@0.2"]),
    }


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return mean, std


def make_mixed_dataset(
    test_data_by_cascade: Mapping[int, Sequence[Any]],
    test_source_ids: Sequence[int],
    target_fpa: int,
    cascades: Sequence[int],
    seed: int,
) -> Tuple[List[Any], Dict[int, int]]:
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(test_source_ids, dtype=int)[rng.permutation(len(test_source_ids))]
    assignment_to_budget: Dict[int, int] = {
        int(source_id): int(cascades[index % len(cascades)])
        for index, source_id in enumerate(shuffled)
    }

    position = {int(source_id): index for index, source_id in enumerate(test_source_ids)}
    mixed: List[Any] = []
    counts = {int(c): 0 for c in cascades}
    for source_id in test_source_ids:
        budget = assignment_to_budget[int(source_id)]
        base = position[int(source_id)] * target_fpa
        mixed.extend(test_data_by_cascade[budget][base : base + target_fpa])
        counts[budget] += 1
    return mixed, counts


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temp, path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = args.output_dir / "predictions"
    if args.save_predictions:
        predictions_dir.mkdir(parents=True, exist_ok=True)

    if args.number_of_assignments != args.source_number_of_assignments:
        raise ValueError(
            "This final experiment expects the same 5000-assignment universe at every "
            "cascade budget. Set number_of_assignments == source_number_of_assignments."
        )
    if args.target_fpa != 2:
        print(f"Warning: target_fpa={args.target_fpa}; requested final design uses FPA=2.")
    if args.train_cascades not in args.test_cascades:
        args.test_cascades = sorted(set(args.test_cascades + [args.train_cascades]))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("========== CROSS-CASCADE GENERALIZATION ==========")
    print(f"Script version:             {SCRIPT_VERSION}")
    print(f"Noise:                      {args.noise_tag}")
    print(f"Device:                     {device}")
    print(f"Assignments:                {args.number_of_assignments}")
    print(f"Training cascades/seed:     {args.train_cascades}")
    print(f"Test cascades/seed:         {args.test_cascades}")
    print(f"Target FPA:                 {args.target_fpa}")
    print(f"Source FPA masters:         {args.source_fpa}")
    print(f"Runs:                       {args.k_runs}")

    local_realization_ids = selected_realization_ids(
        args.source_fpa, args.target_fpa, args.realization_subset_seed
    )
    print(f"Nested realization IDs:     {local_realization_ids}")

    # -------- training data: C=1000, nested FPA=2 from existing FPA=10 master --------
    graph_path, train_features_path = build_paths(args, args.train_cascades)
    all_source_ids = core._nested_assignment_ids(
        args.source_number_of_assignments,
        args.number_of_assignments,
        args.subset_seed,
    )
    require_shards(train_features_path, all_source_ids)

    graph, train_features, selected_source_ids = core.load_master_assignment_subset(
        graph_path,
        train_features_path,
        source_number_of_assignments=args.source_number_of_assignments,
        number_of_assignments=args.number_of_assignments,
        features_per_assignment=args.target_fpa,
        subset_seed=args.subset_seed,
        source_features_per_assignment=args.source_fpa,
        realization_subset_seed=args.realization_subset_seed,
    )
    if selected_source_ids != all_source_ids:
        raise AssertionError("Nested source assignment order changed unexpectedly.")

    core.validate_assignment_layout(
        train_features,
        args.number_of_assignments,
        args.target_fpa,
        verify_labels=True,
    )
    selected_feature_count, available_feature_count = core.resolve_input_feature_count(
        train_features, args.input_feature_count
    )
    split = core.create_hybrid_split(
        number_of_assignments=args.number_of_assignments,
        features_per_assignment=args.target_fpa,
        development_fraction=args.development_fraction,
        unseen_validation_fraction=args.unseen_validation_fraction,
        seen_validation_fraction=args.seen_validation_fraction,
        seen_test_fraction=args.seen_test_fraction,
        split_seed=args.split_seed,
    )
    core.print_split_summary(split)

    feature_mean, feature_std = core.compute_training_feature_normalization(
        train_features,
        split["train"],
        input_feature_count=selected_feature_count,
        available_feature_count=available_feature_count,
        epsilon=args.normalization_epsilon,
    )
    train_all_data, num_features = core.create_pyg_data(
        train_features,
        graph,
        input_feature_count=selected_feature_count,
        available_feature_count=available_feature_count,
        features_per_assignment=args.target_fpa,
        selected_source_assignment_ids=selected_source_ids,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    del train_features

    # Convert local unseen-test assignment IDs into source master IDs.  This same set
    # is used at every cascade budget.
    test_local_ids = [int(value) for value in split["unseen_test_assignments"]]
    test_source_ids = [int(selected_source_ids[local_id]) for local_id in test_local_ids]
    print(f"Unseen-test assignments:    {len(test_source_ids)}")
    print(f"First test source IDs:      {test_source_ids[:10]}")

    # -------- pre-load paired test datasets at every cascade budget --------
    test_data_by_cascade: Dict[int, List[Any]] = {}
    baseline_labels: Dict[int, np.ndarray] | None = None
    for cascades in args.test_cascades:
        _, features_path = build_paths(args, int(cascades))
        selected_test_features = load_selected_test_features(
            features_path=features_path,
            source_assignment_ids=test_source_ids,
            source_fpa=args.source_fpa,
            local_realization_ids=local_realization_ids,
        )
        current_labels = labels_by_assignment(
            selected_test_features, test_source_ids, args.target_fpa
        )
        if baseline_labels is None:
            baseline_labels = current_labels
        else:
            verify_same_labels(baseline_labels, current_labels, int(cascades))

        selected_count, available_count = core.resolve_input_feature_count(
            selected_test_features, args.input_feature_count
        )
        if selected_count != selected_feature_count or available_count != available_feature_count:
            raise ValueError(
                f"Feature dimensionality changed at C={cascades}: "
                f"selected={selected_count}, available={available_count}."
            )
        pyg_data, test_num_features = core.create_pyg_data(
            selected_test_features,
            graph,
            input_feature_count=selected_feature_count,
            available_feature_count=available_feature_count,
            features_per_assignment=args.target_fpa,
            selected_source_assignment_ids=test_source_ids,
            feature_mean=feature_mean,
            feature_std=feature_std,
        )
        if test_num_features != num_features:
            raise ValueError("Model input dimensionality changed across cascade budgets.")
        test_data_by_cascade[int(cascades)] = pyg_data
        print(
            f"Prepared paired test C={cascades}: "
            f"{len(test_source_ids)} assignments, {len(pyg_data)} graph samples"
        )
        del selected_test_features

    mixed_data: List[Any] | None = None
    mixed_counts: Dict[int, int] | None = None
    if args.mixed_test:
        mixed_data, mixed_counts = make_mixed_dataset(
            test_data_by_cascade,
            test_source_ids,
            args.target_fpa,
            args.test_cascades,
            args.mixed_seed,
        )
        print(f"Mixed-test assignment counts by C: {mixed_counts}")

    criterion = core.make_criterion("smooth_l1", beta=0.1)
    eval_l1 = nn.L1Loss()
    model_kwargs: Dict[str, Any] = {}

    run_records: List[Dict[str, Any]] = []
    paired_rows: List[Dict[str, Any]] = []
    mixed_rows: List[Dict[str, Any]] = []
    best_checkpoint_path: Path | None = None
    best_checkpoint_score = float("inf")

    for run in range(args.k_runs):
        train_seed = args.train_seed_start + run
        checkpoint_path = checkpoint_dir / f"{args.name}_{args.noise_tag}_k{run}_{args.model_name}.pt"
        payload: Dict[str, Any]

        if args.resume and checkpoint_path.is_file():
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if not checkpoint_compatible(payload, args, run):
                raise ValueError(
                    f"Existing checkpoint is incompatible with this configuration: {checkpoint_path}"
                )
            print(f"\n========== RUN {run + 1}/{args.k_runs}: REUSE CHECKPOINT ==========")
            print(f"Checkpoint: {checkpoint_path}")
            best_info = dict(payload["best_info"])
            state = payload["model_state_dict"]
        else:
            print(f"\n========== RUN {run + 1}/{args.k_runs}: TRAIN ==========")
            print(f"Training seed: {train_seed}")
            core.set_seed(train_seed)
            loaders = core.create_loaders(
                train_all_data, split, batch_size=args.batch_size, train_seed=train_seed
            )
            model = core.create_model(args.model_name, num_features, device, model_kwargs)
            state, best_info = core.train_model_dual_validation(
                model,
                criterion,
                loaders["train"],
                loaders["validation_seen"],
                loaders["validation_unseen"],
                num_epochs=args.epochs,
                patience=args.patience,
                min_epochs=args.min_epochs,
                device=device,
                lr=args.lr,
                weight_decay=args.weight_decay,
                seen_validation_weight=args.seen_validation_weight,
                min_delta=0.0,
            )
            payload = {
                "script_version": SCRIPT_VERSION,
                "name": args.name,
                "noise_tag": args.noise_tag,
                "p1": args.p1,
                "p2": args.p2,
                "q_val": args.q_val,
                "model_name": args.model_name,
                "model_kwargs": model_kwargs,
                "model_state_dict": cpu_state_dict(state),
                "run_index": run,
                "train_seed": train_seed,
                "best_info": {key: float(value) for key, value in best_info.items()},
                "train_cascades": args.train_cascades,
                "test_cascades": [int(value) for value in args.test_cascades],
                "number_of_assignments": args.number_of_assignments,
                "source_number_of_assignments": args.source_number_of_assignments,
                "target_fpa": args.target_fpa,
                "source_fpa": args.source_fpa,
                "selected_realization_ids": local_realization_ids,
                "subset_seed": args.subset_seed,
                "realization_subset_seed": args.realization_subset_seed,
                "split_seed": args.split_seed,
                "development_fraction": args.development_fraction,
                "unseen_validation_fraction": args.unseen_validation_fraction,
                "seen_validation_fraction": args.seen_validation_fraction,
                "seen_test_fraction": args.seen_test_fraction,
                "seen_validation_weight": args.seen_validation_weight,
                "input_feature_count": args.input_feature_count,
                "available_feature_count": available_feature_count,
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "unseen_test_source_assignment_ids": test_source_ids,
                "graph_path": os.path.abspath(graph_path),
                "train_features_path": os.path.abspath(train_features_path),
            }
            torch.save(payload, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            state = payload["model_state_dict"]

        selection_score = float(best_info["best_selection_loss"])
        if selection_score < best_checkpoint_score:
            best_checkpoint_score = selection_score
            best_checkpoint_path = checkpoint_path

        # Fresh model for frozen evaluation.
        model = core.create_model(args.model_name, num_features, device, model_kwargs)
        model.load_state_dict(state)
        model.eval()

        run_record: Dict[str, Any] = {
            "run_index": run,
            "train_seed": train_seed,
            "checkpoint": str(checkpoint_path),
            "best_info": {key: float(value) for key, value in best_info.items()},
            "paired_test": {},
        }

        print("\n----- frozen-model paired cross-cascade test -----")
        for cascades in args.test_cascades:
            loader = DataLoader(
                test_data_by_cascade[int(cascades)],
                batch_size=args.batch_size,
                shuffle=False,
            )
            metrics = metric_dict(
                core.evaluate_model(model, loader, device, criterion, eval_l1)
            )
            compact_metrics = compact(metrics)
            run_record["paired_test"][str(cascades)] = compact_metrics
            paired_rows.append(
                {
                    "noise": args.noise_tag,
                    "run": run,
                    "train_seed": train_seed,
                    "train_cascades": args.train_cascades,
                    "test_cascades": int(cascades),
                    "test_assignments": len(test_source_ids),
                    "test_graph_samples": len(test_data_by_cascade[int(cascades)]),
                    **compact_metrics,
                }
            )
            print(
                f"C_test={int(cascades):4d} | "
                f"Loss {compact_metrics['loss']:.5f} | "
                f"L1 {compact_metrics['l1']:.5f} | "
                f"Acc@0.1 {compact_metrics['acc@0.1']:.4f} | "
                f"Acc@0.2 {compact_metrics['acc@0.2']:.4f}"
            )
            if args.save_predictions:
                np.savez_compressed(
                    predictions_dir
                    / f"{args.name}_{args.noise_tag}_k{run}_testC{int(cascades)}.npz",
                    predictions=np.asarray(metrics["predictions"], dtype=np.float32),
                    labels=np.asarray(metrics["labels"], dtype=np.float32),
                )

        if mixed_data is not None:
            mixed_loader = DataLoader(mixed_data, batch_size=args.batch_size, shuffle=False)
            mixed_metrics = metric_dict(
                core.evaluate_model(model, mixed_loader, device, criterion, eval_l1)
            )
            mixed_compact = compact(mixed_metrics)
            run_record["mixed_test"] = {
                "assignment_counts_by_cascade": mixed_counts,
                "metrics": mixed_compact,
            }
            mixed_rows.append(
                {
                    "noise": args.noise_tag,
                    "run": run,
                    "train_seed": train_seed,
                    "test_assignments": len(test_source_ids),
                    "test_graph_samples": len(mixed_data),
                    **mixed_compact,
                }
            )
            print(
                "Mixed test      | "
                f"Loss {mixed_compact['loss']:.5f} | "
                f"L1 {mixed_compact['l1']:.5f} | "
                f"Acc@0.1 {mixed_compact['acc@0.1']:.4f} | "
                f"Acc@0.2 {mixed_compact['acc@0.2']:.4f}"
            )

        run_records.append(run_record)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if best_checkpoint_path is None:
        raise RuntimeError("No checkpoint was produced.")
    best_alias = checkpoint_dir / f"{args.name}_{args.noise_tag}_BEST_{args.model_name}.pt"
    shutil.copy2(best_checkpoint_path, best_alias)
    print(f"\nBest validation-selected run checkpoint: {best_checkpoint_path}")
    print(f"Best checkpoint alias:                  {best_alias}")

    # -------- K-run summaries --------
    summary_by_cascade: Dict[str, Any] = {}
    print("\n========== K-RUN PAIRED CROSS-CASCADE SUMMARY ==========")
    for cascades in args.test_cascades:
        rows = [row for row in paired_rows if row["test_cascades"] == int(cascades)]
        summary_metrics: Dict[str, Dict[str, float]] = {}
        for metric in ("loss", "l1", "acc@0.1", "acc@0.2"):
            mean, std = mean_std([float(row[metric]) for row in rows])
            summary_metrics[metric] = {"mean": mean, "std": std}
        summary_by_cascade[str(cascades)] = {
            "test_assignments": len(test_source_ids),
            "test_graph_samples": len(test_data_by_cascade[int(cascades)]),
            "metrics": summary_metrics,
        }
        print(
            f"C_test={int(cascades):4d} | "
            f"L1 {summary_metrics['l1']['mean']:.5f} +/- {summary_metrics['l1']['std']:.5f} | "
            f"Acc@0.1 {summary_metrics['acc@0.1']['mean']:.4f} +/- {summary_metrics['acc@0.1']['std']:.4f} | "
            f"Acc@0.2 {summary_metrics['acc@0.2']['mean']:.4f} +/- {summary_metrics['acc@0.2']['std']:.4f}"
        )

    mixed_summary: Dict[str, Any] | None = None
    if mixed_rows:
        mixed_summary = {"assignment_counts_by_cascade": mixed_counts, "metrics": {}}
        for metric in ("loss", "l1", "acc@0.1", "acc@0.2"):
            mean, std = mean_std([float(row[metric]) for row in mixed_rows])
            mixed_summary["metrics"][metric] = {"mean": mean, "std": std}
        print("\n========== K-RUN MIXED TEST SUMMARY ==========")
        print(json.dumps(mixed_summary, indent=2, sort_keys=True))

    csv_path = args.output_dir / "paired_cross_cascade_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "noise",
            "run",
            "train_seed",
            "train_cascades",
            "test_cascades",
            "test_assignments",
            "test_graph_samples",
            "loss",
            "l1",
            "acc@0.1",
            "acc@0.2",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(paired_rows)

    output_payload = {
        "script_version": SCRIPT_VERSION,
        "name": args.name,
        "noise_tag": args.noise_tag,
        "train_configuration": {
            "train_cascades": args.train_cascades,
            "number_of_assignments": args.number_of_assignments,
            "target_fpa": args.target_fpa,
            "source_fpa": args.source_fpa,
            "selected_realization_ids": local_realization_ids,
            "development_fraction": args.development_fraction,
            "unseen_validation_fraction": args.unseen_validation_fraction,
            "seen_validation_fraction": args.seen_validation_fraction,
            "seen_test_fraction": args.seen_test_fraction,
            "seen_validation_weight": args.seen_validation_weight,
            "normalization_source": "C=1000 training split only; frozen for all test budgets",
        },
        "test_configuration": {
            "test_cascades": [int(value) for value in args.test_cascades],
            "paired_same_unseen_assignments_across_budgets": True,
            "unseen_test_assignment_count": len(test_source_ids),
            "unseen_test_source_assignment_ids": test_source_ids,
        },
        "best_validation_checkpoint": str(best_alias),
        "best_validation_selection_loss": best_checkpoint_score,
        "paired_summary": summary_by_cascade,
        "mixed_summary": mixed_summary,
        "runs": run_records,
    }
    json_path = args.output_dir / "cross_cascade_summary.json"
    save_json(json_path, output_payload)

    if mixed_rows:
        mixed_csv_path = args.output_dir / "mixed_cross_cascade_metrics.csv"
        with mixed_csv_path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "noise",
                "run",
                "train_seed",
                "test_assignments",
                "test_graph_samples",
                "loss",
                "l1",
                "acc@0.1",
                "acc@0.2",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(mixed_rows)

    print("\nFinished.")
    print(f"Summary JSON: {json_path}")
    print(f"Paired CSV:   {csv_path}")


if __name__ == "__main__":
    main()
