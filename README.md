# SAGE-HC: Hidden Independent Cascade Inference

Code accompanying the anonymous submission **“Gated Graph Neural Networks for Learning Hidden Independent Cascade Dynamics.”**

SAGE-HC estimates heterogeneous node-level Independent Cascade susceptibilities from known seed sets and noisy terminal symptom observations. Repeated hidden-cascade observations are summarized as **Symptom-Aware Cascade Features (SACF)** and processed by a residual gated graph neural network.

## Paper setting reproduced by this repository

The main benchmark uses:

- 5,000 susceptibility assignments selected from a 10,000-assignment master set;
- 10 feature realizations per assignment;
- 50 cascades per seed;
- all nodes used as seeds;
- 14 SACF features per node;
- hidden dimensions `256 -> 128 -> 64`;
- Adam, learning rate `1e-3`, weight decay `1e-4`;
- batch size `8`;
- 100 training epochs;
- Smooth L1 loss with `beta=0.1`;
- gradient-norm clipping at `1.0`;
- five independent training runs.

Two observation regimes are used:

| Regime | false positive | false negative-symptom | infected positive symptom |
|---|---:|---:|---:|
| Noise-free | `p1=0.0` | `p2=0.0` | `q=1.0` |
| Hidden/noisy | `p1=0.2` | `p2=0.2` | `q=0.6` |

The main benchmark covers the balanced tree, Karate Club, BA(2), BA(3), and BA(4) graphs.

## Repository structure

```text
sage-hc-iclr/
├── README.md
├── requirements.txt
├── core/
│   ├── model_utils.py
│   ├── train_benchmark.py
│   ├── train_sensitivity.py
│   ├── train_fpa_fixed_test.py
│   ├── train_cross_cascade.py
│   ├── generate_master.py
│   └── simulate_data_parallel_hic_self_seed_partial_final_with_cascades.py
├── networks/
│   └── rggcn.py
├── experiments/
│   ├── run_benchmark.sh
│   ├── run_sensitivity_assignments.sh
│   ├── run_sensitivity_cascades.sh
│   ├── run_sensitivity_nodes.sh
│   ├── run_sensitivity_fpa.sh
│   ├── run_cross_cascade_generalization.sh
│   └── generate_main_master.sh
├── analysis/
│   ├── compile_benchmark_tables.py
│   ├── plot_sensitivity.py
│   ├── plot_training_fpa_sensitivity.py
│   └── plot_test_cascade_generalization.py
└── data/graphs/
```

`core/main_hybrid_split_v7_sensitivity_nested_fpa.py` is retained only because the finalized FPA launcher checks for that compatibility filename.

## SACF features

The 14-feature configuration used in the reported experiments contains:

1. distance-weighted positive symptom ratio;
2. distance-weighted negative symptom ratio;
3. distance-weighted zero-symptom ratio;
4. symptom entropy;
5. minimum non-zero seed distance;
6. mean seed distance;
7. node degree;
8. mean positive-neighbor fraction;
9. mean positive-neighbor fraction conditional on the node being positive;
10. self-seed neighbor-positive mean;
11. self-seed neighbor-positive variance;
12. self-seed global-reach mean;
13. self-seed global-reach variance;
14. self-seed availability indicator.

## Assignment-aware split

The benchmark uses an assignment-aware split:

- 80% development assignments;
- 10% unseen-validation assignments;
- 10% strict unseen-test assignments.

Within development assignments, 10% of feature realizations are reserved for seen validation. Checkpoint selection uses

```text
0.8 * validation_seen_loss + 0.2 * validation_unseen_loss
```

Final benchmark metrics are reported only on strict unseen-test assignments.

Input-feature normalization uses means and standard deviations estimated from training samples only and applies those statistics unchanged to validation and test samples.

## Running experiments

The supplied shell launchers reproduce the HPC commands used for the reported experiments. They are SLURM scripts and currently retain the filesystem conventions of the original compute environment.

### Main benchmark — Tables 2 and 3

```bash
sbatch experiments/run_benchmark.sh
```

Compile the benchmark and empirical prediction-stability summaries with:

```bash
python analysis/compile_benchmark_tables.py --help
```

### Figure 1 — sensitivity to susceptibility assignments

```bash
sbatch experiments/run_sensitivity_assignments.sh
```

### Figure 2 — sensitivity to cascades per seed

```bash
sbatch experiments/run_sensitivity_cascades.sh
```

### Figure 3 — sensitivity to graph size

```bash
sbatch experiments/run_sensitivity_nodes.sh
```

Figures 1–3 are generated from completed sensitivity logs using:

```bash
python analysis/plot_sensitivity.py --help
```

### Figure 4 — training feature realizations with fixed test FPA

```bash
sbatch experiments/run_sensitivity_fpa.sh
```

The paper figure uses a fixed strict unseen-test budget of `F_test=2` while varying the number of feature realizations available during training.

### Figure 5 — generalization across test-time cascade budgets

```bash
sbatch experiments/run_cross_cascade_generalization.sh
python analysis/plot_test_cascade_generalization.py --help
```

The model is trained once at `C_train=1000` cascades per seed and evaluated, without retraining, over smaller and larger test-time cascade budgets.

## Evaluation metrics

The benchmark reports:

- Smooth L1 loss;
- Acc@0.1.

The code also computes L1 error and Acc@0.2 for diagnostics.

For the empirical prediction-stability analysis, predictions are grouped by unseen assignment and node across independent feature realizations. The code reports the prediction standard deviation and the width of

```text
mean_prediction +/- 1.96 * prediction_std / sqrt(number_of_realizations)
```

with endpoints clipped to `[0,1]`. These are empirical stability intervals, not formally calibrated confidence intervals.

## Dependencies

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

Data generation additionally expects the `cascadesimulator` module providing `pyCascadeGenerator`. This module is not bundled in the current archive and must be available in the Python environment before running the simulation scripts.

## Data

The repository contains the graph objects used by the experiments. Large pre-generated SACF master datasets are **not** included in this archive. The training launchers therefore expect the generated feature files at the paths configured in the shell scripts, or equivalent files generated with the supplied simulation entry points.

The balanced-tree graph stored in `data/graphs/graph_tree.pkl` has 100 nodes and 99 edges. Some historical feature filenames use a legacy node-count tag; the graph object itself is the authoritative topology.

## Reproducibility notes

Random seeds are fixed for assignment-subset selection, assignment-aware splitting, and independent model runs. The repository intentionally removes local logs, lock files, cached Python bytecode, macOS metadata, obsolete model families, and duplicate analysis scripts that are not required for the experiments reported in the paper.
