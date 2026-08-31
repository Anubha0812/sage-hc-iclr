# SAGE-HC: Hidden Independent Cascade Inference

Code accompanying the anonymous submission **“Gated Graph Neural Networks for Learning Hidden Independent Cascade Dynamics.”**

SAGE-HC estimates heterogeneous node-level Independent Cascade susceptibilities from known seed sets and noisy terminal symptom observations. Repeated hidden-cascade observations are summarized as **Symptom-Aware Cascade Features (SACF)** and processed by a residual gated graph neural network.


## Installation

Use 

```bash
bash -i install.sh
```

to create a `conda` environment and install torch and related dependencies. Alternatively, install dependencies with `pip`:

```bash
pip install -r requirements.txt
```


## Running experiments

You can run experiments using the following python/bash scripts. Alternatively, you can use SLURM scripts from `experiments/slurm`. See [Slurm README](experiments/slurm/README.MD) for details.


### Create features

Create features from the paper by using
```bash
bash experiments/generate_main_master.sh 
```

or create a small test feature set by using

```bash
bash experiments/generate_main_master.sh small-test
```

### Main benchmark — Tables 2 and 3

Run the full benchmark by running
```bash
bash experiments/run_benchmark.sh 
```

or use the small test set by using

```bash
bash experiments/run_benchmark.sh small-test
```

Compile the benchmark and empirical prediction-stability summaries with:

```bash
python analysis/compile_benchmark_tables.py --log-dir=./main_results
```
or for a small test:
```bash
python analysis/compile_benchmark_tables.py --log-dir=./main_results --assignments=10
```

### Figure 1 — sensitivity to susceptibility assignments

```bash
bash experiments/run_sensitivity_assignments.sh
```

or a small test with

```bash
bash experiments/run_sensitivity_assignments.sh small-test
```

### Figure 2 — sensitivity to cascades per seed

```bash
bash experiments/run_sensitivity_cascades.sh
```

or a small test with

```bash
bash experiments/run_sensitivity_cascades.sh small-test
```
### Figure 3 — sensitivity to graph size

```bash
bash experiments/run_sensitivity_nodes.sh
```

or a small test with

```bash
bash experiments/run_sensitivity_nodes.sh small-test
```

Figures 1–3 are generated from completed sensitivity logs using:

```bash
python analysis/plot_sensitivity.py
```

### Figure 4 — training feature realizations with fixed test FPA

```bash
bash experiments/run_sensitivity_fpa.sh
```

or a small test with

```bash
bash experiments/run_sensitivity_fpa.sh small-test
```


The paper figure uses a fixed strict unseen-test budget of `F_test=2` while varying the number of feature realizations available during training.

### Figure 5 — generalization across test-time cascade budgets

```bash
bash experiments/run_cross_cascade_generalization.sh
```
```bash
python analysis/plot_test_cascade_generalization.py
```

or a small test with

```bash
bash experiments/run_cross_cascade_generalization.sh small-test
```
```bash
python analysis/plot_test_cascade_generalization.py small-test
```

The model is trained once at `C_train=1000` cascades per seed and evaluated, without retraining, over smaller and larger test-time cascade budgets.

### Figure 6 — noise robustness

Noise robustness is tested by first creating a progressive set of graphs based on a tree graph

```bash
python core/create_progressive_graphs.py --graph_path=./data/graphs/graph_tree.pkl --graph_folder=./data/graphs --edges_per_step=100 --node_count=100
```

then creating the feature sets for each graph

```bash
bash experiments/simulate_progressive_tree.sh
```

and training the models for each feature set

```bash
bash experiments/train_progressive_tree.sh
```

for a single-gpu training and

```bash
bash experiments/train_progressive_tree.sh 8 2
```

to train on 8 gpus with 2 processes for each GPU.

Results of training you can see by running
```bash
python analysis/collect_results_progressive.py
```


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
<!-- 
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
``` -->

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

Data generation additionally expects the `cascadesimulator` module providing `pyCascadeGenerator`. This module is not bundled in the current archive and must be available in the Python environment before running the simulation scripts.

## Data

The repository contains the graph objects used by the experiments. Large pre-generated SACF master datasets are **not** included in this archive. The training launchers therefore expect the generated feature files at the paths configured in the shell scripts, or equivalent files generated with the supplied simulation entry points.

The balanced-tree graph stored in `data/graphs/graph_tree.pkl` has 100 nodes and 99 edges. Some historical feature filenames use a legacy node-count tag; the graph object itself is the authoritative topology.

## Reproducibility notes

Random seeds are fixed for assignment-subset selection, assignment-aware splitting, and independent model runs. The tests are done in Ubuntu 24.04.3 LTS with 8x2080Ti GPUs, 50-core Intel CPU and 188GB of RAM.
