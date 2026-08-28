
## Slurm experiments

The shell scripts in `experiments/slurm` reproduce the HPC commands used for the reported experiments. They are SLURM scripts and currently retain the filesystem conventions of the original compute environment.

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
