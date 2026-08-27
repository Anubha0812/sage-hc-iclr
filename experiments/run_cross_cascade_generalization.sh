#!/bin/bash -l
#SBATCH --time=72:00:00
#SBATCH --partition=small
#SBATCH --job-name=ba2_crossC_fpa
#SBATCH --error=err_%x_%A_%a.txt
#SBATCH --output=out_%x_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --gres=gpu:1
#SBATCH --array=0-1%2

set -euo pipefail

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hidden-cascades
PYTHON="$HOME/miniconda3/envs/hidden-cascades/bin/python"

PROJECT_DIR="${SAGE_HC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/core"
PY_SCRIPT="$PROJECT_DIR/core/train_cross_cascade.py"
GRAPH_FOLDER="$PROJECT_DIR/data/graphs"
FEATURES_ROOT="/scratch/svc_td_fincomp/vrango/hic_new/assignment_split/features"
RESULT_ROOT="$PROJECT_DIR/results/cross_cascade_generalization"

# Existing cascade-sensitivity masters were generated with FPA=10.  We take the
# same deterministic nested two realizations from every master, so no new full
# simulation is needed if those feature shards are already present.
SOURCE_FPA=10
TARGET_FPA=10
ASSIGNMENTS=5000
TRAIN_CASCADES=1000
TEST_CASCADES="5,10,50,100,500,700,1000"

# One array task per observation-noise regime.
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    P_SETTING="nop"
    P1="0.0"
    P2="0.0"
    Q_VAL="1.0"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    P_SETTING="p020_q060"
    P1="0.2"
    P2="0.2"
    Q_VAL="0.6"
else
    echo "Invalid SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID" >&2
    exit 2
fi

FEATURE_TAG="self_seed_masked_14features_v1"
FEATURE_MODE="multiple_new_features_self_seed_masked"
FEATURES_FOLDER="$FEATURES_ROOT/seed100/ba2/$P_SETTING/$FEATURE_TAG"
OUTPUT_DIR="$RESULT_ROOT/$P_SETTING"
LOG_DIR="$RESULT_ROOT/logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

if [ ! -f "$PY_SCRIPT" ]; then
    echo "Python script not found: $PY_SCRIPT" >&2
    exit 2
fi

# Redirect the complete experiment log to scratch while still showing it in Slurm output.
LOG_FILE="$LOG_DIR/ba2_trainC1000_fpa_crossC_${P_SETTING}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========== BA(2) TRAIN C=1000 / FPA=2 -> CROSS-CASCADE TEST =========="
echo "Started:                 $(date -Is)"
echo "SLURM job:               ${SLURM_JOB_ID:-unknown}"
echo "Array task:              ${SLURM_ARRAY_TASK_ID:-unknown}"
echo "Noise:                   $P_SETTING"
echo "Assignments:             $ASSIGNMENTS"
echo "Training cascades:       $TRAIN_CASCADES"
echo "Target FPA:              $TARGET_FPA"
echo "Existing master FPA:     $SOURCE_FPA"
echo "Test cascade budgets:    $TEST_CASCADES"
echo "Feature folder:          $FEATURES_FOLDER"
echo "Output directory:        $OUTPUT_DIR"
echo ""

echo "This job REUSES the existing FPA=10 cascade-sensitivity feature masters."
echo "It trains only on the nested FPA=2 subset at C=1000."
echo "The same unseen-test assignments and same two realization IDs are tested at every C."
echo "C=1000 training normalization is frozen and reused for all lower-C test sets."
echo ""

# Direct Python execution is intentional; do not wrap this in srun.  This avoids
# the inherited SLURM_MEM_PER_* conflict encountered on this cluster.
"$PYTHON" -u -s "$PY_SCRIPT" \
    --name="ba2_trainC1000_fpa_crossC" \
    --feature-root="$FEATURES_FOLDER" \
    --graph-folder="$GRAPH_FOLDER" \
    --output-dir="$OUTPUT_DIR" \
    --noise-tag="$P_SETTING" \
    --p1="$P1" \
    --p2="$P2" \
    --q-val="$Q_VAL" \
    --number-of-assignments="$ASSIGNMENTS" \
    --source-number-of-assignments="$ASSIGNMENTS" \
    --target-fpa="$TARGET_FPA" \
    --source-fpa="$SOURCE_FPA" \
    --train-cascades="$TRAIN_CASCADES" \
    --test-cascades="$TEST_CASCADES" \
    --node-count=100 \
    --connections-per-node=2 \
    --alpha=0.8 \
    --seed-percentage=1.0 \
    --feature-mode="$FEATURE_MODE" \
    --input-feature-count=14 \
    --subset-seed=0 \
    --realization-subset-seed=0 \
    --split-seed=0 \
    --development-fraction=0.80 \
    --unseen-validation-fraction=0.10 \
    --seen-validation-fraction=0.10 \
    --seen-test-fraction=0.00 \
    --seen-validation-weight=0.80 \
    --model-name=rggcn \
    --k-runs=1 \
    --train-seed-start=1000 \
    --epochs=100 \
    --min-epochs=100 \
    --patience=20 \
    --batch-size=8 \
    --lr=0.001 \
    --weight-decay=0.0001 \
    --device=cuda:0 \
    --mixed-test \
    --resume

echo ""
echo "Finished: $(date -Is)"
echo "Results:  $OUTPUT_DIR"
