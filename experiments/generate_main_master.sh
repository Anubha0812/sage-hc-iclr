#!/bin/bash -l
#SBATCH --time=72:00:00
#SBATCH --partition=small
#SBATCH --job-name=masked14_10k_sim
#SBATCH --error=err_%x_%A_%a.txt
#SBATCH --output=out_%x_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=512G
#SBATCH --array=0-9%10

set -euo pipefail

export PYTHONNOUSERSITE=1
unset PYTHONPATH

# Prevent BLAS libraries from creating extra threads inside each worker.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hidden-cascades

PYTHON="$HOME/miniconda3/envs/hidden-cascades/bin/python"
PROJECT_DIR="${SAGE_HC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/core"
SIM_SCRIPT="$PROJECT_DIR/core/generate_master.py"
GRAPH_FOLDER="$PROJECT_DIR/data/graphs"
FEATURES_ROOT="/scratch/svc_td_fincomp/vrango/hic_new/assignment_split/features"
LOG_DIR="$PROJECT_DIR/results/master_generation/logs"

MASTER_ASSIGNMENTS=10000
FEATURES_PER_ASSIGNMENT=10
NUM_CASCADES=50

# Simulation-stage parameters only. Train/test IDs are created later by training.
DATA_SEED=0
ALPHA="0.8"

# The masked mode supports any value in (0, 1].
# For 10% seeds, change these two lines to 0.10 and seed10.
SEED_PERCENTAGE="1.0"
SEED_TAG="seed100"

# 14 features: original 9 + four self-seed statistics + observation mask.
FEATURE_MODE="multiple_new_features_self_seed_masked"
FEATURE_TAG="self_seed_masked_14features_v1"
EXPECTED_FEATURE_DIMENSION=14

GRAPH_LABELS=(tree karate ba2 ba3 ba4)
GRAPH_NAMES=(tree karate random random random)
NODE_COUNTS=(5 34 100 100 100)
CONNECTIONS_PER_NODE=(2 2 2 3 4)

NUM_GRAPHS=${#GRAPH_LABELS[@]}
NUM_NOISE_SETTINGS=2
EXPECTED_TASKS=$((NUM_GRAPHS * NUM_NOISE_SETTINGS))

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= EXPECTED_TASKS )); then
    echo "Invalid array task ID: $SLURM_ARRAY_TASK_ID" >&2
    echo "Expected an ID from 0 to $((EXPECTED_TASKS - 1))." >&2
    exit 2
fi

GRAPH_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_NOISE_SETTINGS))
P_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_NOISE_SETTINGS))

GRAPH_LABEL=${GRAPH_LABELS[$GRAPH_INDEX]}
GRAPH_NAME=${GRAPH_NAMES[$GRAPH_INDEX]}
NODE_COUNT=${NODE_COUNTS[$GRAPH_INDEX]}
CPN=${CONNECTIONS_PER_NODE[$GRAPH_INDEX]}

if [ "$P_INDEX" -eq 0 ]; then
    P_SETTING="nop"
    P1="0.0"
    P2="0.0"
    Q_VAL="1.0"
else
    P_SETTING="p020_q060"
    P1="0.2"
    P2="0.2"
    Q_VAL="0.6"
fi

EXP_NAME="master_${GRAPH_LABEL}_${MASTER_ASSIGNMENTS}_fpa${FEATURES_PER_ASSIGNMENT}_${P_SETTING}_${SEED_TAG}_masked14_cascades"

# Keep the 14-feature data separate from the old 9- and 13-feature datasets.
FEATURES_FOLDER="$FEATURES_ROOT/$SEED_TAG/$GRAPH_LABEL/$P_SETTING/$FEATURE_TAG"
LOG_FILE="$LOG_DIR/${EXP_NAME}.log"

mkdir -p "$FEATURES_FOLDER" "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

if [ ! -f "$SIM_SCRIPT" ]; then
    echo "Simulation script not found: $SIM_SCRIPT" >&2
    exit 2
fi

if [ ! -f "$GRAPH_FOLDER/graph_${GRAPH_NAME}.pkl" ] && [ "$GRAPH_NAME" != "random" ]; then
    echo "Warning: expected graph file may be missing for graph_name=$GRAPH_NAME" >&2
fi

echo "========== MASKED SELF-SEED MASTER SIMULATION =========="
echo "SLURM job ID:              ${SLURM_JOB_ID:-unknown}"
echo "Array task ID:             ${SLURM_ARRAY_TASK_ID:-unknown}"
echo "Experiment:                $EXP_NAME"
echo "Graph label:               $GRAPH_LABEL"
echo "Graph name:                $GRAPH_NAME"
echo "Node-count argument:       $NODE_COUNT"
echo "Connections per node:      $CPN"
echo "Master assignments:        $MASTER_ASSIGNMENTS"
echo "Features per assignment:   $FEATURES_PER_ASSIGNMENT"
echo "Cascades per seed:         $NUM_CASCADES"
echo "Seed percentage:           $SEED_PERCENTAGE"
echo "Seed tag:                  $SEED_TAG"
echo "Feature mode:              $FEATURE_MODE"
echo "Expected feature dimension:$EXPECTED_FEATURE_DIMENSION"
echo "Noise setting:             $P_SETTING"
echo "p1:                        $P1"
echo "p2:                        $P2"
echo "q:                         $Q_VAL"
echo "Feature folder:            $FEATURES_FOLDER"
echo "Save raw cascades:         True (2-bit packed master shards)"
echo "Workers:                   ${SLURM_CPUS_PER_TASK:-4}"
echo "Log file:                  $LOG_FILE"
echo "========================================================"

# Assignment shards are the master data.
#
# --no-consolidate avoids constructing one extremely large aggregate pickle.
# --verify-existing validates existing shards and regenerates only corrupt ones.
# Since --force is not used, a resubmitted or timed-out job reuses valid shards.
"$PYTHON" -s "$SIM_SCRIPT" \
    --name "$EXP_NAME" \
    --number-of-assignments "$MASTER_ASSIGNMENTS" \
    --graph-name "$GRAPH_NAME" \
    --node-count "$NODE_COUNT" \
    --connections-per-node "$CPN" \
    --features-per-assignment "$FEATURES_PER_ASSIGNMENT" \
    --num-cascades "$NUM_CASCADES" \
    --seed-percentage "$SEED_PERCENTAGE" \
    --assignment-generator-type "$FEATURE_MODE" \
    --split-mode assignment \
    --train-fraction 0.70 \
    --validation-fraction 0.15 \
    --data-seed "$DATA_SEED" \
    --alpha "$ALPHA" \
    --p1 "$P1" \
    --p2 "$P2" \
    --q-val "$Q_VAL" \
    --workers 0 \
    --features-folder "$FEATURES_FOLDER" \
    --graph-folder "$GRAPH_FOLDER" \
    --verify-existing \
    --save-cascades \
    --no-consolidate

echo "========================================================"
echo "Finished master simulation: $EXP_NAME"
echo "Master assignment shards:   $FEATURES_FOLDER"
echo "Cascade shards:             stored beside the corresponding feature shards"
echo "========================================================"