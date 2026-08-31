TRAIN_SCRIPT="./core/train_fpa_fixed_test.py"
GRAPH_FOLDER="./data/graphs"
FEATURES_ROOT="./data/features"
RESULT_ROOT="./results/sensitivity_fpa"
LOG_DIR="$RESULT_ROOT/logs"
DONE_DIR="$RESULT_ROOT/done"
LOCK_DIR="$RESULT_ROOT/locks"
mkdir -p "$LOG_DIR" "$DONE_DIR" "$LOCK_DIR"

# Same FPA grid as the user's corrected benchmark-split sensitivity launcher.
# FPA=1 is excluded because 10% seen validation would leave no training
# realization for each development assignment.
FPA_VALUES=(2 5 10 20 50 70 100)
NOISE_COUNT=2
GRAPH_LABEL="ba2"
GRAPH_NAME="random"
NODE_COUNT=100
CPN=2
TARGET_ASSIGNMENTS=1000
SOURCE_FPA=100
FIXED_EVAL_FPA=2
NUM_CASCADES=50
ALPHA="0.8"
SEED_PERCENTAGE="1.0"
SUBSET_SEED=0
REALIZATION_SUBSET_SEED=0
SPLIT_SEED=0
FEATURE_MODE="multiple_new_features_self_seed_masked"
FEATURE_TAG="self_seed_masked_14features_v1"
INPUT_FEATURE_COUNT=14
MODEL_NAME="rggcn"
K_RUNS=5
NUM_EPOCHS=100
LEARNING_RATE="0.001"
WEIGHT_DECAY="0.0001"
PATIENCE=20
MIN_EPOCHS=100
BATCH_SIZE=8
UNCERTAINTY_CONFIDENCE="1.96"

if [ "$1" = "small-test" ]; then
    echo "Running in test mode: using small values for training/testing."
    
    FPA_VALUES=(4)
    SOURCE_FPA=4
    TARGET_ASSIGNMENTS=10
    K_RUNS=2
    NUM_EPOCHS=10
    MIN_EPOCHS=10
fi

EXPECTED_TASKS=$(( ${#FPA_VALUES[@]} * NOISE_COUNT ))

for TASK_ID in $(seq 0 $((EXPECTED_TASKS - 1))); do

    echo "Task $TASK_ID: ${FPA_VALUES[$((TASK_ID / NOISE_COUNT))]} FPA, noise setting $((TASK_ID % NOISE_COUNT))"

    FPA_INDEX=$((TASK_ID / NOISE_COUNT))
    NOISE_INDEX=$((TASK_ID % NOISE_COUNT))
    TRAIN_FPA=${FPA_VALUES[$FPA_INDEX]}

    if [ "$NOISE_INDEX" -eq 0 ]; then
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

    FEATURES_FOLDER="$FEATURES_ROOT/seed100/$GRAPH_LABEL/$P_SETTING/$FEATURE_TAG"

    P1_TAG=${P1//./}
    P2_TAG=${P2//./}
    MASTER_ASSIGNMENTS=""
    MASTER_SHARD_DIR=""
    for CANDIDATE in 5000 10000 10; do
        STEM="graph_${GRAPH_NAME}_agt_${FEATURE_MODE}_nc${NODE_COUNT}_cpn${CPN}_na${CANDIDATE}_nc${NUM_CASCADES}_a${ALPHA}_p${P1_TAG}_${P2_TAG}_q${Q_VAL}_fpa${SOURCE_FPA}"
        CANDIDATE_DIR="$FEATURES_FOLDER/features_${STEM}"
        if [ -d "$CANDIDATE_DIR" ]; then
            SHARD_COUNT=$(find "$CANDIDATE_DIR" -maxdepth 1 -type f -name 'assignment_*_features.pkl' | wc -l)
            echo "Candidate master: $CANDIDATE_DIR"
            echo "  feature shards: $SHARD_COUNT/$CANDIDATE"
            if [ "$SHARD_COUNT" -eq "$CANDIDATE" ]; then
                MASTER_ASSIGNMENTS="$CANDIDATE"
                MASTER_SHARD_DIR="$CANDIDATE_DIR"
                break
            fi
        fi
    done

    if [ -z "$MASTER_ASSIGNMENTS" ]; then
        echo "Could not find a complete FPA=100, C=50 master for $P_SETTING." >&2
        echo "Checked 5000- and 10000-assignment masters in: $FEATURES_FOLDER" >&2
        exit 2
    fi

    EXP_NAME="diag_fixedtest2_ba2_fpa_${TRAIN_FPA}_${P_SETTING}_assignments1000_c50_masterfpa100"
    OUT_DIR="$RESULT_ROOT/$P_SETTING/train_fpa_${TRAIN_FPA}"
    LOG_FILE="$LOG_DIR/${EXP_NAME}.log"
    DONE_FILE="$DONE_DIR/${EXP_NAME}.done"
    LOCK_FILE="$LOCK_DIR/${EXP_NAME}.lock"
    mkdir -p "$OUT_DIR"

    exec 9>"$LOCK_FILE"
    if ! flock -n 9; then
        echo "SKIPPED: another job is currently running $EXP_NAME"
        exit 0
    fi

    if [ -f "$DONE_FILE" ]; then
        echo "SKIPPED: completed marker exists: $DONE_FILE"
        exit 0
    fi

    : > "$LOG_FILE"
    exec > >(tee -a "$LOG_FILE") 2>&1

    echo "========== BA(2) TRAIN-FPA -> FIXED TEST FPA=2, 1000 ASSIGNMENTS =========="
    echo "Started:                    $(date -Is)"
    echo "Experiment:                 $EXP_NAME"
    echo "Training FPA:               $TRAIN_FPA"
    echo "Unseen-validation FPA:      same as TRAIN_FPA (original protocol)"
    echo "Fixed strict-test FPA:       $FIXED_EVAL_FPA"
    echo "Master/source FPA:          $SOURCE_FPA"
    echo "Master assignments:         $MASTER_ASSIGNMENTS"
    echo "Selected assignments:       $TARGET_ASSIGNMENTS"
    echo "Master shard directory:     $MASTER_SHARD_DIR"
    echo "Noise:                      $P_SETTING"
    echo "Cascades per seed:          $NUM_CASCADES"
    echo "Split:                      0.80 development / 0.10 unseen-val / 0.10 unseen-test"
    echo "Seen validation:            0.10 of TRAIN_FPA"
    echo "Seen test:                  0.00"
    echo "Checkpoint weighting:       0.80 seen + 0.20 unseen"
    echo "K runs:                     $K_RUNS"

    if [ ! -f "$TRAIN_SCRIPT" ]; then
        echo "Python script not found: $TRAIN_SCRIPT" >&2
        exit 2
    fi
    if [ ! -f "./core/main_hybrid_split_v7_sensitivity_nested_fpa.py" ]; then
        echo "Required core training module is missing from project directory." >&2
        exit 2
    fi

    cd "."

    # Direct Python is intentional; do not use srun on this cluster.
    PYTHONPATH="." python -u -s "$TRAIN_SCRIPT" \
        --name "$EXP_NAME" \
        --train-fpa "$TRAIN_FPA" \
        --fixed-eval-fpa "$FIXED_EVAL_FPA" \
        --source-fpa "$SOURCE_FPA" \
        --number-of-assignments "$TARGET_ASSIGNMENTS" \
        --source-number-of-assignments "$MASTER_ASSIGNMENTS" \
        --subset-seed "$SUBSET_SEED" \
        --realization-subset-seed "$REALIZATION_SUBSET_SEED" \
        --split-seed "$SPLIT_SEED" \
        --num-cascades "$NUM_CASCADES" \
        --alpha "$ALPHA" \
        --p1 "$P1" \
        --p2 "$P2" \
        --q-val "$Q_VAL" \
        --node-count "$NODE_COUNT" \
        --connections-per-node "$CPN" \
        --graph-name "$GRAPH_NAME" \
        --assignment-generator-type "$FEATURE_MODE" \
        --seed-percentage "$SEED_PERCENTAGE" \
        --input-feature-count "$INPUT_FEATURE_COUNT" \
        --development-fraction 0.80 \
        --unseen-validation-fraction 0.10 \
        --seen-validation-fraction 0.10 \
        --seen-test-fraction 0.00 \
        --seen-validation-weight 0.80 \
        --model-name "$MODEL_NAME" \
        --k-runs "$K_RUNS" \
        --num-epochs "$NUM_EPOCHS" \
        --min-epochs "$MIN_EPOCHS" \
        --patience "$PATIENCE" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LEARNING_RATE" \
        --weight-decay "$WEIGHT_DECAY" \
        --uncertainty-confidence "$UNCERTAINTY_CONFIDENCE" \
        --features-folder "$FEATURES_FOLDER" \
        --graph-folder "$GRAPH_FOLDER" \
        --output-dir "$OUT_DIR" \
        --resume

    touch "$DONE_FILE"
    echo "Finished: $EXP_NAME"
    echo "Finished at $(date -Is)"

done