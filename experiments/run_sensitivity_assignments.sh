
TRAIN_SCRIPT="./core/train_sensitivity.py"
GRAPH_FOLDER="./data/graphs"
FEATURES_ROOT="./data/features"
RESULT_ROOT="./results/sensitivity_assignments"
LOG_DIR="$RESULT_ROOT/logs"
DONE_DIR="$RESULT_ROOT/done"
LOCK_DIR="$RESULT_ROOT/locks"
mkdir -p "$LOG_DIR" "$DONE_DIR" "$LOCK_DIR"

VALUES=(10 50 100 500 1000 5000 7000 10000)
NOISE_COUNT=2
GRAPH_LABEL="ba2"
GRAPH_NAME="random"
NODE_COUNT=100
CPN=2
MASTER_ASSIGNMENTS=10000
FPA=10
NUM_CASCADES=50
ALPHA="0.8"
SEED_PERCENTAGE="1.0"
SUBSET_SEED=0
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
COMPUTE_UNCERTAINTY="True"
UNCERTAINTY_CONFIDENCE="1.96"


if [ "$1" = "small-test" ]; then
    echo "Running in test mode: using small values for training/testing."
    
    VALUES=(10)
    MASTER_ASSIGNMENTS=10
    FPA=4
    NUM_CASCADES=5
    K_RUNS=2
    NUM_EPOCHS=10
    MIN_EPOCHS=10
fi

EXPECTED_TASKS=$(( ${#VALUES[@]} * NOISE_COUNT ))


for TASK_ID in $(seq 0 $((EXPECTED_TASKS - 1))); do

    echo "Task $TASK_ID: ${VALUES[$((TASK_ID / NOISE_COUNT))]} assignments, noise setting $((TASK_ID % NOISE_COUNT))"

    VALUE_INDEX=$((TASK_ID / NOISE_COUNT))
    NOISE_INDEX=$((TASK_ID % NOISE_COUNT))
    VALUE=${VALUES[$VALUE_INDEX]}

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
    EXP_NAME="sens_ba2_assignments_${VALUE}_${P_SETTING}_seed100_masked14_fpa10_c50_master10000_norm_uncertainty"
    LOG_FILE="$LOG_DIR/${EXP_NAME}.log"
    DONE_FILE="$DONE_DIR/${EXP_NAME}.done"
    LOCK_FILE="$LOCK_DIR/${EXP_NAME}.lock"

    exec > >(tee -a "$LOG_FILE") 2>&1

    echo "Launcher entered at $(date -Is)"
    echo "Experiment: $EXP_NAME"

    if [ -f "$DONE_FILE" ]; then
        echo "SKIPPED: completed marker exists: $DONE_FILE"
        continue
    fi

    exec 9>"$LOCK_FILE"
    if ! flock -n 9; then
        echo "SKIPPED: another job is running $EXP_NAME"
        continue
    fi

    if [ ! -f "$TRAIN_SCRIPT" ]; then
        echo "Training script not found: $TRAIN_SCRIPT" >&2
        exit 2
    fi

    P1_TAG=${P1//./}
    P2_TAG=${P2//./}
    MASTER_STEM="graph_${GRAPH_NAME}_agt_${FEATURE_MODE}_nc${NODE_COUNT}_cpn${CPN}_na${MASTER_ASSIGNMENTS}_nc${NUM_CASCADES}_a${ALPHA}_p${P1_TAG}_${P2_TAG}_q${Q_VAL}_fpa${FPA}"
    MASTER_SHARDS="$FEATURES_FOLDER/features_${MASTER_STEM}"
    if [ ! -d "$MASTER_SHARDS" ]; then
        echo "Required master shards not found:" >&2
        echo "$MASTER_SHARDS" >&2
        exit 2
    fi

    echo "Selected assignments: $VALUE"
    echo "Master assignments:   $MASTER_ASSIGNMENTS"
    echo "FPA:                  $FPA"
    echo "Cascades:             $NUM_CASCADES"
    echo "Noise:                $P_SETTING"
    echo "Test manifests:       disabled"

    cd "."

    PYTHONPATH="." python -u -s "$TRAIN_SCRIPT" \
        --name="$EXP_NAME" \
        --number_of_assignments="$VALUE" \
        --source_number_of_assignments="$MASTER_ASSIGNMENTS" \
        --subset_seed="$SUBSET_SEED" \
        --features_per_assignment="$FPA" \
        --source_features_per_assignment="$FPA" \
        --realization_subset_seed=0 \
        --num_cascades="$NUM_CASCADES" \
        --alpha="$ALPHA" \
        --node_count="$NODE_COUNT" \
        --model_name="$MODEL_NAME" \
        --graph_name="$GRAPH_NAME" \
        --connections_per_node="$CPN" \
        --assignemnt_generator_type="$FEATURE_MODE" \
        --seed_percentage="$SEED_PERCENTAGE" \
        --features_folder="$FEATURES_FOLDER" \
        --graph_folder="$GRAPH_FOLDER" \
        --p1="$P1" \
        --p2="$P2" \
        --q_val="$Q_VAL" \
        --development_fraction=0.80 \
        --unseen_validation_fraction=0.10 \
        --seen_validation_fraction=0.10 \
        --seen_test_fraction=0.00 \
        --seen_validation_weight=0.80 \
        --normalize_features=True \
        --input_feature_count="$INPUT_FEATURE_COUNT" \
        --compute_uncertainty="$COMPUTE_UNCERTAINTY" \
        --uncertainty_confidence="$UNCERTAINTY_CONFIDENCE" \
        --save_uncertainty_results=False \
        --save_test_manifest=False \
        --split_seed="$SPLIT_SEED" \
        --batch_size="$BATCH_SIZE" \
        --k_runs="$K_RUNS" \
        --lr="$LEARNING_RATE" \
        --weight_decay="$WEIGHT_DECAY" \
        --patience="$PATIENCE" \
        --min_epochs="$MIN_EPOCHS" \
        --num_epochs="$NUM_EPOCHS" \
        --device=cuda:0

    touch "$DONE_FILE"
    echo "Finished: $EXP_NAME"

done