#!/bin/bash
# Local evaluation script for multi-language SimCSE (unsup / sup)

set -e

# ------------------------CONFIGURATION SECTION------------------------
TASK="sup"          # sup / unsup
LANG="en"           # en / ch / hi

MODELS_EN=(
  "bert-base-uncased"                    # models to be filled in
)

MODELS_CH=(
  "bert-base-chinese"                    # models to be filled in
)

MODELS_HI=(
  "bert-base-multilingual-cased"             # models to be filled in
)

if [[ "$LANG" == "en" ]]; then
    MODELS=("${MODELS_EN[@]}")
elif [[ "$LANG" == "ch" ]]; then
    MODELS=("${MODELS_CH[@]}")
elif [[ "$LANG" == "hi" ]]; then
    MODELS=("${MODELS_HI[@]}")
else
    echo "[ERROR] Unknown LANG=$LANG"
    exit 1
fi

if [[ "$LANG" == "en" ]]; then
    TASK_FILES=(
        "./data/english/STS-B/original/sts-dev.tsv"
        "./data/english/STS-B/original/sts-test.tsv"
        "./data/english/STS-B/original/sts-train.tsv"
    )
elif [[ "$LANG" == "ch" ]]; then
    TASK_FILES=(
        "./data/chinese/STS-B/sts-dev.tsv"
        "./data/chinese/STS-B/sts-test.tsv"
        "./data/chinese/STS-B/sts-train.tsv"
    )
elif [[ "$LANG" == "hi" ]]; then
    TASK_FILES=(
        "./data/hindi/STS-B/sts-dev.tsv"
        "./data/hindi/STS-B/sts-test.tsv"
    )
fi

# CKPT will be computed per model inside loop

for MODEL_NAME in "${MODELS[@]}"; do
    CLEAN_MODEL="${MODEL_NAME//\//_}"
    CKPT="./outputs/${LANG}/${CLEAN_MODEL}/${TASK}/checkpoints"
    EVAL_DIR="./outputs/${LANG}/${CLEAN_MODEL}/${TASK}/eval"
    mkdir -p "${EVAL_DIR}"
    OUTPUT_SUMMARY="${EVAL_DIR}/summary.txt"
    LOG_FILE="${EVAL_DIR}/eval.log"
    : > "$LOG_FILE"

    echo "===== Starting Evaluation (${TASK}, ${LANG}, MODEL=${MODEL_NAME}) =====" | tee -a "$LOG_FILE"

    # Loop over STS files (keep English STS-B untouched)
    for FILE in "${TASK_FILES[@]}"; do
        BASENAME=$(basename "$FILE")
        echo "===== Evaluating ${BASENAME} =====" | tee -a "$LOG_FILE"
        python evaluate.py \
            --model_path "$CKPT" \
            --test_file "$FILE" \
            --backbone "$MODEL_NAME" 2>&1 | tee -a "$LOG_FILE" | tee tmp_eval.log

        SCORE=$(grep "Spearman" tmp_eval.log | awk '{print $NF}')
        echo "${BASENAME}   ${SCORE}" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
    done

    rm -f tmp_eval.log
    echo "===== Evaluation Done for MODEL=${MODEL_NAME} =====" | tee -a "$LOG_FILE"
done
