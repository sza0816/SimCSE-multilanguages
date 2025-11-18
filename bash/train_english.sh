#!/bin/bash

set -e
START_TIME=$(date +%s)

# Redirect all output to log (stdout and stderr)
LOG_ALL="./outputs/${LANG}/${MODE}/logs/train.log"
mkdir -p "./outputs/${LANG}/${MODE}/logs"
exec > >(tee -a "$LOG_ALL") 2>&1

# ---------------------CONFIGURATION SECTION---------------------
# mode = unsup / sup
MODE="unsup"

# language = en / ch / hi              *** only english in train_english.sh for now ***
LANG="en"

# model backbone
MODEL_NAME="bert-base-uncased"

# training hyperparameters
EPOCHS=1
BATCH_SIZE=64
LR=5e-5
MAX_LEN=32


# ---------------------DATASET SELECTION---------------------

if [[ "$MODE" == "unsup" ]]; then
    DATA_PATH="./data/english/unsup/wiki1m_for_simcse.txt"
elif [[ "$MODE" == "sup" ]]; then
    DATA_PATH="./data/english/sup/nli_for_simcse.csv"
else
    echo "[ERROR] MODE must be unsup or sup."
    exit 1
fi


# ---------------------OUTPUT DIRECTORY (organized cleanly)---------------------
OUTPUT_DIR="./outputs/${LANG}/${MODE}/checkpoints"

mkdir -p "$OUTPUT_DIR"

# ---------------------GPU CHECK---------------------
echo "===== GPU INFO ====="
nvidia-smi || echo "No GPU detected (maybe using CPU-only)."

echo "===== TRAINING CONFIG ====="
echo "MODE           = $MODE"
echo "LANG           = $LANG"
echo "MODEL_NAME     = $MODEL_NAME"
echo "DATA_PATH      = $DATA_PATH"
echo "OUTPUT_DIR     = $OUTPUT_DIR"
echo "BATCH_SIZE     = $BATCH_SIZE"
echo "EPOCHS         = $EPOCHS"
echo "LR             = $LR"
echo "MAX_LEN        = $MAX_LEN"

# ---------------------RUN TRAINING---------------------
python train_english.py \
    --mode $MODE \
    --model_name $MODEL_NAME \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --max_len $MAX_LEN \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR

echo "===== TRAINING DONE ====="
echo "Logs saved to: $LOG_ALL"
echo "Checkpoints saved to: $OUTPUT_DIR"

# record total training time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "Total training time: ${ELAPSED} seconds"

# Run with:
#   bash bash/train_english.sh

# removed
# WARMUP_RATIO=0.1
# echo "WARMUP_RATIO   = $WARMUP_RATIO"
# --warmup_ratio $WARMUP_RATIO \