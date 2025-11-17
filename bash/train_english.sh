#!/bin/bash
set -e

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
WARMUP_RATIO=0.1
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
mkdir -p "./outputs/${LANG}/${MODE}/logs"
LOG_FILE="./outputs/${LANG}/${MODE}/logs/train.log"

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
echo "WARMUP_RATIO   = $WARMUP_RATIO"
echo "MAX_LEN        = $MAX_LEN"

# ---------------------RUN TRAINING---------------------
python train_english.py \
    --mode $MODE \
    --model_name $MODEL_NAME \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --warmup_ratio $WARMUP_RATIO \
    --max_len $MAX_LEN \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    2>&1 | tee "$LOG_FILE"

echo "===== TRAINING DONE ====="
echo "Logs saved to: $LOG_FILE"
echo "Checkpoints saved to: $OUTPUT_DIR"

# Run with:
#   bash bash/train_english.sh