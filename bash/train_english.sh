#!/bin/bash

set -e
START_TIME=$(date +%s)


# ---------------------CONFIGURATION SECTION---------------------
# mode = unsup / sup                  *** change here ***
MODE="sup"

# language = en / ch / hi              *** only english in train_english.sh for now ***
LANG="en"

# model backbone
MODEL_NAME="bert-base-uncased"

# training hyperparameters             *** use same parameters for sup/unsup ***
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

# Redirect all output to log (stdout and stderr)
LOG_ALL="./outputs/${LANG}/${MODE}/logs/train.log"
mkdir -p "./outputs/${LANG}/${MODE}/logs"
exec > >(tee "$LOG_ALL") 2>&1

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


# Notes: use the same training hyper-parameters for now, 
#       with seed = 42 by default, each train will produce the same model output
#       see outputs/.../logs/train.log


# unsupervised english model: 
# {'train_runtime': 2249.0046, 'train_samples_per_second': 444.641, 'train_steps_per_second': 6.948, 'train_loss': 1.1045955181884766, 'epoch': 1.0}

# supervised english model: 
# {'train_runtime': 407.7674, 'train_samples_per_second': 675.878, 'train_steps_per_second': 10.562, 'train_loss': 0.003008438345838, 'epoch': 1.0}


# *** remember to explain why the train loss is not close to 0 (this matches the paper),
# thus we are not plotting it, rather just use eval_english.py to produce spearman corr, 
# just like what the paper did ***