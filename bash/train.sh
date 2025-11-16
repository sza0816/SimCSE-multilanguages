#!/bin/bash
set -e

###########################################################
#                 CONFIGURATION SECTION
###########################################################
TASK="unsup"                     # unsup / sup （only run unsup now）
LANG="en"                        # en / ch / hi
MODEL_NAME="bert-base-uncased"
EPOCHS=1
BATCH_SIZE=64
LR=3e-5
WARMUP_RATIO=0.1

# data path in local repo
DATA_PATH="./data/english/unsup/wiki1m_for_simcse.txt"

# checkpoint in local repo
OUTPUT_DIR="./outputs/${LANG}/${TASK}/checkpoints"
###########################################################


echo "===== GPU INFO ====="
nvidia-smi

###########################################################
#       OPTIONAL: Auto-select model by language
###########################################################
if [[ "$LANG" == "en" ]]; then
    MODEL_NAME="bert-base-uncased"
elif [[ "$LANG" == "ch" ]]; then
    MODEL_NAME="bert-base-chinese"
elif [[ "$LANG" == "hi" ]]; then
    MODEL_NAME="google/muril-base-cased"
fi
echo "MODEL = $MODEL_NAME"

###########################################################
#                     RUN TRAINING
###########################################################
# create output directories
mkdir -p "./outputs/${LANG}/${TASK}/checkpoints"
mkdir -p "./outputs/${LANG}/${TASK}/logs"

LOG_FILE="./outputs/${LANG}/${TASK}/logs/train.log"

python train_unsup.py \
    --model_name $MODEL_NAME \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --warmup_ratio $WARMUP_RATIO \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR | tee "$LOG_FILE"

echo "===== TRAINING DONE ====="

# Run with:  bash bash/train.sh