#!/bin/bash

set -e
START_TIME=$(date +%s)

# ----------------------Configuration Section !!!-------------------------

# MODE = sup / unsup
MODE="unsup"

# LANG = en / ch / hi
LANG="ch"

MODELS_EN=(
#   "bert-base-uncased"
#   "roberta-base"
#   "distilbert-base-uncased"
)
MODELS_CH=(
#   "bert-base-chinese"
  "hfl/chinese-roberta-wwm-ext"
#   "hfl/chinese-macbert-base"
)
MODELS_HI=(
#   "bert-base-multilingual-cased"
#   "xlm-roberta-base"
#   "ai4bharat/IndicBERTv2-MLM-only"
)

# Training hyperparameters - currently have no time for tunning hyperparams, mention in report
EPOCHS=1
BATCH_SIZE=64
LR=5e-5
MAX_LEN=32

# --------------------------------------------------------------------------------------

# select data path

if [[ "$LANG" == "en" ]]; then
    if [[ "$MODE" == "unsup" ]]; then
        DATA_PATH="./data/english/unsup/wiki_english.txt"
    else
        DATA_PATH="./data/english/sup/nli_english.csv"
    fi
elif [[ "$LANG" == "ch" ]]; then
    if [[ "$MODE" == "unsup" ]]; then
        DATA_PATH="./data/chinese/unsup/wiki_chinese.txt"
    else
        DATA_PATH="./data/chinese/sup/nli_chinese.csv"
    fi
elif [[ "$LANG" == "hi" ]]; then
    if [[ "$MODE" == "unsup" ]]; then
        DATA_PATH="./data/hindi/unsup/wiki_hindi.txt"
    else
        DATA_PATH="./data/hindi/sup/nli_hindi.csv"
    fi
else
    echo "[ERROR] Unknown LANG = $LANG"
    exit 1
fi

if [[ "$LANG" == "en" ]]; then
    MODELS=("${MODELS_EN[@]}")
elif [[ "$LANG" == "ch" ]]; then
    MODELS=("${MODELS_CH[@]}")
elif [[ "$LANG" == "hi" ]]; then
    MODELS=("${MODELS_HI[@]}")
fi


# --------------------------loop over all models of a language--------------------------------
# might take several hours if training too many models
# recommanded: for experiment, train one model of one language each time

for MODEL_NAME in "${MODELS[@]}"; do

    echo "==============================================="
    echo " Training model: $MODEL_NAME"
    echo " MODE=$MODE  LANG=$LANG"
    echo " DATA_PATH=$DATA_PATH"
    echo "==============================================="

    CLEAN_MODEL_NAME="${MODEL_NAME//\//_}"

    OUTPUT_DIR="./outputs/${LANG}/${CLEAN_MODEL_NAME}/${MODE}/checkpoints"
    mkdir -p "$OUTPUT_DIR"

    LOG_DIR="./outputs/${LANG}/${CLEAN_MODEL_NAME}/${MODE}/logs"
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/train.log"

    {
        python train.py \
            --mode $MODE \
            --lang $LANG \
            --model_name "$MODEL_NAME" \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --lr $LR \
            --max_len $MAX_LEN \
            --data_path $DATA_PATH \
            --output_dir $OUTPUT_DIR
    } 2>&1 | tee "$LOG_FILE"

    echo "[DONE] Model finished: $MODEL_NAME"
    echo "Log saved to $LOG_FILE"
    echo "Checkpoints saved to $OUTPUT_DIR"
    echo "-----------------------------------------------"
done

END_TIME=$(date +%s)
echo "Total runtime: $((END_TIME - START_TIME)) seconds"