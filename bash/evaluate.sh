#!/bin/bash
# Local Vast.ai evaluation script

set -e

###########################################################
#                 CONFIGURATION SECTION
###########################################################
TASK="unsup"          # unsup / sup
LANG="en"             # en / ch / hi

CKPT="./outputs/${LANG}/${TASK}/checkpoints"
OUTPUT_SUMMARY="./outputs/${LANG}/${TASK}/eval/summary.txt"
LOG_FILE="./outputs/${LANG}/${TASK}/eval/eval.log"
# STS-B evaluation files (TSV)
TASK_FILES=(
    "../../data/english/sts/original/sts-dev.tsv"
    "../../data/english/sts/original/sts-test.tsv"
    "../../data/english/sts/original/sts-train.tsv"
)

SUMMARY="${OUTPUT_SUMMARY}"
mkdir -p "./outputs/${LANG}/${TASK}/eval"
echo "===== Starting Evaluation =====" > $SUMMARY
echo "" >> $SUMMARY

echo "===== GPU INFO ====="
nvidia-smi | tee -a "$LOG_FILE"

echo "===== RUNNING EVALUATION ====="

for FILE in "${TASK_FILES[@]}"; do
    echo "===== Evaluating $(basename $FILE) ====="
    python evaluate.py --model_path $CKPT --test_file $FILE | tee tmp_eval.log | tee -a "$LOG_FILE"
    SCORE=$(grep "Spearman" tmp_eval.log | awk '{print $NF}')
    echo "$(basename $FILE)   $SCORE" >> $SUMMARY
    echo "$(basename $FILE)   $SCORE" >> "$LOG_FILE"
done

rm -f tmp_eval.log

echo "===== EVALUATION DONE ====="

# Run with:  bash bash/evaluate.sh