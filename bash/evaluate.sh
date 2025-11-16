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
# STS-B evaluation files (TSV)
TASK_FILES=(
    "../data/sts_data/sts-dev.tsv"
    "../data/sts_data/sts-test.tsv"
    "../data/sts_data/sts-train.tsv"
)

SUMMARY="${OUTPUT_SUMMARY}"
mkdir -p "./outputs/${LANG}/${TASK}/eval"
echo "===== Starting Evaluation =====" > $SUMMARY
echo "" >> $SUMMARY

echo "===== GPU INFO ====="
nvidia-smi

echo "===== RUNNING EVALUATION ====="

for FILE in "${TASK_FILES[@]}"; do
    echo "===== Evaluating $(basename $FILE) ====="
    python evaluate.py --model_path $CKPT --test_file $FILE | tee tmp_eval.log
    SCORE=$(grep "Spearman" tmp_eval.log | awk '{print $NF}')
    echo "$(basename $FILE)   $SCORE" >> $SUMMARY
done

rm -f tmp_eval.log

echo "===== EVALUATION DONE ====="

# Run with:  bash evaluate.sh