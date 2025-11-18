#!/bin/bash
# Local Vast.ai evaluation script for English SimCSE (unsup / sup)

set -e

# ------------------------CONFIGURATION SECTION------------------------
TASK="unsup"          # unsup / sup
LANG="en"             # en / ch / hi (currently only english STS-B wired up)


CKPT="./outputs/${LANG}/${TASK}/checkpoints"

EVAL_DIR="./outputs/${LANG}/${TASK}/eval"
OUTPUT_SUMMARY="${EVAL_DIR}/summary.txt"
LOG_FILE="${EVAL_DIR}/eval.log"

# STS-B evaluation files
TASK_FILES=(
    "./data/english/STS-B/original/sts-dev.tsv"
    "./data/english/STS-B/original/sts-test.tsv"
    "./data/english/STS-B/original/sts-train.tsv"
)

mkdir -p "${EVAL_DIR}"
SUMMARY="${OUTPUT_SUMMARY}"

# Redirect all output to log (stdout and stderr)
mkdir -p "${EVAL_DIR}"
: > "$LOG_FILE"
exec > >(tee "$LOG_FILE") 2>&1

# ------------------------HEADER------------------------

echo "===== Starting Evaluation (${TASK}, ${LANG}) =====" > "$SUMMARY"
echo "" >> "$SUMMARY"

echo "===== GPU INFO ====="
nvidia-smi

echo "===== RUNNING EVALUATION ====="

# ------------------------LOOP OVER STS-B FILES------------------------

for FILE in "${TASK_FILES[@]}"; do
    BASENAME=$(basename "$FILE")
    echo "===== Evaluating ${BASENAME} ====="

    # evaluate.py will:
    #   * load the backbone (bert-base-uncased by default or from config.txt)
    #   * load our contrastive head weights from ${CKPT}/pytorch_model.bin
    #   * compute sentence embeddings + Spearman corr.
    python evaluate_english.py \
        --model_path "$CKPT" \
        --test_file "$FILE" | tee tmp_eval.log

    SCORE=$(grep "Spearman" tmp_eval.log | awk '{print $NF}')
    echo "${BASENAME}   ${SCORE}" >> "$SUMMARY"
    echo "${BASENAME}   ${SCORE}" >> "$LOG_FILE"
    echo "" >> "$SUMMARY"
    echo "" >> "$LOG_FILE"

done

rm -f tmp_eval.log

echo "===== EVALUATION DONE (${TASK}, ${LANG}) ====="

echo "\nSummary written to: $SUMMARY"

# Run with:
#   bash bash/eval_english.sh                 # use defaults (unsup / en)


# Note: sts-test.tsv has no gold labels in our local copy, so Spearman may
#       appear as NaN or be skipped depending on evaluate_english.py logic.
# Note: Original SimCSE used full Wikipedia dump; with wiki1m_for_simcse.txt
#       we expect lower Spearman, which is normal. 
#       Remember to compare our result with paper, ~0.8, to say it is reasonable. 

# Unsupervised result: 
# Spearman = 0.6854 for dev, 0.6263 for train, see outputs/en/unsup/eval

