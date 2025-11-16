#!/bin/bash
set -e

# Download English NLI supervised data into data/english/sup

TARGET_DIR="data/english/sup"
mkdir -p "${TARGET_DIR}"

echo "Downloading NLI supervised data..."
wget -O "${TARGET_DIR}/nli_for_simcse.csv" \
  https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/nli_for_simcse.csv

echo "Done. File saved to ${TARGET_DIR}/nli_for_simcse.csv"