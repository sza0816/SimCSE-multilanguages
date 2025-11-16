#!/bin/bash
set -e

# Download English unsupervised wiki data into data/english/unsup/

TARGET_DIR="data/english/unsup"
mkdir -p "${TARGET_DIR}"

echo "Downloading wiki1m_for_simcse.txt ..."
wget -O "${TARGET_DIR}/wiki1m_for_simcse.txt" \
  https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/wiki1m_for_simcse.txt

echo "Done. Saved to ${TARGET_DIR}/wiki1m_for_simcse.txt"