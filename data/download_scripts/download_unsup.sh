#!/bin/bash
set -e

# Download English unsupervised wiki data into data/english/unsup/

TARGET_DIR="data/english/unsup"
mkdir -p "${TARGET_DIR}"

echo "Downloading wiki_english.txt ..."
wget -O "${TARGET_DIR}/wiki_english.txt" \
  https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/wiki1m_for_simcse.txt

echo "Done. Saved to ${TARGET_DIR}/wiki_english.txt"


# ---------------- Hindi wiki unsup ----------------
TARGET_DIR_HI="data/hindi/unsup"
mkdir -p "${TARGET_DIR_HI}"

echo "Downloading hindi_corpus.txt ..."
wget -O "${TARGET_DIR_HI}/wiki_hindi.txt" \
  https://huggingface.co/Sindhuuu12/hindi-wikipedia-corpus/resolve/main/hindi_corpus.txt

echo "Done. Saved to ${TARGET_DIR_HI}/wiki_hindi.txt"