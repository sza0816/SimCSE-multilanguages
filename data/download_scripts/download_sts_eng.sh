#!/bin/bash
set -e

# Download English STS-B evaluation data into data/english/STS-B/original/

BASE_DIR="data/english"
TARGET_DIR="${BASE_DIR}/STS-B"
ORIGINAL_DIR="${TARGET_DIR}/original"

mkdir -p "${BASE_DIR}"

echo "Downloading STS-B.zip into ${BASE_DIR} ..."
wget -O "${BASE_DIR}/STS-B.zip" \
  https://dl.fbaipublicfiles.com/glue/data/STS-B.zip

echo "Unzipping..."
unzip "${BASE_DIR}/STS-B.zip" -d "${BASE_DIR}"

echo "Cleaning up zip file..."
rm "${BASE_DIR}/STS-B.zip"

# Keep only the original/ folder (which contains sts-dev/test/train.tsv)
# Remove duplicate files from STS-B root level
echo "Removing duplicate root-level STS-B files..."
rm -f "${TARGET_DIR}/dev.tsv" \
      "${TARGET_DIR}/test.tsv" \
      "${TARGET_DIR}/train.tsv" \
      "${TARGET_DIR}/LICENSE.txt" \
      "${TARGET_DIR}/readme.txt" 2>/dev/null || true

echo "Done. STS-B data saved under ${ORIGINAL_DIR}"