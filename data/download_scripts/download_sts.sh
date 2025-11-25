#!/bin/bash
set -e

# Download STS-B (or STS) data for English, Chinese, and Hindi.
# English → GLUE STS-B
# Chinese → mteb/STSB (default)
# Hindi → SemRel/SemRel2024 (hin)

echo "Downloading multilingual STS evaluation data..."

LANGS=("english" "chinese" "hindi")

for LANG in "${LANGS[@]}"; do
    echo ""
    echo "---> Processing language: ${LANG}"

    if [[ "$LANG" == "english" ]]; then
        BASE_DIR="data/english/STS-B/original"
        mkdir -p "${BASE_DIR}"

        python3 - << 'EOF'
import os
from datasets import load_dataset

base_dir = "data/english/STS-B/original"
os.makedirs(base_dir, exist_ok=True)

print("Loading English GLUE STS-B ...")
ds = load_dataset("glue", "stsb")

split_map = {
    "train": "sts-train.tsv",
    "validation": "sts-dev.tsv",
    "test": "sts-test.tsv",
}

for split, filename in split_map.items():
    df = ds[split].to_pandas()
    df.to_csv(os.path.join(base_dir, filename), sep="\t", index=False)
    print(f"Wrote {split} split with {len(df)} rows.")
EOF

        echo "English STS-B saved to ${BASE_DIR}"

    elif [[ "$LANG" == "chinese" ]]; then
        BASE_DIR="data/chinese/STS-B"
        mkdir -p "${BASE_DIR}"

        python3 - << 'EOF'
import os
from datasets import load_dataset

base_dir = "data/chinese/STS-B"
os.makedirs(base_dir, exist_ok=True)

print("Loading Chinese STS-B (mteb/STSB)...")
ds = load_dataset("mteb/STSB", "default")

split_map = {
    "train": "sts-train.tsv",
    "validation": "sts-dev.tsv",
    "test": "sts-test.tsv",
}

for split, filename in split_map.items():
    df = ds[split].to_pandas()
    df.to_csv(os.path.join(base_dir, filename), sep="\t", index=False)
    print(f"Wrote {split} split with {len(df)} rows.")
EOF

        echo "Chinese STS-B saved to ${BASE_DIR}"

    elif [[ "$LANG" == "hindi" ]]; then
        BASE_DIR="data/hindi/STS-B"
        mkdir -p "${BASE_DIR}"

        python3 - << 'EOF'
import os
from datasets import load_dataset
import pandas as pd

base_dir = "data/hindi/STS-B"
os.makedirs(base_dir, exist_ok=True)

print("Loading Hindi STS dataset (SemRel/SemRel2024, hin)...")
ds = load_dataset("SemRel/SemRel2024", "hin")

def strip_outer_quotes(text):
    # Remove repeated outer quotes like """text""" or "text"
    if not isinstance(text, str):
        return text
    # Strip multiple layers if present
    while (text.startswith('"') and text.endswith('"')) or \
          (text.startswith('"""') and text.endswith('"""')):
        if text.startswith('"""') and text.endswith('"""'):
            text = text[3:-3]
        elif text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        else:
            break
    return text

# SemRel hin has: dev, test
split_map = {
    "dev": "sts-dev.tsv",
    "test": "sts-test.tsv",
}

for split, filename in split_map.items():
    if split not in ds:
        print(f"Split {split} not found, skipping.")
        continue
    df = ds[split].to_pandas()
    df["sentence1"] = df["sentence1"].apply(strip_outer_quotes)
    df["sentence2"] = df["sentence2"].apply(strip_outer_quotes)
    df.to_csv(os.path.join(base_dir, filename), sep="\t", index=False)
    print(f"Wrote {split} split with {len(df)} rows.")

print("Hindi STS (SemRel) written to data/hindi/STS-B/")
EOF

        echo "Hindi STS (SemRel2024) saved to ${BASE_DIR}"
    fi
done

echo ""
echo "All multilingual STS downloads finished."