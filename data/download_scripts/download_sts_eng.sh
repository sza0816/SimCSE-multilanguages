#!/bin/bash
set -e

# Download English STS-B evaluation data into data/english/STS-B/original/

BASE_DIR="data/english/STS-B/original"

echo "Preparing STS-B data in ${BASE_DIR} ..."
mkdir -p "${BASE_DIR}"

python - << 'EOF'
import os
from datasets import load_dataset

base_dir = "data/english/STS-B/original"
os.makedirs(base_dir, exist_ok=True)

print("Loading GLUE STS-B from Hugging Face ...")

ds = load_dataset("glue", "stsb")

split_map = {
    "train": "sts-train.tsv",
    "validation": "sts-dev.tsv",
    "test": "sts-test.tsv",
}

for split, filename in split_map.items():
    df = ds[split].to_pandas()
    out_path = os.path.join(base_dir, filename)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {split} split to {out_path} with {len(df)} rows")

print("All STS-B splits written.")
EOF

echo "Done. STS-B data saved under ${BASE_DIR}"



# to be modified !!