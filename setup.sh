#!/bin/bash
set -e


echo "==== SimCSE Data + Environment Setup ===="

echo "Python version:"
python3 --version

echo "Pip version:"
pip --version

# ----------------- install packages ------------------
# Install required Python packages into current environment
# (Vast.ai main environment is OK; venv is optional)

echo "==== Using current Python environment (e.g., Vast.ai main venv) ===="
echo "Installing required Python packages into current env..."

pip install -U "transformers>=4.40" "tokenizers>=0.15" \
    "datasets>=2.16" "accelerate>=1.0.0" \
    sentencepiece scikit-learn pandas tqdm

# ------------------ English ------------------------

# 1. Download English NLI (Supervised)
echo "[1/3] Downloading English NLI data..."
bash data/download_scripts/download_nli_sup.sh

# 2. Download English Wiki (Unsupervised)
echo "[2/3] Downloading English Wiki unsupervised data..."
bash data/download_scripts/download_wiki_unsup.sh

# 3. Download English STS-B
echo "[3/3] Downloading STS-B data..."
bash data/download_scripts/download_sts_eng.sh

# ------------------ Chinese -----------------------

# echo, bash ...


# ------------------ Hindi ----------------------

# echo, bash ...

echo "==== All datasets downloaded successfully! ===="

echo "==== Setup complete. You can now run: ===="
echo "bash bash/train.sh"
echo "bash bash/evaluate.sh"