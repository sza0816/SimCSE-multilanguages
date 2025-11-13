#!/bin/bash
#SBATCH --job-name=simcse
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=simcse_%x_%j.out
#SBATCH --error=simcse_%x_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -e

echo "===== LOADING MODULES ====="
module load gcc/12.1.0
module load python/3.9.7

echo "===== ACTIVATE ENV ====="
source /gpfs/scratch/zshang/simcse_env/bin/activate
echo "PYTHON = $(which python)"
python --version

echo "===== GPU INFO ====="
nvidia-smi

echo "===== SET HF CACHE ====="
export HF_HOME=/gpfs/scratch/zshang/hf_cache
mkdir -p $HF_HOME

echo "===== ENTER PROJECT DIR ====="
cd /gpfs/home/zshang/simcse
echo "PWD = $(pwd)"

# ----------------------------------------------------------
# Arguments
# ----------------------------------------------------------
TASK=$1          # unsup / sup
LANG=$2          # en / ch / hi
EPOCHS=$3        # 1 / 0.1 / 3 ...
BS=$4            # 64 / 32 ...
LR=$5            # 5e-5 / 1e-5 ...
DATA_PATH=$6     # dataset path in repo
# e.g. data/wiki1m_for_simcse.txt

echo "[ARGUMENTS]"
echo "TASK = $TASK"
echo "LANG = $LANG"
echo "EPOCHS = $EPOCHS"
echo "BS = $BS"
echo "LR = $LR"
echo "DATA_PATH = $DATA_PATH"

# ----------------------------------------------------------
# Output directory in SCRATCH
# ----------------------------------------------------------
OUTDIR="/gpfs/scratch/zshang/ckpt_${TASK}_${LANG}"
mkdir -p $OUTDIR
echo "OUTPUT DIR = $OUTDIR"

# ----------------------------------------------------------
# Run Training
# ----------------------------------------------------------
if [ "$TASK" = "unsup" ]; then
    if [ "$LANG" = "en" ]; then
        MODEL="bert-base-uncased"
    elif [ "$LANG" = "ch" ]; then
        MODEL="bert-base-chinese"
    elif [ "$LANG" = "hi" ]; then
        MODEL="google/muril-base-cased"
    else
        echo "Invalid LANG"
        exit 1
    fi

    python train_unsup.py \
        --model_name $MODEL \
        --epochs $EPOCHS \
        --batch_size $BS \
        --lr $LR \
        --data_path $DATA_PATH \
        --output_dir $OUTDIR

elif [ "$TASK" = "sup" ]; then
    if [ "$LANG" = "en" ]; then
        MODEL="bert-base-uncased"
    elif [ "$LANG" = "ch" ]; then
        MODEL="bert-base-chinese"
    elif [ "$LANG" = "hi" ]; then
        MODEL="google/muril-base-cased"
    else
        echo "Invalid LANG"
        exit 1
    fi

    python train_sup.py \
        --model_name $MODEL \
        --epochs $EPOCHS \
        --batch_size $BS \
        --lr $LR \
        --data_path $DATA_PATH \
        --output_dir $OUTDIR
else
    echo "Invalid TASK"
    exit 1
fi

echo "===== TRAINING DONE ====="


# Example commands:
# sbatch train.slurm unsup en 1 64 5e-5 data/wiki1m_for_simcse.txt
# sbatch train.slurm sup en 1 32 5e-5 data/nli_for_simcse.csv