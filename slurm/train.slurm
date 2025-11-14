#!/bin/bash
#SBATCH --job-name=simcse_unsup_en
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=simcse_%j.out
#SBATCH --error=simcse_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -e

###########################################################
#                 CONFIGURATION SECTION
###########################################################
TASK="unsup"                     # unsup / sup （only run unsup now）
LANG="en"                        # en / ch / hi
MODEL_NAME="bert-base-uncased"
EPOCHS=0.02
BATCH_SIZE=64
LR=5e-5

# data path in scratch - only for english unsup for now
DATA_PATH="/gpfs/scratch/zshang/simcse/data/wiki1m_for_simcse.txt"

# checkpoint in scratch 
OUTPUT_DIR="/gpfs/scratch/zshang/simcse/ckpt_unsup_en"
###########################################################


echo "===== LOADING MODULES ====="
module load gcc/12.1.0
module load python/3.9.7

echo "===== ACTIVATE ENV ====="
source /gpfs/scratch/zshang/simcse_env/bin/activate
python --version

echo "===== GPU INFO ====="
nvidia-smi

echo "===== SET HF CACHE ====="
export HF_HOME=/gpfs/scratch/zshang/hf_cache
mkdir -p $HF_HOME

echo "===== ENTER PROJECT DIR ====="
cd /gpfs/home/zshang/simcse
echo "PWD = $(pwd)"

###########################################################
#       OPTIONAL: Auto-select model by language
###########################################################
if [[ "$LANG" == "en" ]]; then
    MODEL_NAME="bert-base-uncased"
elif [[ "$LANG" == "ch" ]]; then
    MODEL_NAME="bert-base-chinese"
elif [[ "$LANG" == "hi" ]]; then
    MODEL_NAME="google/muril-base-cased"
fi
echo "MODEL = $MODEL_NAME"

###########################################################
#                     RUN TRAINING
###########################################################
python train_unsup.py \
    --model_name $MODEL_NAME \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR

echo "===== TRAINING DONE ====="

# sbatch /gpfs/home/zshang/simcse/slurm/train.slurm
