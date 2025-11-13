#!/bin/bash
#SBATCH --job-name=simcse
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=train_log.out
#SBATCH --error=train_err.out

# 1. Load modules
module load anaconda3
source activate simcse

# 2. (Optional) Print environment information
echo "Running on GPU:"
nvidia-smi

# 3. Start training
python train.py