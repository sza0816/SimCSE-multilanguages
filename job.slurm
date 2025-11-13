#!/bin/bash
#SBATCH --job-name=simcse_unsup
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=simcse_unsup.out
#SBATCH --error=simcse_unsup.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

echo "===== LOADING MODULES ====="
module load gcc/12.1.0
module load python/3.9.7

echo "===== ACTIVATING ENV ====="
source /gpfs/scratch/zshang/simcse_env/bin/activate
echo "PYTHON = $(which python)"
python --version

echo "===== CHANGE TO PROJECT DIRECTORY ====="
cd /gpfs/home/zshang/simcse
echo "PWD = $(pwd)"

echo "===== START TRAINING ====="
# training
# python train_unsup.py --epochs 1 --batch_size 64 --lr 5e-5

# local testing
python train_unsup.py --epochs 0.1 --batch_size 32 --lr 3e-5

# evaluation
# python evaluate.py --model_path ckpt_unsup --test_file data/sts-english-sample.csv --device cuda


echo "===== JOB DONE ====="