#!/bin/bash
#SBATCH --job-name=simcse_eval
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=simcse_eval_%j.out
#SBATCH --error=simcse_eval_%j.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

set -e

###########################################################
#                 CONFIGURATION SECTION
###########################################################
TASK="unsup"          # unsup / sup
LANG="en"             # en / ch / hi

SCRATCH="/gpfs/scratch/zshang/simcse"
CKPT="${SCRATCH}/ckpt_${TASK}_${LANG}"

if [[ "$LANG" == "en" ]]; then
    TESTFILE="${SCRATCH}/data/sts-english-sample.csv"
elif [[ "$LANG" == "ch" ]]; then
    TESTFILE="${SCRATCH}/data/sts-chinese-sample.csv"
elif [[ "$LANG" == "hi" ]]; then
    TESTFILE="${SCRATCH}/data/sts-hindi-sample.csv"
else
    echo "Invalid LANG option."
    exit 1
fi

###########################################################
echo "===== LOADING MODULES ====="
module load gcc/12.1.0
module load python/3.9.7

echo "===== ACTIVATE ENV ====="
source /gpfs/scratch/zshang/simcse_env/bin/activate

echo "===== GPU INFO ====="
nvidia-smi

echo "===== RUNNING EVALUATION ====="
cd /gpfs/home/zshang/simcse

python evaluate.py \
    --model_path $CKPT \
    --test_file $TESTFILE

echo "===== EVALUATION DONE ====="


# sbatch evaluate.slurm