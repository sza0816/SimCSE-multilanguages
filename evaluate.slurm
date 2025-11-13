#!/bin/bash
#SBATCH --job-name=simcse_eval
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=eval_%x_%j.out
#SBATCH --error=eval_%x_%j.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

set -e

module load gcc/12.1.0
module load python/3.9.7

source /gpfs/scratch/zshang/simcse_env/bin/activate

echo "===== GPU ====="
nvidia-smi

cd /gpfs/home/zshang/simcse

# ----------------------------------------------------------
# Arguments
# ----------------------------------------------------------
TASK=$1       # unsup / sup
LANG=$2       # en / ch / hi

CKPT="/gpfs/scratch/zshang/ckpt_${TASK}_${LANG}"

if [ "$LANG" = "en" ]; then
    TESTFILE="data/sts-english-sample.csv"          # modify sts data later
elif [ "$LANG" = "ch" ]; then
    TESTFILE="data/sts-chinese-sample.csv"
elif [ "$LANG" = "hi" ]; then
    TESTFILE="data/sts-hindi-sample.csv"
else
    echo "Invalid LANG"
    exit 1
fi

echo "CKPT = $CKPT"
echo "TESTFILE = $TESTFILE"

python evaluate.py \
    --model_path $CKPT \
    --test_file $TESTFILE

echo "===== EVALUATION DONE ====="


# evaluate unsup english
# sbatch evaluate.slurm unsup en