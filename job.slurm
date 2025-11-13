#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=00:03:00
#SBATCH --output=test_gpu.out

module load gcc/12.1.0
module load python/3.9.7

source /gpfs/scratch/zshang/simcse_env/bin/activate

python - << 'EOF'
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0))
EOF