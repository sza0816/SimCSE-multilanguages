#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=gpu_test.out

module load anaconda3
conda activate simcse

python - << 'EOF'
import torch
print("cuda available:", torch.cuda.is_available())
print("gpu name:", torch.cuda.get_device_name(0))
EOF