#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=gpu_test.out

module load gcc/12.1.0
module load python/3.9.7

source ~/simcse_env/bin/activate

python - << 'EOF'
import torch
print("cuda available:", torch.cuda.is_available())
print("gpu name:", torch.cuda.get_device_name(0))
EOF

deactivate
module purge