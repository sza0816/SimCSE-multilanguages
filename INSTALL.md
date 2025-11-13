# Environment Setup

If you are using SeaWulf!!

Do NOT install packages in your home directory.
Always create the environment in SCRATCH folder:

module load gcc/12.1.0
module load python/3.9.7

python -m venv /gpfs/scratch/$USER/simcse_env
source /gpfs/scratch/$USER/simcse_env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt