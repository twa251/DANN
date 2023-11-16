#!/bin/bash

#SBATCH --partition=wang
#SBATCH --gpus-per-node=rtxa5500:1
#SBATCH --nodes=1 --mem=128G --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%A.%a.out
#SBATCH --error=slurm-%A.%a.err
#SBATCH --mail-user=zeya.wang@uky.edu
#SBATCH --mail-type=ALL

module load cuda/12.1
conda info --envs
eval $(conda shell.bash hook)
source ~/miniconda/etc/profile.d/conda.sh
conda activate myenv
python3 /home/zwa281/DANN/dann_office.py --source webcam --target amazon --lr 0.005 --beta 0.1
