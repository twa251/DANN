#!/bin/bash

#SBATCH --partition=wang
#SBATCH --gpus-per-node=rtxa5500:8
#SBATCH --nodes=1 --mem=128G --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm-%A.%a.out
#SBATCH --error=slurm-%A.%a.err
#SBATCH --mail-user=twa251@uky.edu 
#SBATCH --mail-type=ALL

module load cuda/12.1
conda info --envs
eval $(conda shell.bash hook)
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pylava10

python3 /home/twa251/DANN/dann_office.py --source amazon --target webcam --lr 0.01 --beta 0.1

