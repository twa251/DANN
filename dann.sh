#!/bin/bash

#SBATCH --partition=wang
#SBATCH --gpus-per-node=rtxa5500:1
#SBATCH --nodes=1 --mem=80G --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=slurm-%A.%a.out
#SBATCH --error=slurm-%A.%a.err
#SBATCH --mail-user=zeya.wang@uky.edu 
#SBATCH --mail-type=ALL

python3 /home/zwa281/DANN/dann_office.py --source amazon --target webcam --lr 0.01 -beta 0.1

