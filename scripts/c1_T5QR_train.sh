#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --time=6:00:00
#SBATCH --output=script_logging/slurm_%A.out

# Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0
# module load IPython/8.5.0-GCCcore-11.3.0

srun $HOME/CIS_TSH_AUG/component1_query_rewriting/T5QR/train_t5qr.py
