#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --output=script_logging/slurm_%A.out

# Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0
module load Java/11.0.2

export OPENAI_API_KEY=''

srun $HOME/CIS_TSH_AUG/component1_query_rewriting/ConvRE/1_generate_gold_prl.py

