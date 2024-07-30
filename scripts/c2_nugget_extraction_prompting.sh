#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --output=script_logging/slurm_%A.out

module load 2022
module load Python/3.10.4-GCCcore-11.3.0
# pip install vllm

export HF_TOKEN='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'
huggingface-cli login --token $HF_TOKEN

srun $HOME/CIS_TSH_AUG/component2_nugget_generation/prompt_based.py
