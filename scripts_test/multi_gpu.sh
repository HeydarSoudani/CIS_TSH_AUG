#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --mem=64G
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=script_logging/slurm_%A.out




# Load the necessary modules
# module load cuda/11.8
# module load python/3.9

# Activate the virtual environment or conda environment if needed
# source /path/to/your/env/bin/activate

# Print the GPU devices allocated
# echo "Allocated GPUs:"
# nvidia-smi

# Run your inference script
# python run_llama_inference.py
