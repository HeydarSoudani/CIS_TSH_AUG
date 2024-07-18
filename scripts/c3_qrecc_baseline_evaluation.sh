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

# pip install git+https://github.com/huggingface/transformers
srun $HOME/CIS_TSH_AUG/component3_retriever/bm25/qrecc_baseline_t5_evaluation.py


# dataset_name = ["topiocqa", "inscit", "qrecc"]
# query_format = ['original', 'human_rewritten', 'all_history', 'same_topic']