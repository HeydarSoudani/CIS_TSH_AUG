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

# pip install git+https://github.com/huggingface/transformers
srun $HOME/TAConvDR/component3_retriever/bm25/evaluation.py \
    --index_dir_path "corpus/indexes" \
    --result_qrel_path "component3_retriever/results" \
    --gold_qrel_path "component3_retriever/data/topiocqa/dev/qrel_gold.trec" \
    --dataset_name "topiocqa" \
    --query_format "human_rewritten" \
    --seed 42


# dataset_name = ["topiocqa", "inscit", "qrecc"]
# query_format = ['original', 'human_rewritten', 'all_history', 'same_topic']