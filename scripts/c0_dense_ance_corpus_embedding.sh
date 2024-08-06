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
module load Java/11.0.2

python -m pyserini.encode \
  input   --corpus corpus/TopiOCQA/full_wiki_segments_pyserini_format_0.5m.jsonl \
          --fields text \
          --delimiter "\n" \
          --shard-id 0 \
          --shard-num 1 \
  output  --embeddings corpus/TopiOCQA/ance_index_20sh \
          --to-faiss \
  encoder --encoder castorini/ance-msmarco-passage \
          --fields text \
          --batch 32 \
          --fp16