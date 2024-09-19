#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --time=4:00:00
#SBATCH --output=script_logging/slurm_%A.out


module load 2022
module load Python/3.10.4-GCCcore-11.3.0
module load Java/11.0.2

pip install --upgrade pyserini

python -m pyserini.encode \
  input   --corpus corpus/QReCC/full_collection_segments_pyserini_format_1.jsonl \
          --fields text \
          --delimiter "\n\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t" \
          --shard-id 0 \
          --shard-num 20 \
  output  --embeddings corpus/QReCC/ance_index_0 \
          --to-faiss \
  encoder --encoder castorini/ance-msmarco-passage \
          --fields text \
          --batch 8 \
          --fp16

