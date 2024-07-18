#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --time=5:00:00
#SBATCH --output=script_logging/slurm_%A.out


module load 2022
module load Python/3.10.4-GCCcore-11.3.0
module load IPython/8.5.0-GCCcore-11.3.0
module load Java/11.0.2

python -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator -threads 20 -input "corpus/QReCC" -index "corpus/QReCC/bm25_index" -storePositions -storeDocvectors -storeRaw
