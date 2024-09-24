#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --time=18:00:00
#SBATCH --output=script_logging/slurm_%A.out


module load 2022
module load Python/3.10.4-GCCcore-11.3.0
# module load Java/11.0.2
# export JVM_PATH=/home/hsoudani/java/jdk-21.0.4/lib/server/libjvm.so

srun $HOME/CIS_TSH_AUG/analysis_preliminary/corpus_graph_expriments_docs/doc_passage_mapping.py
srun $HOME/CIS_TSH_AUG/analysis_preliminary/retrieve_with_costum_bm25/bm25_retriever_costum.py

