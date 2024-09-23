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

srun $HOME/CIS_TSH_AUG/component0_preprocessing/topiocqa_files_preprocessing/3_create_document_corpus.py
# python -m pyserini.index.lucene -collection JsonCollection -generator DefaultLuceneDocumentGenerator -threads 20 -input "corpus/TopiOCQA/doc_corpus" -index "corpus/TopiOCQA/doc_corpus_bm25_index" -storePositions -storeDocvectors -storeRaw

