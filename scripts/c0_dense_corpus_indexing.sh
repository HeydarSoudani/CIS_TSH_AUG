
module load 2022
module load Python/3.10.4-GCCcore-11.3.0


python $HOME/CIS_TSH_AUG/component0_preprocessing/dense_corpus_indexing.py \
    --raw_collection_path "/kaggle/input/full-wiki-100k/full_wiki_segments_100k.tsv" \
    --tokenized_output_path "dense_tokenized" \
    --embedded_output_path "dense_embedded"