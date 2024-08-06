### ==============================================================================
### Ref: https://github.com/fengranMark/HAConvDR
### ==============================================================================

# === For running on server - interactive mode
# ssh gcn1
# module load 2022
# module load IPython/8.5.0-GCCcore-11.3.0
# module load Java/11.0.2

import io
import json, os, csv
from tqdm import tqdm
from argparse import ArgumentParser

csv.field_size_limit(10**8)
id_col= 0
text_col= 1
title_col = 2

def remove_nul_lines(file):
    for line in file:
        if '\x00' not in line:
            yield line

def convert_to_pyserini_file(args):
    corpus_file = f"corpus/{args.dataset_name}/full_collection_segments.tsv"
    output_file = f"corpus/{args.dataset_name}/full_collection_segments_pyserini_format.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(corpus_file, 'r') as input:
        filtered_file = remove_nul_lines(input)
        
        reader = csv.reader(filtered_file, delimiter="\t")
        with open(output_file, 'w') as output:
            for i, row in enumerate(tqdm(reader)):
                if row[id_col] == "id":
                    continue
                text = row[text_col]
                
                if args.dataset_name == "QReCC":
                    obj = {"contents": text, "id": f"doc{row[id_col]}"}
                else:
                    title = row[title_col]
                    title = ' '.join(title.split(' [SEP] '))
                    obj = {"contents": " ".join([title, text]), "id": f"doc{row[id_col]}"}
                output.write(json.dumps(obj, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="QReCC", choices=["TopiOCQA", "INSCIT", "QReCC"])
    args = parser.parse_args()

    # = Step 1) Convert corpus to pyserini file
    convert_to_pyserini_file(args)
    
    # = Step 2-1) BM25 corpus indexing using pyserini
    # python -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator -threads 20 -input "corpus/QReCC" -index "corpus/QReCC/bm25_index" -storePositions -storeDocvectors -storeRaw
        
    # = Step 2-2) ANCE corpus indexing using pyserini
    # python -m pyserini.encode \
    #   input   --corpus corpus/TopiOCQA/full_wiki_segments_pyserini_format.jsonl \
    #           --fields text \
    #           --delimiter "\n" \
    #           --shard-id 0 \
    #           --shard-num 1 \
    #   output  --embeddings corpus/TopiOCQA/ance_index \
    #           --to-faiss \
    #   encoder --encoder castorini/ance-msmarco-passage \
    #           --fields text \
    #           --batch 32 \
    #           --fp16
    
    
# python component0_preprocessing/corpus_indexing/bm25_corpus_indexing.py
