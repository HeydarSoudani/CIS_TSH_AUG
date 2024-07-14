### ==============================================================================
### Ref: https://github.com/fengranMark/HAConvDR
### ==============================================================================

# === For running on server - interactive mode
# ssh gcn1
# module load 2022
# module load IPython/8.5.0-GCCcore-11.3.0
# module load Java/11.0.2


import json, os, csv
from tqdm import tqdm
from argparse import ArgumentParser

WIKI_FILE = "corpus/TopiOCQA/full_wiki_segments.tsv"
OUTPUT_FILE = "corpus/TopiOCQA/full_wiki_segments_pyserini_format.jsonl"
id_col= 0
text_col= 1
title_col = 2

def convert_to_pyserini_file(args):
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    with open(args.wiki_file, 'r') as input:
        reader = csv.reader(input, delimiter="\t")
        with open(args.output_file, 'w') as output:
            for i, row in enumerate(tqdm(reader)):
                if row[id_col] == "id":
                    continue
                title = row[title_col]
                text = row[text_col]
                title = ' '.join(title.split(' [SEP] '))
                obj = {"contents": " ".join([title, text]), "id": f"doc{i}"}
                output.write(json.dumps(obj, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--wiki_file", type=str, default=WIKI_FILE)
    parser.add_argument("--output_file", type=str, default=OUTPUT_FILE)
    args = parser.parse_args()
    
    # = Step 1) needed libraries
    # pip install -q faiss-gpu==1.7.2
    # pip install pyserini==0.16
    # pip install -q pytrec_eval
    
    # = Step 2) download corpus
    # wget https://zenodo.org/records/6149599/files/data/wikipedia_split/full_wiki_segments.tsv -O datasets/topiocqa/full_wiki_segments.tsv
    
    # = Step 3) Convert corpus to pyserini file
    # python component0_preprocessing/bm25_corpus_indexing.py \
    #     --wiki_file "corpus/TopiOCQA/full_wiki_segments.tsv" \
    #     --output_file "corpus/TopiOCQA/full_wiki_segments_pyserini_format.jsonl"
    convert_to_pyserini_file(args)
    
    # = Step 4) Index corpus using pyserini
    # python -m pyserini.index -collection JsonCollection \
    #                         -generator DefaultLuceneDocumentGenerator \
    #                         -threads 20 \
    #                         -input "corpus/TopiOCQA" \
    #                         -index "corpus/indexes" \
	# 						  -storePositions -storeDocvectors -storeRaw
    