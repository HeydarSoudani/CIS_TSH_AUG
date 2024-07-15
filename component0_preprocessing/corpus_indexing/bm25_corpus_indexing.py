### ==============================================================================
### Ref: https://github.com/fengranMark/HAConvDR
### ==============================================================================

# === For running on server - interactive mode
# ssh gcn1
# module load 2022
# module load IPython/8.5.0-GCCcore-11.3.0
# module load Java/11.0.2

import gdown
import zipfile
import pathlib
import json, os, csv
from tqdm import tqdm
from argparse import ArgumentParser


csv.field_size_limit(10**6)
id_col= 0
text_col= 1
title_col = 2


def inscit_corpus_download_unzip():
    file_id = '1rwQNmvJIRpBEs58ewcrxczZYPOa0nuQS'
    zip_file = 'corpus/INSCIT/corpus.zip'
    extract_to = 'corpus/INSCIT'
    
    # == Step 1) Download 
    gdown.download(f'https://drive.google.com/uc?id={file_id}&confirm=t', zip_file, quiet=False)
    
    # == Step 2) Unzip  
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def convert_corpus_to_tsv():
    # Src: https://github.com/ellenmellon/INSCIT/blob/main/models/DPR/create_corpus_tsv.py
    raw_corpus_dir = 'corpus/INSCIT/text_0420_processed'
    output_file = 'corpus/INSCIT/full_wiki_segments.tsv'
    
    def process_file(user_filename, output_file):
        out_dir = os.path.dirname(output_file)
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        
        with open(user_filename, 'rb') as source, open(output_file, 'a') as dest:
            # Load the json objects from file and then open the new tsv
            # file
            tsv_writer = csv.writer(dest, delimiter='\t', lineterminator='\n')
            tsv_writer.writerow(['id', 'text', 'title']) # set headers
            json_reader = json.load(source)

            # Move the data from json format to tsv
            for json_object in json_reader:
                current_title = json_object['title'].strip()

                # Parsing the text from each json object in the file
                for pid, passage in enumerate(json_object['passages']):
                    new_row = []
                    cur_id = passage['id']
                    new_row.append(cur_id)
                    new_row.append(' '.join(passage['text'].split()))
                    
                    # Use all subtitles
                    new_row.append(" [SEP] ".join(passage['titles']))
                    tsv_writer.writerow(new_row)
    
    for filename in os.listdir(raw_corpus_dir):
        print('processing', filename)
        dir_or_file = os.path.join(raw_corpus_dir, filename)
        if os.path.isfile(dir_or_file):
            process_file(dir_or_file, output_file)

def convert_to_pyserini_file(args):
    corpus_file = f"corpus/{args.dataset_name}/full_wiki_segments.tsv"    
    output_file = f"corpus/{args.dataset_name}/full_wiki_segments_pyserini_format.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(corpus_file, 'r') as input:
        reader = csv.reader(input, delimiter="\t")
        with open(output_file, 'w') as output:
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
    # parser.add_argument("--wiki_file", type=str, default=WIKI_FILE)
    # parser.add_argument("--output_file", type=str, default=OUTPUT_FILE)
    parser.add_argument("--dataset_name", type=str, default="INSCIT", choices=["TopiOCQA", "INSCIT", "qrecc"])
    args = parser.parse_args()
    
    # = Step 1) needed libraries
    # pip install -q faiss-gpu==1.7.2
    # pip install pyserini==0.16
    # pip install -q pytrec_eval
    
    
    # = Step 2) download corpus
    # ===== For TopiOCQA =======
    # wget https://zenodo.org/records/6149599/files/data/wikipedia_split/full_wiki_segments.tsv -O datasets/topiocqa/full_wiki_segments.tsv
    # ===== For INSCIT =========
    # inscit_corpus_download_unzip()
    # convert_corpus_to_tsv()
    
    
    
    # = Step 3) Convert corpus to pyserini file
    convert_to_pyserini_file(args)
    
    
    # = Step 4) Index corpus using pyserini
    # python -m pyserini.index -collection JsonCollection \
    #                         -generator DefaultLuceneDocumentGenerator \
    #                         -threads 20 \
    #                         -input "corpus/INSCIT" \
    #                         -index "corpus/INSCIT/bm25_index" \
	# 						  -storePositions -storeDocvectors -storeRaw

