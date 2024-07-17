### ==================================================================================
# Src: https://github.com/ellenmellon/INSCIT/blob/main/models/DPR/create_corpus_tsv.py
### ==================================================================================

import gdown
import zipfile
import pathlib
import json, os, csv

csv.field_size_limit(10**8)

def inscit_corpus_download_unzip():
    file_id = '1rwQNmvJIRpBEs58ewcrxczZYPOa0nuQS'
    zip_file = 'corpus/INSCIT/corpus.zip'
    extract_to = 'corpus/INSCIT'
    
    # == Step 1) Download 
    gdown.download(f'https://drive.google.com/uc?id={file_id}&confirm=t', zip_file, quiet=False)
    
    # == Step 2) Unzip  
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def inscit_convert_corpus_to_tsv():
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


if __name__ == "__main__":
    # ===== Step 1) download files ==============
    inscit_corpus_download_unzip()
    inscit_convert_corpus_to_tsv()