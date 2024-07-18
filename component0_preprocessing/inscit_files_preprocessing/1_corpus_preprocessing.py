#!/usr/bin/env python3

### ==================================================================================
# Src: https://github.com/ellenmellon/INSCIT/blob/main/models/DPR/create_corpus_tsv.py
### ==================================================================================

import gdown
import random
import sqlite3
import pathlib
import zipfile
import requests
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

def inscit_query_files_download():

    dataset_subsec = "train"
    file_url = f"https://raw.githubusercontent.com/ellenmellon/INSCIT/main/data/{dataset_subsec}.json"
    save_path = f"corpus/INSCIT/{dataset_subsec}.json"
    response = requests.get(file_url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded and saved as {save_path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

def train_test_file_to_qrecc_format():
    
    # === Define files ========================
    dataset_subsec = "dev"
    input_file = f"corpus/INSCIT/{dataset_subsec}.json"
    # output_file = f"processed_datasets/INSCIT/{dataset_subsec}_qrecc_format.json"
    output_file = f"processed_datasets/INSCIT/{dataset_subsec}_new_qrecc_format.json"
    
    # === Define T5-rewritting file ============
    t5_rewritten_file = f"component3_retriever/input_data/INSCIT/T5QR/{dataset_subsec}_t5_rewrite.json"
    t5_rewritten = {}
    with open(t5_rewritten_file, 'r') as file:
        t5_data = json.load(file)
    for item in t5_data:
        t5_rewritten[item["sample_id"]] = item["t5_rewrite"]
    
    # === Loading t2id =========================
    print("Loading t2id file ...")
    with open("corpus/INSCIT/title2ids.json", 'r') as file:
        title2ids_data = json.load(file)
    
    def get_value_with_exception_handling(key):
        try:
            return title2ids_data[key]
        except KeyError:
            print(f"Key {key} not found")
            return 0
    
    # === Loading t2id =========================
    if dataset_subsec in ["train", "dev"]:
        print("Handling collection ...")
        collection_file = "corpus/INSCIT/full_wiki_segments_pyserini_format.jsonl"
        db_path = "corpus/INSCIT/full_wiki_segments.db"

        def load_data_into_db(file_path, db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS records (
                    id TEXT PRIMARY KEY,
                    contents TEXT
                )
            ''')
            
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    collection_data = json.loads(line.strip())
                    cursor.execute('INSERT OR IGNORE INTO records (id, contents) VALUES (?, ?)', 
                                (collection_data['id'], collection_data['contents']))
            
            conn.commit()
            conn.close()
        
        def id2text(db_path, id):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT contents FROM records WHERE id = ?', (id,))
            result = cursor.fetchone()
            
            conn.close()
            
            if result:
                return result[0]
            else:
                return "ID not found"
        
        # load_data_into_db(collection_file, db_path)
         
    
    # === Converting ===========================
    print("Converting ...")
    with open(input_file, 'r') as file:
        row_data = json.load(file)
    
    with open(output_file, "w") as of:
        for conv_idx, (conv_id, conv_sample) in enumerate(row_data.items()):
            
            if conv_idx % 10 == 0:
                print(f"processing {conv_idx+1} conversations ...")
            
            turns = conv_sample['turns']
            for turn_idx, turn in enumerate(turns):
                sample_id = f"{conv_idx+1}_{turn_idx+1}"
                query = turn["context"][-1]
                answer = turn["labels"][0]["response"]
                p_titles = [e["passage_id"] for e in turn["labels"][0]["evidence"]]
                p_texts = [e["passage_text"] for e in turn["labels"][0]["evidence"]]
                pids = [get_value_with_exception_handling(title) for title in p_titles]

                if dataset_subsec in ["train", "dev"]:
                    history_uttr = turn["context"][:-1]
                    t5_rewrite = t5_rewritten[sample_id]
                    
                    # random negatives
                    neg_nums = 5
                    full_range = set(range(49000000))
                    exclude_pids = {int(pid[3:]) for pid in pids}
                    remaining_numbers = list(full_range - exclude_pids)
                    neg_random_ids = random.sample(remaining_numbers, neg_nums)
                    neg_random_pids = [f"doc{str(id)}" for id in neg_random_ids]
                    neg_random_text = [id2text(db_path, key) for key in neg_random_pids]
                    
                    item = {
                        "sample_id": sample_id,
                        "cur_utt_text": query,
                        "oracle_utt_text": t5_rewrite,
                        "cur_response_text": answer,
                        "ctx_utts_text": history_uttr,
                        "pos_docs_pids": pids,
                        "pos_docs_text": p_texts,
                        "random_neg_docs_pids": neg_random_pids, 
                        "random_neg_docs_text": neg_random_text
                    }
                    
                elif dataset_subsec == "test":
                    history_query = turn["context"][:-1:2]
                    history_answer = turn["context"][1:-1:2]             
                    item = {
                        "sample_id": sample_id,
                        "cur_utt_text": query,
                        "oracle_utt_text": "",
                        "cur_response_text": answer,
                        "ctx_utts_text": history_query,
                        "ctx_resps_text": history_answer,
                        "pos_docs_pids": pids
                    }
                
                of.write(json.dumps(item) + '\n')

def train_to_qrecc_format_for_convgqr():
    # === Define files ========================
    dataset_subsec = "train"
    input_file = f"processed_datasets/INSCIT/{dataset_subsec}_qrecc_format.json"
    output_file = f"processed_datasets/INSCIT/{dataset_subsec}_new_qrecc_format_1.json"
    
    # === Define T5-rewritting file ============
    t5_rewritten_file = f"component3_retriever/input_data/INSCIT/T5QR/{dataset_subsec}_t5_rewrite.json"
    t5_rewritten = {}
    with open(t5_rewritten_file, 'r') as file:
        t5_data = json.load(file)
    for item in t5_data:
        t5_rewritten[item["sample_id"]] = item["t5_rewrite"]
    
    # === Handling collection ==================
    print("Handling collection ...")
    collection_file = "corpus/INSCIT/full_wiki_segments_pyserini_format.jsonl"
    db_path = "corpus/INSCIT/full_wiki_segments.db"

    def load_data_into_db(file_path, db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS records (
                id TEXT PRIMARY KEY,
                contents TEXT
            )
        ''')
        
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                collection_data = json.loads(line.strip())
                cursor.execute('INSERT OR IGNORE INTO records (id, contents) VALUES (?, ?)', 
                            (collection_data['id'], collection_data['contents']))
        
        conn.commit()
        conn.close()
    
    def id2text(db_path, id):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT contents FROM records WHERE id = ?', (id,))
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            return result[0]
        else:
            return "ID not found"
    
    # load_data_into_db(collection_file, db_path)
    
    # === Converting ===========================
    print("Converting ...")
    with open(input_file, 'r') as in_file, open(output_file, "w") as out_file:
        for turn_idx, line in enumerate(in_file):
            
            if turn_idx % 20 == 0:
                print(f"{turn_idx+1} turn(s) are processed ...")
            
            # if turn_idx == 5:
            #     break
            
            turn = json.loads(line.strip())
            sample_id = turn["sample_id"]
            cur_utt_text = turn["cur_utt_text"]
            oracle_utt_text = t5_rewritten[sample_id]
            cur_response_text = turn["cur_response_text"]
            ctx_utts_text = turn["ctx_utts_text"]
            pos_docs_pids = turn["pos_docs_pids"]
            p_texts = [id2text(db_path, key) for key in pos_docs_pids]
            
            # random negatives
            neg_nums = 5
            full_range = set(range(49000000))
            exclude_pids = {int(pid[3:]) for pid in pos_docs_pids}
            remaining_numbers = list(full_range - exclude_pids)
            neg_random_ids = random.sample(remaining_numbers, neg_nums)
            neg_random_pids = [f"doc{str(id)}" for id in neg_random_ids]
            neg_random_text = [id2text(db_path, key) for key in neg_random_pids]
                                
            item = {
                "sample_id": sample_id,
                "cur_utt_text": cur_utt_text,
                "oracle_utt_text": oracle_utt_text,
                "cur_response_text": cur_response_text,
                "ctx_utts_text": ctx_utts_text,
                "pos_docs_pids": pos_docs_pids,
                "pos_docs_text": p_texts,
                "random_neg_docs_pids": neg_random_pids, 
                "random_neg_docs_text": neg_random_text
            }
            
            out_file.write(json.dumps(item) + '\n')
    

if __name__ == "__main__":
    # ===== Step 1) download files ==============
    # inscit_corpus_download_unzip()
    # inscit_convert_corpus_to_tsv()
    # inscit_query_files_download()
    # train_test_file_to_qrecc_format()
    train_to_qrecc_format_for_convgqr()
    
# python component0_preprocessing/inscit_files_preprocessing/1_corpus_preprocessing.py

