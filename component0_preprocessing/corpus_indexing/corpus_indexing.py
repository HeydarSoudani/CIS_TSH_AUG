### ==============================================================================
### Ref: https://github.com/fengranMark/HAConvDR
### ==============================================================================

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
    # convert_to_pyserini_file(args)
    
    
    # file_path = "corpus/INSCIT/full_wiki_segments_pyserini_format.jsonl"
    input_file = "corpus/QReCC/full_collection_segments_pyserini_format.jsonl"
    output_file = "corpus/QReCC/full_collection_segments_pyserini_format_1.jsonl"
    
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        
        for idx, line in tqdm(enumerate(infile)):
            try:
                # Parse the line as a JSON object
                json_obj = json.loads(line.strip())
                
                # Check if the line should be removed based on the condition function
                if idx != 22346:
                    outfile.write(json.dumps(json_obj) + '\n')
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")
    
    
    # file_path = "corpus/TopiOCQA/full_wiki_segments_pyserini_format.jsonl"
    # with open(file_path, 'r') as f:
    #     first_line = f.readline()
    # print(first_line)
    
    # with open(file_path, 'r') as f:
    #     for i, line in tqdm(enumerate(f)):
    #         try:
    #             # Parse and process each line
    #             json_data = json.loads(line)
                
    #             if i > 22340 and i <  22350:
    #                 print(f"{i}:")
    #                 print(json_data)
    #                 # break
                
    #             if i == 22350:
    #                 break
                
    #             # Your processing code here
    #             # E.g., accessing a list element in json_data
    #             # print(json_data['some_key'])
                
    #         except IndexError as e:
    #             print(f"Error on line {i}: {e}")
    #             print(f"Problematic line content: {line}")
    #             break  # Stop the loop if you want to debug just this issue
    #         except Exception as e:
    #             print(f"Other error on line {i}: {e}")
    
    
    # = Step 2-1) BM25 corpus indexing using pyserini
    # python -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator -threads 20 -input "corpus/QReCC" -index "corpus/QReCC/bm25_index" -storePositions -storeDocvectors -storeRaw
        
    # = Step 2-2) ANCE corpus indexing using pyserini
    # = Src: https://github.com/castorini/pyserini/blob/master/docs/usage-index.md#building-a-dense-vector-index
    
    # python -m pyserini.encode \
    #   input   --corpus corpus/QReCC/full_collection_segments_pyserini_format.jsonl \
    #           --fields text \
    #           --delimiter "\n" \
    #           --shard-id 0 \
    #           --shard-num 1 \
    #   output  --embeddings corpus/QReCC/ance_index \
    #           --to-faiss \
    #   encoder --encoder-class "ance" \
    #           --encoder castorini/ance-msmarco-passage \
    #           --fields text \
    #           --batch 64 \
    #           --fp16
    
    # For merging:
    # python -m pyserini.index.merge_faiss_indexes --prefix indexes/dindex-sample-dpr-multi- --shard-num 4
    
    
# python component0_preprocessing/corpus_indexing/corpus_indexing.py
