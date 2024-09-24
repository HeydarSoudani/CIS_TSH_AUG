import json
import re
from tqdm import tqdm

def main():
    passage_corpus_file = "corpus/TopiOCQA/full_wiki_segments_pyserini_format.jsonl"
    doc_corpus_file = "analysis_preliminary/corpus_graph_expriments_docs/wiki_document_corpus.jsonl"
    mapping_outpur_file = "analysis_preliminary/corpus_graph_expriments_docs/doc_to_passages.json"
    
    output  = {}
    with open(doc_corpus_file, 'r') as in_file:
        for idx, line in tqdm(enumerate(in_file), desc="Reading topics from doc corpus ..."):
            
            # if idx == 10000:
            #     break
            
            item = json.loads(line)
            output[item['topic']] = {"doc_id": item['id'], "passage_ids": []}
    
    topics = list(output.keys())
    pattern = re.compile('|'.join(re.escape(topic) for topic in topics))
      
    with open(passage_corpus_file, 'r') as in_file:
        for idx, line in tqdm(enumerate(in_file), desc="Reading passages ..."):
            
            # if idx == 10000:
            #     break
            
            item = json.loads(line)
            first_part = item['contents'].split(',')[0]
            match = pattern.search(first_part)
            if match:
                topic = match.group(0)
                output[topic]['passage_ids'].append(item['id'])
      
    # with open(passage_corpus_file, 'r') as in_file:
    #     for idx, line in tqdm(enumerate(in_file), desc="Reading passages ..."):
            
    #         # if idx == 100:
    #         #     break
            
    #         item = json.loads(line)
    #         first_part = item['contents'].split(',')[0]
            
    #         for topic, value in output.items():
    #             if topic in first_part:
    #                 value['passage_ids'].append(item['id'])
    #                 break
            
    with open(mapping_outpur_file, 'w') as json_file:
        json.dump(output, json_file, indent=4)
    
    

if __name__ == "__main__":
    main()
    
    # passage_corpus_file = "corpus/TopiOCQA/full_wiki_segments_pyserini_format.jsonl"
    # # jsonl_file_path = "analysis_preliminary/corpus_graph_expriments_docs/wiki_document_corpus.jsonl"
    # with open(passage_corpus_file, 'r') as file:
    #     # file.readline()
    #     # file.readline()
    #     # file.readline()
    #     # file.readline()
    #     line = file.readline()
    # record = json.loads(line)
    # print(record)
    
    
    # python analysis_preliminary/corpus_graph_expriments_(docs)/doc_passage_mapping.py
    