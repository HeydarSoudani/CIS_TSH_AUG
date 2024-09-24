#!/usr/bin/env python3

import json
import re
from tqdm import tqdm

def main():
    
    # ==== Get dev topic list ======================
    query_file = "datasets/TopiOCQA/topiocqa_dev.json"
    with open (query_file, 'r') as file:
        queries_data = json.load(file)
    
    topics = []
    dev_topics = []
    all_needed_docids = []
    for conv_turn in tqdm(queries_data, desc="Reading dev topics ..."):
        topic = conv_turn['Topic']
        if topic not in dev_topics:
            dev_topics.append(conv_turn['Topic'])
    
    # ==== Get topic to document id mapping =========
    topic2docid  = {}
    docid2topic  = {}
    doc_corpus_file = "analysis_preliminary/corpus_graph_expriments_docs/wiki_document_corpus.jsonl"
    with open(doc_corpus_file, 'r') as in_file:
        for idx, line in tqdm(enumerate(in_file), desc="Reading topics from doc corpus ..."):
            # if idx == 10000:
            #     break
            item = json.loads(line)
            topic2docid[item['topic']] = item['id']
            docid2topic[item['id']] = item['topic']
            
            if item['topic'] in dev_topics and item['topic'] not in topics:
                topics.append(item['topic'])
            
    topic2docid_output_file = "analysis_preliminary/corpus_graph_expriments_docs/topic_to_docid.json"
    docid2topic_output_file = "analysis_preliminary/corpus_graph_expriments_docs/docid_to_topic.json"
    with open(topic2docid_output_file, 'w') as json_file:
        json.dump(topic2docid, json_file, indent=4)
    with open(docid2topic_output_file, 'w') as json_file:
        json.dump(docid2topic, json_file, indent=4)
    
    
    # === Get document to passage mapping ============    
    similar_docs_file = "analysis_preliminary/corpus_graph_expriments_docs/original_bm25_results.trec"
    with open(similar_docs_file, 'r') as f:
        qrel_data = f.readlines()
    
    for line in qrel_data:
        line = line.strip().split()
        if line[2] not in all_needed_docids:
            topic = docid2topic[f"doc{line[2]}"]
            if topic not in topics:
                topics.append(topic)
    
    
    print("All topics to be processed: ", len(topics))
    pattern = re.compile('|'.join(re.escape(topic) for topic in topics))
      
    docid2psgids = {}
    passage_corpus_file = "corpus/TopiOCQA/full_wiki_segments_pyserini_format.jsonl"
    with open(passage_corpus_file, 'r') as in_file:
        for idx, line in tqdm(enumerate(in_file), desc="Reading passages ..."):
            # if idx == 10000:
            #     break
            item = json.loads(line)
            first_part = item['contents'].split(',')[0]
            match = pattern.search(first_part)
            if match:
                topic = match.group(0)
                docid = topic2docid[topic]
                if docid not in docid2psgids:
                    docid2psgids[docid] = []
                docid2psgids[docid].append(item['id'])

    doc_to_passages_output_file = "analysis_preliminary/corpus_graph_expriments_docs/doc_to_passages.json"
    with open(doc_to_passages_output_file, 'w') as json_file:
        json.dump(docid2psgids, json_file, indent=4)
    
    

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
    