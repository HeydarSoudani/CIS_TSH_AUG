import json

def main():
    passage_corpus_file = "corpus/TopiOCQA/full_wiki_segments_pyserini_format.jsonl"
    doc_corpus_file = "analysis_preliminary/corpus_graph_expriments_(docs)/wiki_document_corpus.jsonl"
    mapping_outpur_file = "doc_to_passages.json"
    
    

if __name__ == "__main__":
    main()
    
    
    jsonl_file_path = 'corpus/TopiOCQA/full_wiki_segments_pyserini_format.jsonl'
    with open(jsonl_file_path, 'r') as file:
        file.readline()
        file.readline()
        file.readline()
        file.readline()
        line = file.readline()
    record = json.loads(line)
    print(record)
    
    
    # python analysis_preliminary/corpus_graph_expriments_(docs)/doc_passage_mapping.py
    