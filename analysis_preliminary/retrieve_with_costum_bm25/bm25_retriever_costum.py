
import math
import torch
import random
import numpy as np
from tqdm import tqdm
import argparse, logging, json, os
from pyserini.index import IndexReader

subset_percentage = 1.0

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CustomBM25:
    def __init__(self, index_dir, lc_threshold, k1=1.5, b=0.75):
        self.index_reader = IndexReader(index_dir)
        self.k1 = k1
        self.b = b
        self.lc_threshold = lc_threshold
        self.N = self.index_reader.stats()['documents']
        self.avg_doc_len = self._calculate_avg_doc_len()

    def _calculate_avg_doc_len(self):
        total_length = sum(self.index_reader.get_document_vector(doc_id).size()
                           for doc_id in self.index_reader.doc_ids())
        return total_length / self.N
    
    def _idf(self, term):
        df = self.index_reader.get_term_counts(term, analyzer=None)[0]
        if df == 0:
            return 0
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)
    
    def score(self, query, doc_id, local_corpus):
        doc_vector = self.index_reader.get_document_vector(doc_id)
        doc_len = sum(doc_vector.values())
        
        query_terms = query.split()
        score = 0
        
        for term in query_terms:
            if term in doc_vector:
                term_freq = doc_vector[term]
                idf = self._idf(term)
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                score += idf * (numerator / denominator)
        
        if doc_id[3:] in local_corpus:
            return score * 1.0
        else:
            return score * self.lc_threshold
    
    def search(self, query, local_corpus, top_k=10):
        doc_scores = []
        for doc_id in self.index_reader.doc_ids():
            score = self.score(query, doc_id, local_corpus)
            doc_scores.append((doc_id, score))
        
        scored_results = sorted(doc_scores, key=lambda x: x[1], reverse=True)[:top_k]
        
        return scored_results

def main():
    print("Preprocessing files ...")
    index_dir = f"{args.index_dir_base_path}/{args.dataset_name}/{args.retriever_model}_index"
    
    relevant_psgs_file = "analysis_preliminary/corpus_graph_expriments_docs/doc_to_passages.json"
    
    topic2docid_file = "analysis_preliminary/corpus_graph_expriments_docs/topic_to_docid.json"
    docid2topic_file = "analysis_preliminary/corpus_graph_expriments_docs/docid_to_topic.json"
    
    topic_2_docid = {}
    docid_2_psgs = {}
    with open(relevant_psgs_file, 'r') as file:
        doc_psgs_data = json.load(file)
    for topic, value in doc_psgs_data.items():
        topic_2_docid[topic] = value['doc_id']
        docid_2_psgs[value['doc_id']] = value['passage_ids']
        
    docid_2_similar_docs = {}
    similar_docs_file = "analysis_preliminary/corpus_graph_expriments_docs/original_bm25_results.trec"
    with open(similar_docs_file, 'r') as f:
        qrel_data = f.readlines()
    
    for line in qrel_data:
        line = line.strip().split()
        if line[0] not in docid_2_similar_docs:
            docid_2_similar_docs[line[0]] = []
        docid_2_similar_docs[line[0]].append(line[2])
    
    
    queries = {}
    query_file = "datasets/TopiOCQA/topiocqa_dev.json"
    with open (query_file, 'r') as file:
        queries_data = json.load(file)
    
    for conv_turn in queries_data:
        query_topic = conv_turn['Topic']
        query_id = f"{conv_turn['Conversation_no']}_{conv_turn['Turn_no']}"
        
        docid = topic_2_docid[query_topic]
        similar_docs = docid_2_similar_docs[docid]
        local_corpus = []
        for doc in similar_docs:
            psgs = docid_2_psgs[doc]
            local_corpus.extend(psgs)
            
        queries[query_id] = {"query": conv_turn["Question"], "local_corpus": local_corpus}
        
    # === Select a subset of queries ===========
    if subset_percentage != 1.0:
        subset_size = int(len(queries) * subset_percentage)
        subset_keys = random.sample(list(queries.keys()), subset_size)
        subset_queries = {key: queries[key] for key in subset_keys}
    else:
        subset_queries = queries

    qid_list = list(subset_queries.keys())
    query_list = [subset_queries[qid] for qid in qid_list]
    print(f"Query_id: {qid_list[1]}\nQuery: {query_list[1]}\n")
    
    # === Retriever Model ======================
    local_corpus = ['doc1', 'doc3', 'doc7']
    costum_bm25_model = CustomBM25(index_dir, args.lc_threshold)

    hits = {}
    for qid, item in queries.items():
        q_hits = costum_bm25_model.search(item["query"], item["local_corpus"], top_k=args.top_k)
        hits[qid] = q_hits

    # === Write to output file ===============
    print("Writing to output file ...")
    with open(args.output_res_file, "w") as f:
        for qid in qid_list:
            for i, item in enumerate(hits[qid]):
                result_line = f"{qid} Q0 {item.docid[3:]} {i+1} {item.score} {args.retriever_model}"
                f.write(result_line)
                f.write('\n')    
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bm25_k1", type=float, default="0.9")
    parser.add_argument("--bm25_b", type=float, default="0.4")
    parser.add_argument("--top_k", type=int, default="100")
    parser.add_argument("--lc_threshold", type=float, default="0.8")
    parser.add_argument("--rel_threshold", type=int, default="1")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    print("Available GPUs:", torch.cuda.device_count())
    args.device = 'cuda:0'
    set_seed(args.seed)
    
    main()