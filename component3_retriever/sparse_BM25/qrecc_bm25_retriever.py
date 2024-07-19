#!/usr/bin/env python3


import torch
import random
import numpy as np
import argparse, os, json
from pyserini.search.lucene import LuceneSearcher

print("Available GPUs:", torch.cuda.device_count())
device = 'cuda:0'
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

def bm25_retriever(args):
    
    # === Read queries =======================
    queries = {}
    if args.query_format == "t5_rewritten":
        args.query_file = "component3_retriever/input_data/QReCC/T5QR/t5_rewrite.json"
        with open(args.query_file, 'r') as file:
            data = json.load(file)
        
        for item in data:
            query = item["t5_rewrite"]
            queries[item['sample_id']] = query
        
    elif args.query_format == "ConvGQR_rewritten":
        query_oracle_path = "component3_retriever/input_data/QReCC/ConvGQR/convgqr_rewrite_oracle_prefix.json"
        query_expand_path = "component3_retriever/input_data/QReCC/ConvGQR/convgqr_rewrite_answer_prefix.json"
        
        query_oracle_data = []
        query_expand_data = []
        with open(query_oracle_path, 'r') as file:
            for line in file:
                query_oracle_data.append(json.loads(line.strip()))
        with open(query_expand_path, 'r') as file:
            for line in file:
                query_expand_data.append(json.loads(line.strip()))
        
        for i, oracle_sample in enumerate(query_oracle_data):
            if args.query_type == "raw":
                queries[oracle_sample['id']] = oracle_sample["query"]
            elif args.query_type == "rewrite":
                queries[oracle_sample['id']] = oracle_sample['rewrite']

            elif args.query_type == "decode":
                query = oracle_sample['oracle_utt_text']
                if args.eval_type == "answer":
                    queries[oracle_sample['sample_id']] = query_expand_data[i]['answer_utt_text']
                elif args.eval_type == "oracle+answer":
                    queries[oracle_sample['sample_id']] = query + ' ' + query_expand_data[i]['answer_utt_text']    
    
    else:
        args.query_file = "processed_datasets/QReCC/new_test.json" # test.json for all_history
        with open (args.query_file, 'r') as file:
            for line in file:
                data = json.loads(line.strip())
                
                if args.query_format == "original":
                    query = data["cur_utt_text"]
                elif args.query_format == "human_rewritten":
                    query = data["oracle_utt_text"]
                elif args.query_format == "all_history":
                    query = ' '.join(data["ctx_utts_text"]) + " " + data["cur_utt_text"]
                
                queries[data['sample_id']] = query
    
    # = Select a subset of queries ===========
    if subset_percentage != 1.0:
        subset_size = int(len(queries) * subset_percentage)
        subset_keys = random.sample(list(queries.keys()), subset_size)
        subset_queries = {key: queries[key] for key in subset_keys}
    else:
        subset_queries = queries

    qid_list = list(subset_queries.keys())
    query_list = [subset_queries[qid] for qid in qid_list]
    print(f"Query_id: {qid_list[0]}\nQuery: {query_list[0]}\n") # Print a sample
    
    # === Retriever Model: pyserini search ===
    print("Retrieving using BM25 ...")
    searcher = LuceneSearcher(args.index_dir)
    searcher.set_bm25(args.bm25_k1, args.bm25_b)
    hits = searcher.batch_search(query_list, qid_list, k=args.top_k, threads=20)

    total = 0
    os.makedirs(args.results_base_path, exist_ok=True)
    output_res_file = f"{args.results_base_path}/{args.query_format}_bm25_results.trec"
    with open(output_res_file, "w") as f:
        for qid in qid_list:
            for i, item in enumerate(hits[qid]):
                result_line = f"{qid} Q0 {item.docid[3:]} {i+1} {item.score} bm25"
                f.write(result_line)
                f.write('\n')
                total += 1
    print(total)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", type=str, default="corpus/QReCC/bm25_index")
    # parser.add_argument("--query_file", type=str, default="processed_datasets/QReCC/test.json")
    parser.add_argument("--results_base_path", type=str, default="component3_retriever/output_results/QReCC")
    parser.add_argument("--query_format", type=str, default="ConvGQR_rewritten", choices=['original', 'human_rewritten', 'all_history', 'same_topic', 't5_rewritten', 'ConvGQR_rewritten'])
    
    parser.add_argument("--query_type", type=str, default="decode", help="for ConvGQR")
    parser.add_argument("--eval_type", type=str, default="oracle+answer", help="for ConvGQR")
    
    parser.add_argument("--bm25_k1", type=float, default="0.9")
    parser.add_argument("--bm25_b", type=float, default="0.4")
    parser.add_argument("--top_k", type=int, default="100")
    parser.add_argument("--seed", type=int, default="1")
    
    args = parser.parse_args()
    bm25_retriever(args)
    
    # python component3_retriever/bm25/qrecc_baseline_t5_evaluation.py