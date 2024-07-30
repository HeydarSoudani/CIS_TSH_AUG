#!/usr/bin/env python3

### ==============================================================================
# Ref: https://github.com/fengranMark/HAConvDR/blob/main/bm25/bm25_topiocqa.py
### ==============================================================================

import torch
import random
import numpy as np
import argparse, logging, os, json
from pyserini.search.lucene import LuceneSearcher

logging.basicConfig(level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    handlers=[logging.StreamHandler()]
)
os.environ["WANDB_MODE"] = "offline"

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
    
    print("Preprocessing files ...")
    index_dir = f"{args.index_dir_base_path}/{args.dataset_name}/bm25_index"
    
    # === Read query file ====================
    if args.add_gold_topic:
        topics = {}
        topic_file = f"component3_retriever/input_data/{args.dataset_name}/baselines/{args.dataset_subsec}/original.jsonl"
        with open (topic_file, 'r') as file:
            for line in file:
                data = json.loads(line.strip())
                if args.dataset_name == "TopiOCQA":
                    topics[data['id']] = data["title"].split('[SEP]')[0]
                elif args.dataset_name == "INSCIT":
                    topics[data['id']] = data["topic"]
    
    queries = {}
    if args.query_format == "t5_rewritten":
        args.query_file = f"component3_retriever/input_data/{args.dataset_name}/T5QR/t5_rewrite.json"
        with open(args.query_file, 'r') as file:
            data = json.load(file)
        
        for item in data:
            query = item["t5_rewrite"]
            if args.add_gold_topic:
                queries[item['sample_id']] = topics[item['sample_id']] + ' [SEP] ' + query
            else:
                queries[item['sample_id']] = query
    
    elif args.query_format == "ConvGQR_rewritten":
        query_oracle_path = "component3_retriever/input_data/INSCIT/ConvGQR/convgqr_rewrite_oracle_prefix.json"
        query_expand_path = "component3_retriever/input_data/INSCIT/ConvGQR/convgqr_rewrite_answer_prefix.json"
        
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
        args.query_file = f"component3_retriever/input_data/{args.dataset_name}/baselines/{args.dataset_subsec}/{args.query_format}.jsonl"
        with open (args.query_file, 'r') as file:
            for line in file:
                item = json.loads(line.strip())
                
                if args.add_gold_topic:
                    queries[item['id']] = topics[item['id']] + ' [SEP] ' + item["query"]
                else:
                    queries[item['id']] = item["query"]
    
    
    # = Select a subset of queries ===========
    if subset_percentage != 1.0:
        subset_size = int(len(queries) * subset_percentage)
        subset_keys = random.sample(list(queries.keys()), subset_size)
        subset_queries = {key: queries[key] for key in subset_keys}
    else:
        subset_queries = queries

    qid_list = list(subset_queries.keys())
    query_list = [subset_queries[qid] for qid in qid_list]
    print(f"Query_id: {qid_list[1]}\nQuery: {query_list[1]}\n")
    
    
    # === Retriever Model: pyserini search ===
    print("Retrieving using BM25 ...")
    searcher = LuceneSearcher(index_dir)
    searcher.set_bm25(args.bm25_k1, args.bm25_b)
    hits = searcher.batch_search(query_list, qid_list, k=args.top_k, threads=20)

    os.makedirs(args.results_base_path, exist_ok=True)
    os.makedirs(f"{args.results_base_path}/{args.dataset_name}", exist_ok=True)
    
    if args.add_gold_topic:
        output_res_file = f"{args.results_base_path}/{args.dataset_name}/topic+{args.query_format}_bm25_results.trec"
    else:
        output_res_file = f"{args.results_base_path}/{args.dataset_name}/{args.query_format}_bm25_results.trec"
    
    with open(output_res_file, "w") as f:
        for qid in qid_list:
            for i, item in enumerate(hits[qid]):
                result_line = f"{qid} Q0 {item.docid[3:]} {i+1} {item.score} bm25"
                f.write(result_line)
                f.write('\n')
                
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir_base_path", type=str, default="corpus")
    parser.add_argument("--results_base_path", type=str, default="component3_retriever/output_results")
    parser.add_argument("--dataset_name", type=str, default="INSCIT", choices=["TopiOCQA", "INSCIT"])
    parser.add_argument("--dataset_subsec", type=str, default="test", choices=["train", "dev", "test"])
    parser.add_argument("--query_format", type=str, default='same_topic', choices=[
        'original', 'human_rewritten', 'all_history', 'same_topic', 't5_rewritten', 'ConvGQR_rewritten',
    ])
    parser.add_argument("--add_gold_topic", action="store_true")
    parser.add_argument("--query_type", type=str, default="decode", help="for ConvGQR")
    parser.add_argument("--eval_type", type=str, default="oracle+answer", help="for ConvGQR")
    
    parser.add_argument("--bm25_k1", type=float, default="0.9")
    parser.add_argument("--bm25_b", type=float, default="0.4")
    parser.add_argument("--top_k", type=int, default="100")
    parser.add_argument("--rel_threshold", type=int, default="1")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # print("Available GPUs:", torch.cuda.device_count())
    # device = 'cuda:0'
    set_seed(args.seed)
    
    bm25_retriever(args)
    
    
    # python component3_retriever/sparse_BM25/topiocqa_inscit_bm25_retriever.py
    