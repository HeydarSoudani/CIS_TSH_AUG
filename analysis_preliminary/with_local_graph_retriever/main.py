#!/usr/bin/env python3

import torch
import random
import pytrec_eval
import numpy as np
from tqdm import tqdm
import argparse, logging, json, os
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher
from pyserini.encode import AnceQueryEncoder

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

def local_corpus_pyserini_retriever(args):
    print("Preprocessing files ...")
    index_dir = f"{args.index_dir_base_path}/{args.dataset_name}/{args.retriever_model}_index"

    ### === Prepare queries ==========================================
    queries = {}
    args.query_file = f"component3_retriever/input_data/{args.dataset_name}/baselines/{args.dataset_subsec}/{args.query_format}.jsonl"
    with open (args.query_file, 'r') as file:
        for line in file:
            item = json.loads(line.strip())
            queries[item['id']] = item["query"]
        
    # Select a subset of queries
    if subset_percentage != 1.0:
        subset_size = int(len(queries) * subset_percentage)
        subset_keys = random.sample(list(queries.keys()), subset_size)
        subset_queries = {key: queries[key] for key in subset_keys}
    else:
        subset_queries = queries

    qid_list = list(subset_queries.keys())
    query_list = [subset_queries[qid] for qid in qid_list]
    print(f"Query_id: {qid_list[1]}\nQuery: {query_list[1]}\n")


    ### === Retriever Model: pyserini search =========================
    print(f"Retrieving using {args.retriever_model} ...")
    if args.retriever_model == "bm25":
        searcher = LuceneSearcher(index_dir)
        searcher.set_bm25(args.bm25_k1, args.bm25_b)
        hits = searcher.batch_search(query_list, qid_list, k=args.top_k, threads=20)
    
        # Coding ....
    
    
    elif args.retriever_model == "ance":
        encoder = AnceQueryEncoder(args.query_encoder, device=args.device)
        searcher = FaissSearcher(index_dir, encoder)
        hits = searcher.batch_search(query_list, qid_list, k=args.top_k, threads=20)
    

    ### === Write to output file =====================================
    print("Writing to output file ...")
    os.makedirs(args.results_base_path, exist_ok=True)
    os.makedirs(f"{args.results_base_path}/{args.dataset_name}", exist_ok=True)
    output_res_file = f"{args.results_base_path}/{args.dataset_name}/{args.query_format}_{args.retriever_model}_results.trec"
    
    with open(output_res_file, "w") as f:
        for qid in qid_list:
            for i, item in enumerate(hits[qid]):
                result_line = f"{qid} Q0 {item.docid[3:]} {i+1} {item.score} {args.retriever_model}"
                f.write(result_line)
                f.write('\n')
                
    print("Done!")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever_model", type=str, default="ance", choices=["bm25", "ance"])
    parser.add_argument("--query_encoder", type=str, default="castorini/ance-msmarco-passage")
    parser.add_argument("--index_dir_base_path", type=str, default="corpus")
    parser.add_argument("--results_base_path", type=str, default="analysis_preliminary/corpus_graph_expriments")
    parser.add_argument("--dataset_name", type=str, default="TopiOCQA", choices=["TopiOCQA", "INSCIT"])
    parser.add_argument("--dataset_subsec", type=str, default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--query_format", type=str, default='original', choices=[
        'original', 'human_rewritten', 'all_history', 'same_topic', 't5_rewritten',
    ])
    parser.add_argument("--bm25_k1", type=float, default="0.9")
    parser.add_argument("--bm25_b", type=float, default="0.4")
    parser.add_argument("--top_k", type=int, default="100")
    parser.add_argument("--rel_threshold", type=int, default="1")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    print("Available GPUs:", torch.cuda.device_count())
    args.device = 'cuda:0'
    set_seed(args.seed)
    
    local_corpus_pyserini_retriever(args)
    