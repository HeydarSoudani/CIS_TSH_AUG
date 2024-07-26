
import torch
import random
import numpy as np
import argparse, os
from pyserini.search.lucene import LuceneSearcher

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def single_query_bm25_retriever(args):
    
    print("Preprocessing files ...")
    index_dir = f"{args.index_dir_base_path}/{args.dataset_name}/bm25_index"
    
    query_list = [
        "Topic: CBS [SEP]  what is the full form of the channel you just mentioned?"
    ]
    
    qid_list = [
        "6_4"   
    ]
    
    # === Retriever Model: pyserini search ===
    print("Retrieving using BM25 ...")
    searcher = LuceneSearcher(index_dir)
    searcher.set_bm25(args.bm25_k1, args.bm25_b)
    hits = searcher.batch_search(query_list, qid_list, k=args.top_k, threads=20)
    
    os.makedirs(args.results_base_path, exist_ok=True)
    os.makedirs(f"{args.results_base_path}/{args.dataset_name}", exist_ok=True)
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
    parser.add_argument("--dataset_name", type=str, default="TopiOCQA", choices=["TopiOCQA", "INSCIT"])
    parser.add_argument("--dataset_subsec", type=str, default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--query_format", type=str, default="single_query")
    
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
    
    single_query_bm25_retriever(args)
    
    # python component3_retriever/sparse_BM25/single_query_bm25_retriever.py
    
    