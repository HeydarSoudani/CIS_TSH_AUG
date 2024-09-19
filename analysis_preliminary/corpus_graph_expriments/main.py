#!/usr/bin/env python3

import torch
import random
import numpy as np
import argparse, logging, json, os
from pyserini.search.lucene import LuceneSearcher
# from pyserini.search.faiss import FaissSearcher
# from pyserini.dsearch import AnceQueryEncoder

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

def processed_row_file(row_file, output_file):
    with open(row_file, 'r') as json_file:
        data = json.load(json_file)
    
    with open(output_file, 'w') as jsonl_file:
        for turn in data:
            for pos_ctx in turn["positive_ctxs"]:
                item = {
                    "conv_num": turn["conv_id"],
                    "turn_num": turn["turn_id"],
                    "id": pos_ctx["passage_id"],
                    "title": pos_ctx["title"],
                    "query": pos_ctx["text"]
                }
                jsonl_file.write(json.dumps(item) + '\n')
        
def similar_docs_pyserini(args):
    print("Preprocessing files ...")
    index_dir = f"{args.index_dir_base_path}/{args.dataset_name}/{args.retriever_model}_index"
    
    queries = {}
    with open (args.doc_as_query, 'r') as file:
        for line in file:
            item = json.loads(line.strip())
            queries[item['id']] = item["query"]
        
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

    # === Retriever Model: pyserini search ===
    print(f"Retrieving using {args.retriever_model} ...")
    if args.retriever_model == "bm25":
        searcher = LuceneSearcher(index_dir)
        searcher.set_bm25(args.bm25_k1, args.bm25_b)
        hits = searcher.batch_search(query_list, qid_list, k=args.top_k, threads=20)
    # elif args.retriever_model == "ance":
    #     encoder = AnceQueryEncoder(args.query_encoder, device=args.device)
    #     searcher = FaissSearcher(index_dir, encoder)
    #     hits = searcher.batch_search(query_list, qid_list, k=args.top_k, threads=20)

    
    # === Write to output file ===============
    print("Writing to output file ...")
    os.makedirs(args.results_base_path, exist_ok=True)
    args.output_res_file = f"{args.results_base_path}/similar_doc_{args.dataset_name}_{args.dataset_subsec}_{args.retriever_model}_results.trec"
    
    with open(args.output_res_file, "w") as f:
        for qid in qid_list:
            for i, item in enumerate(hits[qid]):
                result_line = f"{qid} Q0 {item.docid[3:]} {i+1} {item.score} {args.retriever_model}"
                f.write(result_line)
                f.write('\n')    
    print("Done!")

def postprocessing_results(args):
    
    def find_rank_and_score(args, current_passage, next_passage):
        rank, score = None, None
        with open(args.output_res_file, 'r') as file:
            for line in file:
                parts = line.strip().split()
                cur_passage_id = parts[0]
                similar_passage_id = parts[2]
                rank_value = int(parts[3])
                score_value = float(parts[4])
        
                if cur_passage_id == current_passage:
                    if similar_passage_id == next_passage:
                        rank, score = rank_value, score_value
                        break
        
        if rank is not None and score is not None:
            return rank, score
        else:
            return None, None
    
    all_turns = []
    with open(args.doc_as_query, 'r') as in_file:
        for line in in_file:
            turn_obj = json.loads(line)
            all_turns.append(turn_obj)
    
    connections = {}
    for idx, turn in enumerate(all_turns):
        
        # Current turn
        cur_conv = turn["conv_num"]
        
        if idx < len(all_turns)-1:
            nxt_conv = all_turns[idx+1]["conv_num"]
        
            if cur_conv == nxt_conv:
                if cur_conv not in connections:
                    connections[cur_conv] = []
                
                cur_turn = turn["id"]
                nxt_turn = all_turns[idx+1]["id"]
                rank, score = find_rank_and_score(args, cur_turn, nxt_turn)
                connections[cur_conv].append((cur_turn, rank, score, nxt_turn))
            
            else:
                connections[cur_conv].append((cur_turn, None, None, None))
        
        else:
            connections[cur_conv].append((cur_turn, None, None, None))
       
    connections_output_file = f"{args.results_base_path}/connections_{args.dataset_name}_{args.dataset_subsec}_{args.retriever_model}_results.jsonl"     
    with open(connections_output_file, 'w') as out_file:
        for conv_num, value in connections.items():
            item = {conv_num: value} 
            out_file.write(json.dumps(item) + '\n')
            

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever_model", type=str, default="bm25", choices=["bm25", "ance"])
    parser.add_argument("--query_encoder", type=str, default="castorini/ance-msmarco-passage")
    parser.add_argument("--index_dir_base_path", type=str, default="corpus")
    parser.add_argument("--results_base_path", type=str, default="analysis_preliminary/corpus_graph_expriments")
    parser.add_argument("--dataset_name", type=str, default="TopiOCQA", choices=["TopiOCQA", "INSCIT"])
    parser.add_argument("--dataset_subsec", type=str, default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--bm25_k1", type=float, default="0.9")
    parser.add_argument("--bm25_b", type=float, default="0.4")
    parser.add_argument("--top_k", type=int, default="1000")
    parser.add_argument("--rel_threshold", type=int, default="1")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    print("Available GPUs:", torch.cuda.device_count())
    args.device = 'cuda:0'
    set_seed(args.seed)
    
    
    # === 1) file preprocessing ==============
    row_file = f"processed_datasets/TopiOCQA/ir_all_history_{args.dataset_subsec}.json"
    args.doc_as_query = f"analysis_preliminary/corpus_graph_expriments/doc_as_query_{args.dataset_subsec}.jsonl"
    # processed_row_file(row_file, args.doc_as_query)
    
    
    # === 2) file preprocessing ==============
    similar_docs_pyserini(args)
    # postprocessing_results(args)
    
    # python analysis_preliminary/corpus_graph_expriments/main.py
    
    