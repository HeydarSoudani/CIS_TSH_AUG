#!/usr/bin/env python3

import pytrec_eval
import torch
import random
import numpy as np
import argparse, logging, json, os

from tqdm import tqdm

from distutils.command.build_scripts import first_line_re
# from pyserini.search.lucene import LuceneSearcher
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

def create_gold_trec_file(args):
    gold_trec_file = f"analysis_preliminary/corpus_graph_expriments/gold_trec_{args.dataset_name}_{args.dataset_subsec}.trec"
    
    all_turns = []
    with open(args.doc_as_query, 'r') as in_file:
        for line in in_file:
            turn_obj = json.loads(line)
            all_turns.append(turn_obj)

    with open(gold_trec_file, 'w') as out_file:
        for idx, turn in tqdm(enumerate(all_turns)):
            cur_conv = turn["conv_num"]
            if idx < len(all_turns)-1:
                nxt_conv = all_turns[idx+1]["conv_num"]
            
                if cur_conv == nxt_conv:
                    cur_turn = turn["id"]
                    nxt_turn = all_turns[idx+1]["id"]
                    out_file.write(f"{cur_turn} Q0 {nxt_turn} {args.retriever_model} 1\n")

def create_gold_trec_files_per_type(args):
    first_gold_trec_file = f"analysis_preliminary/corpus_graph_expriments/gold_trec_first_{args.dataset_name}_{args.dataset_subsec}.trec"
    concentrated_gold_trec_file = f"analysis_preliminary/corpus_graph_expriments/gold_trec_concentrated_{args.dataset_name}_{args.dataset_subsec}.trec"
    shifted_gold_trec_file = f"analysis_preliminary/corpus_graph_expriments/gold_trec_shifted_{args.dataset_name}_{args.dataset_subsec}.trec"
    
    turns_buckets = "processed_datasets/TopiOCQA/turn_buckets/per_shift.json"
    with open(turns_buckets, 'r', encoding='utf-8') as file:
        bucket_data = json.load(file)
    
    
    all_turns = []
    with open(args.doc_as_query, 'r') as in_file:
        for line in in_file:
            turn_obj = json.loads(line)
            all_turns.append(turn_obj)

    with open(first_gold_trec_file, 'w') as first_out_file, open(concentrated_gold_trec_file, 'w') as conc_out_file, open(shifted_gold_trec_file, 'w') as shift_out_file:
        for idx, turn in tqdm(enumerate(all_turns)):
            cur_conv = turn["conv_num"]
            cur_turn_num = turn["turn_num"]
            if idx < len(all_turns)-1:
                nxt_conv = all_turns[idx+1]["conv_num"]
                
                if cur_conv == nxt_conv:
                    cur_turn = turn["id"]
                    nxt_turn = all_turns[idx+1]["id"]
                    
                    query_id = f"{str(cur_conv)}_{str(cur_turn_num)}"
                    print(query_id)
                    if query_id in bucket_data["First"]:
                        first_out_file.write(f"{cur_turn} Q0 {nxt_turn} 1\n")
                    elif query_id in bucket_data["Topic-concentrated"]:
                        conc_out_file.write(f"{cur_turn} Q0 {nxt_turn} 1\n")
                    elif query_id in bucket_data["Topic-shift"]:
                        shift_out_file.write(f"{cur_turn} Q0 {nxt_turn} 1\n")

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
    for idx, turn in tqdm(enumerate(all_turns)):
        
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

def retriever_eval(args):
    first_gold_trec_file = f"analysis_preliminary/corpus_graph_expriments/gold_trec_first_{args.dataset_name}_{args.dataset_subsec}.trec"
    concentrated_gold_trec_file = f"analysis_preliminary/corpus_graph_expriments/gold_trec_concentrated_{args.dataset_name}_{args.dataset_subsec}.trec"
    shifted_gold_trec_file = f"analysis_preliminary/corpus_graph_expriments/gold_trec_shifted_{args.dataset_name}_{args.dataset_subsec}.trec"
    with open(shifted_gold_trec_file, 'r') as f:
        qrel_data = f.readlines()
    
    qrels = {}
    qrels_ndcg = {}
    query_id = []
    for line in qrel_data:
        line = line.strip().split()
        query = line[0]
        passage = line[2]
        rel = int(line[3])
        if query not in qrels:
            qrels[query] = {}
            query_id.append(query)
        if query not in qrels_ndcg:
            qrels_ndcg[query] = {}

        # for NDCG
        qrels_ndcg[query][passage] = rel
        # for MAP, MRR, Recall
        if rel >= args.rel_threshold:
            rel = 1
        else:
            rel = 0
        qrels[query][passage] = rel
    
    
    runs = {}
    with open(args.output_res_file, 'r') as f:
        run_data = f.readlines()
    for line in run_data:
        line = line.split(" ")
        query = line[0]
        passage = line[2]
        rel = int(float(line[4]))
        if query not in runs:
            runs[query] = {}
        runs[query][passage] = rel

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "recip_rank", "recall.5", "recall.10", "recall.100", "recall.1000"})
    res = evaluator.evaluate(runs)
    map_list = [v['map'] for v in res.values()]
    mrr_list = [v['recip_rank'] for v in res.values()]
    recall_1000_list = [v['recall_1000'] for v in res.values()]
    recall_100_list = [v['recall_100'] for v in res.values()]
    recall_10_list = [v['recall_10'] for v in res.values()]
    recall_5_list = [v['recall_5'] for v in res.values()]

    evaluator = pytrec_eval.RelevanceEvaluator(qrels_ndcg, {"ndcg_cut.3"})
    res = evaluator.evaluate(runs)
    ndcg_3_list = [v['ndcg_cut_3'] for v in res.values()]
    res = {
            "MAP": np.average(map_list),
            "MRR": np.average(mrr_list),
            "NDCG@3": np.average(ndcg_3_list),
            "Recall@5": np.average(recall_5_list),
            "Recall@10": np.average(recall_10_list),
            "Recall@100": np.average(recall_100_list), 
            "Recall@1000": np.average(recall_1000_list),
            
        }
    print("---------------------Evaluation results:---------------------")    
    print(res)


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
    
    # create_gold_trec_file(args)
    # create_gold_trec_files_per_type(args)
    
    
    # === 2) file preprocessing ==============
    os.makedirs(args.results_base_path, exist_ok=True)
    args.output_res_file = f"{args.results_base_path}/similar_doc_{args.dataset_name}_{args.dataset_subsec}_{args.retriever_model}_results.trec"
    
    # similar_docs_pyserini(args)
    # postprocessing_results(args)
    
    retriever_eval(args)
    
    # python analysis_preliminary/corpus_graph_expriments/main.py
    
    