

import torch
import random
import pytrec_eval
import numpy as np
from tqdm import tqdm, trange
import argparse, logging, os, json
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

def bm25_evaluation(args):
    print("Evaluating ...")
    # === Read results and gold_qrel files ===========
    results_file = f"{args.results_base_path}/{args.query_format}_bm25_results.trec"
    with open(results_file, 'r') as f:
        run_data = f.readlines()
    
    gold_qrel_file = "processed_datasets/QReCC/qrecc_qrel.tsv"
    with open(gold_qrel_file, 'r') as f:
        qrel_data = f.readlines()
    
    qrels = {}
    qrels_ndcg = {}
    runs = {}
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
    
    for line in run_data:
        line = line.split(" ")
        query = line[0]
        passage = line[2]
        rel = int(float(line[4]))
        if query not in runs:
            runs[query] = {}
        runs[query][passage] = rel

    # pytrec_eval eval
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "recip_rank", "recall.5", "recall.10", "recall.20", "recall.100"})
    res = evaluator.evaluate(runs)
    map_list = [v['map'] for v in res.values()]
    mrr_list = [v['recip_rank'] for v in res.values()]
    recall_100_list = [v['recall_100'] for v in res.values()]
    recall_20_list = [v['recall_20'] for v in res.values()]
    recall_10_list = [v['recall_10'] for v in res.values()]
    recall_5_list = [v['recall_5'] for v in res.values()]

    evaluator = pytrec_eval.RelevanceEvaluator(qrels_ndcg, {"ndcg_cut.3"})
    res = evaluator.evaluate(runs)
    ndcg_3_list = [v['ndcg_cut_3'] for v in res.values()]

    # context_affect(query_id, mrr_list)

    res = {
            "MAP": np.average(map_list),
            "MRR": np.average(mrr_list),
            "NDCG@3": np.average(ndcg_3_list),
            "Recall@5": np.average(recall_5_list),
            "Recall@10": np.average(recall_10_list),
            "Recall@20": np.average(recall_20_list),
            "Recall@100": np.average(recall_100_list), 
        }
    print("---------------------Evaluation results:---------------------")    
    print(res)

def bm25_evaluation_per_turn(args):
    pass

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", type=str, default="corpus/QReCC/bm25_index")
    parser.add_argument("--query_file", type=str, default="processed_datasets/QReCC/new_test.json")
    parser.add_argument("--results_base_path", type=str, default="component3_retriever/output_results/QReCC")
    parser.add_argument("--query_format", type=str, default="original", choices=['original', 'human_rewritten', 'all_history', 'same_topic'])
    parser.add_argument("--bm25_k1", type=float, default="0.9")
    parser.add_argument("--bm25_b", type=float, default="0.4")
    parser.add_argument("--top_k", type=int, default="100")
    parser.add_argument("--rel_threshold", type=int, default="1")
    parser.add_argument("--seed", type=int, default="1")
    
    args = parser.parse_args()
    
    bm25_retriever(args)
    # bm25_evaluation(args)
    
    # python component3_retriever/bm25/qrecc_baseline_evaluation.py