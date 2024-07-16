### ==============================================================================
# Ref: https://github.com/fengranMark/HAConvDR/blob/main/bm25/bm25_topiocqa.py
### ==============================================================================

import torch
import random
import pytrec_eval
import numpy as np
from tqdm import tqdm, trange
import argparse, logging, os, json
from pyserini.search.lucene import LuceneSearcher


logging.basicConfig(level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    handlers=[logging.StreamHandler()]
)
os.environ["WANDB_MODE"] = "offline"
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
    
    print("Preprocessing files ...")
    output_res_file = f"{args.result_qrel_base_path}/{args.dataset_name}/{args.dataset_subsec}/bm25_{args.query_format}_results.trec"
    os.makedirs(args.result_qrel_base_path, exist_ok=True)
    os.makedirs(f"{args.result_qrel_base_path}/{args.dataset_name}", exist_ok=True)
    os.makedirs(f"{args.result_qrel_base_path}/{args.dataset_name}/{args.dataset_subsec}", exist_ok=True)
    
    index_dir = f"{args.index_dir_base_path}/{args.dataset_name}/bm25_index"
    
    # === Read query file ====================
    query_path = f"component3_retriever/data/{args.dataset_name}/{args.dataset_subsec}/{args.query_format}.jsonl"
    queries = {}
    with open (query_path, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            queries[data['id']] = data

    # = Select a subset of queries ===========
    if subset_percentage != 1.0:
        subset_size = int(len(queries) * subset_percentage)
        subset_keys = random.sample(list(queries.keys()), subset_size)
        subset_queries = {key: queries[key] for key in subset_keys}
    else:
        subset_queries = queries

    qid_list = list(subset_queries.keys())
    query_list = [subset_queries[qid]['query'] for qid in qid_list]
    
    
    # === Retriever Model: pyserini search ===
    print("Retrieving using BM25 ...")
    searcher = LuceneSearcher(index_dir)
    searcher.set_bm25(args.bm25_k1, args.bm25_b)
    hits = searcher.batch_search(query_list, qid_list, k=args.top_k, threads=20)

    total = 0
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
    input_file = f"{args.result_qrel_base_path}/{args.dataset_name}/{args.dataset_subsec}/bm25_{args.query_format}_results.trec"
    with open(input_file, 'r' )as f:
        run_data = f.readlines()
    
    gold_qrel_file = f"component3_retriever/data/{args.dataset_name}/{args.dataset_subsec}/qrel_gold.trec"
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
    
def evaluation_per_turn(args):
    
    with open(args.gold_qrel_path, 'r') as f:
        qrel_gold_data = f.readlines()
    with open(f"{args.result_qrel_path}/{args.dataset_name}_dev_bm25_res_{args.query_format}.trec", 'r' )as f:
        qrel_result_data = f.readlines()
        
    # == Prepare gold qrels ========
    qrels = {}
    qrels_ndcg = {}
    query_id = []
    for line in qrel_gold_data:
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
    
    
    # == Prepare result runs ========
    runs = {}
    for line in qrel_result_data:
        line = line.split(" ")
        
        turn = int(line[0].split("_")[1])
        
        if f"turn_{turn}" not in runs:
            runs[f"turn_{turn}"] = {}
        
        query = line[0]
        passage = line[2]
        rel = int(float(line[4]))
        if query not in runs[f"turn_{turn}"]:
            runs[f"turn_{turn}"][query] = {}
        runs[f"turn_{turn}"][query][passage] = rel
    
    # print(runs.keys())
    sorted_runs = {k: runs[k] for k in sorted(runs, key=lambda x: int(x.split('_')[1]))}

    results_per_turns = {
        "MAP": [],
        "MRR": [],
        "NDCG@3": [],
        "Recall@5": [],
        "Recall@10": [],
        "Recall@20": [],
        "Recall@100": [],
    }
    for turn, runs in sorted_runs.items():
        print(f"Turn: {turn}")
        
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
        
        print("---------------------Evaluation results:---------------------")
        res = {
            "MAP": np.average(map_list),
            "MRR": np.average(mrr_list),
            "NDCG@3": np.average(ndcg_3_list),
            "Recall@5": np.average(recall_5_list),
            "Recall@10": np.average(recall_10_list),
            "Recall@20": np.average(recall_20_list),
            "Recall@100": np.average(recall_100_list), 
        }  
        print(res)
        
        results_per_turns["MAP"].append(np.average(map_list))
        results_per_turns["MRR"].append(np.average(mrr_list))
        results_per_turns["NDCG@3"].append(np.average(ndcg_3_list))
        results_per_turns["Recall@5"].append(np.average(recall_5_list))
        results_per_turns["Recall@10"].append(np.average(recall_10_list))
        results_per_turns["Recall@20"].append(np.average(recall_20_list))
        results_per_turns["Recall@100"].append(np.average(recall_100_list))
    print("---------------------Evaluation results (per turn):----------")
    print(results_per_turns)
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir_base_path", type=str, default="corpus")
    parser.add_argument("--result_qrel_base_path", type=str, default="component3_retriever/results")
    parser.add_argument("--dataset_name", type=str, default="INSCIT", choices=["TopiOCQA", "INSCIT", "qrecc"])
    parser.add_argument("--dataset_subsec", type=str, default="test", choices=["train", "dev", "test"])
    parser.add_argument("--query_format", type=str, default="same_topic", choices=['original', 'human_rewritten', 'all_history', 'same_topic'])
    parser.add_argument("--bm25_k1", type=float, default="0.9")
    parser.add_argument("--bm25_b", type=float, default="0.4")
    parser.add_argument("--top_k", type=int, default="100")
    parser.add_argument("--rel_threshold", type=int, default="1")
    parser.add_argument("--seed", type=int)
    
    args = parser.parse_args()
    
    bm25_retriever(args)
    bm25_evaluation(args)
    # evaluation_per_turn(args)
    
    # python component3_retriever/bm25/evaluation.py
    