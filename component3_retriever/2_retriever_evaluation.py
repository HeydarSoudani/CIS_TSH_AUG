
import pytrec_eval
import numpy as np
import argparse, os, json, csv

def retiever_evaluation(args):

    print(f"=== Evaluating {args.dataset_name}/{args.query_format}/{args.retriever_model}/topic: {args.add_topic}...")
    # === Read results and gold_qrel files ===========    
    if args.add_topic in ["prev_topics", "cur_topic"]:
        results_file = f"component3_retriever/output_results/{args.dataset_name}/{args.add_topic}+{args.query_format}_{args.retriever_model}_results.trec"
    else:
        results_file = f"component3_retriever/output_results/{args.dataset_name}/{args.query_format}_{args.retriever_model}_results.trec"
    
    with open(results_file, 'r') as f:
        run_data = f.readlines()
    
    gold_qrel_file = f"processed_datasets/{args.dataset_name}/test_gold_qrels.trec"
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

def retiever_evaluation_per_buckets(args):
    print(f"=== Evaluating {args.dataset_name}/{args.query_format}/{args.retriever_model}/topic: {args.add_topic}...")
    
    # === Read files ====================
    if args.add_topic in ["prev_topics", "cur_topic"]:
        results_file = f"component3_retriever/output_results/{args.dataset_name}/{args.add_topic}+{args.query_format}_{args.retriever_model}_results.trec"
    else:
        results_file = f"component3_retriever/output_results/{args.dataset_name}/{args.query_format}_{args.retriever_model}_results.trec"
        
    with open(results_file, 'r') as f:
        run_data = f.readlines()
    
    bucket_file = f"processed_datasets/{args.dataset_name}/turn_buckets/per_{args.bucket_type}.json"
    with open(bucket_file, 'r') as f:
        bucket_data = json.load(f)

    gold_qrel_file = f"processed_datasets/{args.dataset_name}/test_gold_qrels.trec"
    with open(gold_qrel_file, 'r') as f:
        qrel_data = f.readlines()

    # === Prepare gold qrels ==============
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
    
    # === Prepare result qrels ============
    runs = {}
    for line in run_data:
        line = line.split(" ")
        query = line[0]
        passage = line[2]
        rel = int(float(line[4]))
        if query not in runs:
            runs[query] = {}
        runs[query][passage] = rel
    
    # === Select results per bucket =======
    results = {
        "MAP": [],
        "MRR": [],
        "NDCG@3": [],
        "Recall@5": [],
        "Recall@10": [],
        "Recall@20": [],
        "Recall@100": []
    }
    for bk_title, bk_samples in bucket_data.items():
        print(f"= Bucket title: {bk_title}")
        bucket_runs = {key: runs[key] for key in bk_samples if key in runs}
        
        # === Calculate eval metrics ==========
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "recip_rank", "recall.5", "recall.10", "recall.20", "recall.100"})
        res = evaluator.evaluate(bucket_runs)
        map_list = [v['map'] for v in res.values()]
        mrr_list = [v['recip_rank'] for v in res.values()]
        recall_100_list = [v['recall_100'] for v in res.values()]
        recall_20_list = [v['recall_20'] for v in res.values()]
        recall_10_list = [v['recall_10'] for v in res.values()]
        recall_5_list = [v['recall_5'] for v in res.values()]

        evaluator = pytrec_eval.RelevanceEvaluator(qrels_ndcg, {"ndcg_cut.3"})
        res = evaluator.evaluate(runs)
        ndcg_3_list = [v['ndcg_cut_3'] for v in res.values()]
        
        results["MAP"].append(np.average(map_list))
        results["MRR"].append(np.average(mrr_list))
        results["NDCG@3"].append(np.average(ndcg_3_list))
        results["Recall@5"].append(np.average(recall_5_list))
        results["Recall@10"].append(np.average(recall_10_list))
        results["Recall@20"].append(np.average(recall_20_list))
        results["Recall@100"].append(np.average(recall_100_list))
        
    print("---------------------Evaluation results:---------------------")
    print(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever_model", type=str, default="ance", choices=["bm25", "ance"])
    parser.add_argument("--dataset_name", type=str, default="TopiOCQA", choices=["QReCC", "TopiOCQA", "INSCIT"])
    parser.add_argument("--bucket_type", type=str, default="shift", choices=["turn_number", "shift"])
    parser.add_argument("--query_format", type=str, default="LLM4CS", choices=[
        'original', 'human_rewritten', 'all_history', 'same_topic',
        't5_rewritten', 'ConvGQR_rewritten', 'LLM4CS'
    ])
    parser.add_argument("--add_topic", default="no", choices=["no", "cur_topic", "prev_topics"])
    parser.add_argument("--rel_threshold", type=int, default="1")
    args = parser.parse_args()
    
    retiever_evaluation(args)
    # retiever_evaluation_per_buckets(args)

# python component3_retriever/2_retriever_evaluation.py
