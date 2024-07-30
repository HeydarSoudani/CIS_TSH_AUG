import json
import argparse
import pytrec_eval
import numpy as np


def create_label_rel_token(inputs, output):
    with open(inputs, "r") as f, open(output, "w") as g:
        obj = f.readlines()
        total_nums = len(obj)
        query_pair_nums = 0
        for i in range(total_nums):
            obj[i] = json.loads(obj[i])
            sample_id = obj[i]['id']
            conv_id = obj[i]['conv_id']
            turn_id = obj[i]['turn_id']
            history_query = obj[i]["history_query"]
            history_answer = obj[i]["history_answer"]
            query = obj[i]["query"]
            answer = obj[i]["answer"]
            pos_docs_id = obj[i]["pos_docs_id"]

            token_set = []
            for key in history_query:
                sent = key.strip().split()
                token_set.extend(sent)

            if int(turn_id) > 1: 
                g.write(
                    json.dumps({
                        "id": str(conv_id) + '-' + str(turn_id) + '-0',
                        "conv_id": conv_id,
                        "turn_id": turn_id,
                        "query": query,
                        "query_pair": "",
                        #"history_answer": history_answer,
                        #"last_response": last_response,
                        #"pos_docs": pos_docs,
                        "pos_docs_id": pos_docs_id,
                    }) + "\n")
                query_pair_nums += 1

                for tid in range(0, len(token_set)):
                    query_pair = token_set[tid]
                    g.write(
                        json.dumps({
                            "id": str(conv_id) + '-' + str(turn_id) + '-' + str(tid + 1),
                            "conv_id": conv_id,
                            "turn_id": turn_id,
                            "query": query,
                            "query_pair": query_pair,
                            #"history_answer": history_answer,
                            #"last_response": last_response,
                            #"pos_docs": pos_docs,
                            "pos_docs_id": pos_docs_id,
                        }) + "\n")
                    query_pair_nums += 1
        print(query_pair_nums)

def create_label_rel_turn(inputs, output):
    with open(inputs, "r") as f, open(output, "w") as g:
        obj = f.readlines()
        total_nums = len(obj)
        query_pair_nums = 0
        for i in range(total_nums):
            obj[i] = json.loads(obj[i])
            sample_id = obj[i]['id']
            conv_id = obj[i]['conv_id']
            turn_id = obj[i]['turn_id']
            history_query = obj[i]["history_query"]
            history_rewrite = obj[i]["history_rewrite"]
            history_answer = obj[i]["history_answer"]
            last_response = obj[i]["last_response"]
            topic = obj[i]["topic"]
            sub_topic = obj[i]["sub_topic"]
            query = obj[i]["query"]
            rewrite = obj[i]["rewrite"]
            answer = obj[i]["answer"]
            pos_docs = obj[i]["pos_docs"]
            pos_docs_id = obj[i]["pos_docs_id"]

            if int(turn_id) > 1: # if first turn
                g.write(
                    json.dumps({
                        "id": str(conv_id) + '-' + str(turn_id) + '-0',
                        "conv_id": conv_id,
                        "turn_id": turn_id,
                        "query": query,
                        "rewrite": rewrite,
                        "query_pair": "",
                        "rewrite_query_pair": "",
                        "history_answer": history_answer,
                        "last_response": last_response,
                        "topic": topic,
                        "sub_topic": sub_topic,
                        "pos_docs": pos_docs,
                        "pos_docs_id": pos_docs_id,
                    }) + "\n")
                query_pair_nums += 1

                for tid in range(0, int(turn_id) - 1):
                    query_pair = history_query[tid]
                    rewrite_query_pair = history_rewrite[tid]
                    #turn_pair_id = str(turn_id) + '-' + str(tid + 1)
                    g.write(
                        json.dumps({
                            "id": str(conv_id) + '-' + str(turn_id) + '-' + str(tid + 1),
                            "conv_id": conv_id,
                            "turn_id": turn_id,
                            "query": query,
                            "rewrite": rewrite,
                            "query_pair": query_pair,
                            "rewrite_query_pair": rewrite_query_pair,
                            "history_answer": history_answer,
                            "last_response": last_response,
                            "topic": topic,
                            "sub_topic": sub_topic,
                            "pos_docs": pos_docs,
                            "pos_docs_id": pos_docs_id,
                        }) + "\n")
                    query_pair_nums += 1
        print(query_pair_nums)

def bm25_evaluation(args):
    topic_text = "with gold topic" if args.add_gold_topic else "wo gold topic"
    print(f"=== Evaluating {args.dataset_name}/{args.query_format}/{args.retriever_model}/{topic_text}...")
    # === Read results and gold_qrel files ===========    
    if args.add_gold_topic:
        results_file = f"component3_retriever/output_results/{args.dataset_name}/topic+{args.query_format}_{args.retriever_model}_results.trec"
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
    
    return mrr_list

def improve_judge(input_query_file, score_list):
    with open(input_query_file, "r") as f:
        data = f.readlines()
    rel_label = {}
    rel_list = []
    base_score = 0
    for i, line in enumerate(data):
        line = json.loads(line)
        id_list = line["id"].split('-')
        conv_id = int(id_list[0]) 
        turn_id = int(id_list[1])
        type_id = int(id_list[-1])
        if (i + 1) != len(data):
            next_turn_id = int(json.loads(data[i + 1])["id"].split('-')[1])

        if type_id == 0 and turn_id > 1: 
            base_score = score_list[i]
        elif type_id > 0 and turn_id > 1: 
            if score_list[i] > base_score:
                rel_list.append(1)
            else:
                rel_list.append(0)
        
        if (i + 1) == len(data) or turn_id != next_turn_id:
            rel_label[id_list[0] + '-1'] = []
            rel_label[id_list[0] + '-' + id_list[1]] = rel_list
            rel_list = []
            base_score = 0

    return rel_label

def main(mrr_list, input_query_file, output_file):
    
    mrr_one_nums = 0
    mrr_zero_nums = 0
    res_mrr_dict = improve_judge(input_query_file, mrr_list)
    for key, value in res_mrr_dict.items():
        if (len(value) > 0 and 1 in value[1:]) or len(value) == 1:
            mrr_one_nums += 1
        elif len(value) > 0 and 1 not in value[1:]:
            mrr_zero_nums += 1

    with open(output_file, "w") as f:
        for key, value in res_mrr_dict.items():
            id_list = key.split('-')
            conv_id = id_list[0]
            turn_id = id_list[1]
            f.write(
                json.dumps({
                    "id": str(key),
                    "conv_id": str(conv_id),
                    "turn_id": str(turn_id),
                    "rel_label": value
                }) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever_model", type=str, default="bm25", choices=["bm25", "ance"])
    parser.add_argument("--dataset_name", type=str, default="TopiOCQA", choices=["QReCC", "TopiOCQA", "INSCIT"])
    parser.add_argument("--query_format", type=str, default="original", choices=[
        'original', 'human_rewritten', 'all_history', 'same_topic', 't5_rewritten', 'ConvGQR_rewritten',
    ])
    parser.add_argument("--add_gold_topic", action="store_true")
    parser.add_argument("--rel_threshold", type=int, default="1")
    args = parser.parse_args()
    
    input_file = "processed_datasets/TopiOCQA/dev_new.json"
    dev_rel_token_file = "processed_datasets/TopiOCQA/dev_rel_token.json"
    dev_rel_turn_file = "processed_datasets/TopiOCQA/dev_rel_turn.json"
    dev_rel_label_rawq_token_file = "processed_datasets/TopiOCQA/dev_rel_label_rawq_token.json"
    
    # create_label_rel_token(input_file, dev_rel_token_file)
    # create_label_rel_turn(input_file, dev_rel_turn_file)
    mrr_list = bm25_evaluation(args)
    main(mrr_list, dev_rel_token_file, dev_rel_label_rawq_token_file)
    
    # python component1_query_rewriting/ConvRE/1_generate_prl.py
    