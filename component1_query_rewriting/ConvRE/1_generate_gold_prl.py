#!/usr/bin/env python3

import os
import json
import torch
import pickle
import random
import logging
import argparse
import pytrec_eval
import numpy as np
from tqdm import tqdm
from os.path import join as oj
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaConfig
from pyserini.dsearch import AnceQueryEncoder
from pyserini.search.faiss import FaissSearcher

from src.data_structure import ConvDataset_topiocqa_rel
from src.models import ANCE


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

def convert_gold_to_trec(gold_file, trec_file):
    with open(gold_file, "r") as f, open(trec_file, "w") as g:
        data = f.readlines()
        for line in data:
            line = json.loads(line)
            qid = line["id"]
            #query = line["query"]
            doc_id = line["pos_docs_id"][0]
            g.write("{} {} {} {}".format(qid,
                                        "Q0",
                                        doc_id,
                                        1,
                                        ))
            g.write('\n')



def get_test_query_embedding(args):
    
    # === Get the data 
    config = RobertaConfig.from_pretrained(
        args.query_encoder,
        finetuning_task="MSMarco",
    )
    query_encoder = ANCE.from_pretrained(args.query_encoder, config=config)
    # query_encoder = AnceQueryEncoder(args.encoder, device=args.device)
    query_tokenizer = RobertaTokenizer.from_pretrained(args.query_encoder, do_lower_case=True)
    test_dataset = ConvDataset_topiocqa_rel(
                            args,
                            query_tokenizer,
                            args.dev_rel_turn_file,
                            add_doc_info=False)
    test_loader = DataLoader(test_dataset, 
                            batch_size = args.eval_batch_size, 
                            shuffle=False, 
                            collate_fn=test_dataset.get_collate_fn(args, add_doc_info = False))

    query_encoder.to(args.device)
    query_encoder.zero_grad()
    embeddings = []
    embedding2id = []

    with torch.no_grad():
        for bc_idx, batch in enumerate(tqdm(test_loader)):
            
            if bc_idx == 10:
                break
            
            query_encoder.eval()
            batch_sample_id = batch["bt_sample_id"]
            
            # test type
            input_ids = batch["bt_pair_query"].to(args.device)
            input_masks = batch["bt_pair_query_mask"].to(args.device)

            query_embs = query_encoder(input_ids, input_masks)
            query_embs = query_embs.detach().cpu().numpy()
            embeddings.append(query_embs)
            embedding2id.extend(batch_sample_id)

    embeddings = np.concatenate(embeddings, axis = 0)
    torch.cuda.empty_cache()

    return embeddings, embedding2id

def search_one_by_one_with_faiss(args, query_embeddings, query_embedding2id):
    
    logger.info("Start Top-k retrieval ...")
    encoder = AnceQueryEncoder(args.query_encoder, device=args.device)
    searcher = FaissSearcher(args.index_dir, encoder)    
    hits = searcher.batch_search(query_embeddings, query_embedding2id, k=args.top_k, threads=20)
    logger.info("retrieval Done!")
    
    retrieved_scores_mat = []
    retrieved_pid_mat = []
    for qid in query_embedding2id:
        score_row = []
        id_row = []
        for i, item in enumerate(hits[qid]):
            score_row.append(item.score)
            id_row.append(item.docid[3:])
        retrieved_scores_mat.append(score_row)
        retrieved_pid_mat.append(id_row)

    return np.array(retrieved_scores_mat), np.array(retrieved_pid_mat)
   
def output_test_res(query_embedding2id,
                    retrieved_scores_mat, # score_mat: score matrix, test_query_num * (top_n * block_num)
                    retrieved_pid_mat, # pid_mat: corresponding passage ids
                    args):
    

    qids_to_ranked_candidate_passages = {}
    topN = args.top_k

    for query_idx in range(len(retrieved_pid_mat)):
        seen_pid = set()
        query_id = query_embedding2id[query_idx]

        top_ann_pid = retrieved_pid_mat[query_idx].copy()
        top_ann_score = retrieved_scores_mat[query_idx].copy()
        selected_ann_idx = top_ann_pid[:topN]
        selected_ann_score = top_ann_score[:topN].tolist()
        rank = 0

        if query_id in qids_to_ranked_candidate_passages:
            pass
        else:
            tmp = [(0, 0)] * topN
            tmp_ori = [0] * topN
            qids_to_ranked_candidate_passages[query_id] = tmp

        for idx, score in zip(selected_ann_idx, selected_ann_score):
            pred_pid = idx    
            # pred_pid = offset2pid[idx]

            if not pred_pid in seen_pid:
                qids_to_ranked_candidate_passages[query_id][rank] = (pred_pid, score)
                rank += 1
                seen_pid.add(pred_pid)


    # for case study and more intuitive observation
    logger.info('Loading query and passages\' real text...')
    logger.info('begin to write the output...')
    output_trec_file = oj(args.qrel_output_path, "dev_dense_rel_res.trec")
    
    with open(output_trec_file, "w") as g:
        for qid, passages in qids_to_ranked_candidate_passages.items():
            rank_list = []
            for i in range(topN):
                pid, score = passages[i]
                rank_list.append(
                    {
                        "doc_id": str(pid),
                        "rank": i+1,
                        "retrieval_score": score,
                    }
                )
                g.write(str(qid) + " Q0 " + str(pid) + " " + str(i + 1) +
                        " " + str(-i - 1 + 200) + " ance\n")
            
            #merged_data.append(
            #    {
            #        "query": query,
            #        "query_id": str(qid),
            #        "conv_id": str(conv_id),
            #        "turn_id": str(turn_id),
            #        "ctxs": rank_list,
            #    })

        #f.write(json.dumps(merged_data, indent=4) + "\n")

    logger.info("output file write ok at {}".format(args.qrel_output_path))

    # print result   
    #res = print_res(output_file, args.gold_qrel_file_path)
 
def print_trec_res(run_file, qrel_file, rel_threshold, input_query_file):
    with open(run_file, 'r' )as f:
        run_data = f.readlines()
    with open(qrel_file, 'r') as f:
        qrel_data = f.readlines()
    
    qrels = {}
    qrels_ndcg = {}
    runs = {}
    
    for line in qrel_data:
        line = line.split(" ")
        query = line[0]
        passage = line[2]
        rel = int(line[3])
        if query not in qrels:
            qrels[query] = {}
        if query not in qrels_ndcg:
            qrels_ndcg[query] = {}

        # for NDCG
        qrels_ndcg[query][passage] = rel
        # for MAP, MRR, Recall
        if rel >= rel_threshold:
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
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "recip_rank", "recall.5", "recall.20", "recall.100"})
    res = evaluator.evaluate(runs)
    map_list = [v['map'] for v in res.values()]
    mrr_list = [v['recip_rank'] for v in res.values()]
    recall_100_list = [v['recall_100'] for v in res.values()]
    recall_20_list = [v['recall_20'] for v in res.values()]
    recall_5_list = [v['recall_5'] for v in res.values()]

    evaluator = pytrec_eval.RelevanceEvaluator(qrels_ndcg, {"ndcg_cut.3"})
    res = evaluator.evaluate(runs)
    ndcg_3_list = [v['ndcg_cut_3'] for v in res.values()]

    mrr_one_nums = 0
    mrr_zero_nums = 0
    res_mrr_dict = improve_judge(input_query_file, mrr_list)
    for key, value in res_mrr_dict.items():
        if (len(value) > 0 and 1 in value[1:]) or len(value) == 1:
            mrr_one_nums += 1
        elif len(value) > 0 and 1 not in value[1:]:
            mrr_zero_nums += 1

    with open(oj("processed_datasets/TopiOCQA", "dev_rel_label_rawq_token.json"), "w") as f:
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

    
    logger.info("---------------------Evaluation results:---------------------")    
    logger.info(res)
    return res

def improve_judge(input_query_file, score_list):

    with open(input_query_file, "r") as f:
        data = f.readlines()
    rel_label = {}
    rel_list = []
    base_score = 0
    for i, line in enumerate(data):
        
        if i == 40:
            break
        
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



def generate_gold_prl(args):
    
    # === Get query embedding ====================
    query_embeddings, query_embedding2id = get_test_query_embedding(args)
    logger.info("Query embedding step, done!")
    logger.info(f"Query embedding shape: {query_embeddings.shape}")
    logger.info(f"Query embedding id length: {len(query_embedding2id)}")
    
    retrieved_scores_mat, retrieved_pid_mat = search_one_by_one_with_faiss(args, query_embeddings, query_embedding2id)
    logger.info("Passage embedding step, done!")
    logger.info(f"Passage embedding score shape: {retrieved_scores_mat.shape}")
    logger.info(f"Passage embedding id shape: {retrieved_pid_mat.shape}")

    output_test_res(query_embedding2id, retrieved_scores_mat, retrieved_pid_mat, args)
    
    output_trec_file = oj(args.qrel_output_path, "dev_dense_rel_res.trec")
    trec_res = print_trec_res(output_trec_file, args.trec_gold_qrel_file_path, args.rel_threshold, args.dev_rel_turn_file)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_encoder", type=str, default="castorini/ance-msmarco-passage")
    parser.add_argument("--retriever_model", type=str, default="bm25", choices=["bm25", "ance"])
    parser.add_argument("--dataset_name", type=str, default="TopiOCQA", choices=["QReCC", "TopiOCQA", "INSCIT"])
    parser.add_argument("--query_format", type=str, default="original", choices=[
        'original', 'human_rewritten', 'all_history', 'same_topic', 't5_rewritten', 'ConvGQR_rewritten',
    ])
    parser.add_argument("--add_gold_topic", action="store_true")
    parser.add_argument("--use_last_response", action="store_true")
    parser.add_argument("--use_answer", action="store_true")
    parser.add_argument("--max_doc_length", type=int, default=384)
    parser.add_argument("--max_concat_length", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--rel_threshold", type=int, default=1)
    parser.add_argument("--top_k", type=int, default="100")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Available GPUs:", torch.cuda.device_count())
    args.device = 'cuda:0'
    args.index_dir = f"corpus/{args.dataset_name}/ance_index"
    args.qrel_output_path = "processed_datasets/TopiOCQA"
    
    set_seed(args.seed)
    
    print("Preprocessing files ...")
    
    ### === 1) generate the qrel file by function
    # Ref: 
    input_file = "processed_datasets/TopiOCQA/dev_new.json"
    args.dev_rel_token_file = "processed_datasets/TopiOCQA/dev_rel_token.json"
    args.dev_rel_turn_file = "processed_datasets/TopiOCQA/dev_rel_turn.json"
    args.trec_gold_qrel_file_path = "processed_datasets/TopiOCQA/dev_rel_turn_gold.trec"
    
    args.dev_rel_label_rawq_token_file = "processed_datasets/TopiOCQA/dev_rel_label_rawq_token.json"
    args.dev_rel_label_rawq_turn_file = "processed_datasets/TopiOCQA/dev_rel_label_rawq_turn.json"
    # create_label_rel_token(input_file, dev_rel_token_file)
    # create_label_rel_turn(input_file, dev_rel_turn_file)
    # convert_gold_to_trec(args.dev_rel_turn_file, args.trec_gold_qrel_file_path)
    
    
    ### === 2) generate the pseudo relevant label (PRL)
    generate_gold_prl(args)
    
    # python component1_query_rewriting/ConvRE/1_generate_gold_prl.py
    