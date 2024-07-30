#!/usr/bin/env python3

### =================================================================================
# Ref: https://github.com/kyriemao/T5QR/blob/main/evaluation/eval_dense_retrieval.py
### =================================================================================

# from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import os
import pickle
import sys
sys.path.append('..')
sys.path.append('.')

import json
import time
import copy
import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from pprint import pprint
# from trec_eval import trec_eval
from os.path import join as oj

from src.dense_retrievers import load_dense_retriever
from src.utils import set_seed, get_has_qrel_label_sample_ids


def build_faiss_index(args):
    logger.info("Building Faiss Index...")
    # ngpu = faiss.get_num_gpus()
    ngpu = args.n_gpu_for_faiss
    gpu_resources = []
    tempmem = -1

    for i in range(ngpu):
        res = faiss.StandardGpuResources()
        if tempmem >= 0:
            res.setTempMemory(tempmem)
        gpu_resources.append(res)

    cpu_index = faiss.IndexFlatIP(768)  
    index = None
    if args.use_gpu_in_faiss:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.usePrecomputed = False
        # gpu_vector_resources, gpu_devices_vector
        vres = faiss.GpuResourcesVector()
        vdev = faiss.Int32Vector()
        for i in range(0, ngpu):
            vdev.push_back(i)
            vres.push_back(gpu_resources[i])
        gpu_index = faiss.index_cpu_to_gpu_multiple(vres,
                                                    vdev,
                                                    cpu_index, co)
        index = gpu_index
    else:
        index = cpu_index

    return index

def get_query_embs(args):
    query_tokenizer, query_encoder = load_dense_retriever("ANCE", "query", args.retriever_path)
    query_encoder = query_encoder.to(args.device)
    
    query_encoding_dataset = []
    # with open(args.eval_file_path, "r") as f:
    #     for line in f:
    #         record = json.loads(line.strip())
    #         sample_id = record['sample_id']
    #         query = record[args.eval_field_name]
    #         query_encoding_dataset.append([sample_id, query])

    with open(args.eval_file_path, "r") as f:
        data = json.load(f)
    for record in data:
        sample_id = record['sample_id']
        query = record[args.eval_field_name]
        query_encoding_dataset.append([sample_id, query])

    def query_encoding_collate_fn(batch):
        bt_sample_ids, bt_src_seq = list(zip(*batch)) # unzip
        bt_src_encoding = query_tokenizer(bt_src_seq, 
                                          padding="longest", 
                                          max_length=args.max_query_length, 
                                          truncation=True, 
                                          return_tensors="pt")
        bt_input_ids, bt_attention_mask = bt_src_encoding.input_ids, bt_src_encoding.attention_mask
        return {"bt_sample_ids": bt_sample_ids, 
                "bt_input_ids":bt_input_ids, 
                "bt_attention_mask":bt_attention_mask}

    test_loader = DataLoader(query_encoding_dataset, 
                             batch_size=2, 
                             shuffle=False, 
                             collate_fn=query_encoding_collate_fn)

    query_encoder.zero_grad()
    embeddings = []
    eid2sid = []    # embedding idx to sample id
    has_qrel_label_sample_ids = get_has_qrel_label_sample_ids(args.qrel_file_path)
    with torch.no_grad():
        for q_idx, batch in enumerate(tqdm(test_loader)):
            
            if q_idx == 1:
                break
            
            query_encoder.eval()
            bt_sample_ids = batch["bt_sample_ids"]
            bt_input_ids = batch['bt_input_ids'].to(args.device)
            bt_attention_mask = batch['bt_attention_mask'].to(args.device)

            # print(f"bt_sample_ids: {bt_sample_ids}")
            # print(f"bt_input_ids: {bt_input_ids}")

            query_embs = query_encoder(bt_input_ids, bt_attention_mask)
            query_embs = query_embs.detach().cpu().numpy()

            # print(f"query embs: {query_embs}")

            sifted_sample_ids = []
            sifted_query_embs = []
            for i in range(len(bt_sample_ids)):
                if bt_sample_ids[i] not in has_qrel_label_sample_ids:
                    continue
                sifted_sample_ids.append(bt_sample_ids[i])
                sifted_query_embs.append(query_embs[i].reshape(1, -1))
        
            if len(sifted_query_embs) > 0:
                sifted_query_embs = np.concatenate(sifted_query_embs)

                embeddings.append(sifted_query_embs)
                eid2sid.extend(sifted_sample_ids)
            else:
                continue

        embeddings = np.concatenate(embeddings, axis = 0)
        # print(embeddings)
        # print(embeddings.shape)
        # print(eid2sid)
    
    torch.cuda.empty_cache()
    return embeddings, eid2sid

def faiss_flat_retrieval_one_by_one_and_finally_merge(args, query_embs):
    index = build_faiss_index(args)

    merged_candidate_matrix = None
    # Automaticall get the number of doc blocks
    for filename in os.listdir(args.index_path):
        try:
            args.passage_block_num = int(filename.split(".")[1]) + 1
        except:
            continue
    print("Automatically detect that the number of doc blocks is: {}".format(args.passage_block_num))
    
    for block_id in range(args.passage_block_num):
        
        # if block_id == 1:
        #     break
        
        logger.info("Loading doc block " + str(block_id))

        # load doc embeddings
        with open(os.path.join(args.index_path, "passage_emb_block_{}.pb".format(block_id)), 'rb') as handle:
            cur_doc_embs = pickle.load(handle)
        with open(os.path.join(args.index_path, "passage_embid_block_{}.pb".format(block_id)), 'rb') as handle:
            cur_eid2did = pickle.load(handle)
            if isinstance(cur_eid2did, list):
                cur_eid2did = np.array(cur_eid2did)

        # Split to avoid the doc embeddings to be too large
        num_total_doc_per_block = len(cur_doc_embs)
        num_doc_per_split = 5000    # please set it according to your GPU size. 700w doc needs ~28GB
        num_split_block = max(1, num_total_doc_per_block // num_doc_per_split)
        logger.info("num_total_doc: {}".format(num_total_doc_per_block))
        logger.info("num_doc_per_split: {}".format(num_doc_per_split))
        logger.info("num_split_block: {}".format(num_split_block))
        cur_doc_embs_list = np.array_split(cur_doc_embs, num_split_block)
        cur_eid2did_list = np.array_split(cur_eid2did, num_split_block)
        
        for split_idx in range(len(cur_doc_embs_list)):
            
            # if split_idx == 1:
            #     break
            
            cur_doc_embs = cur_doc_embs_list[split_idx]
            cur_eid2did = cur_eid2did_list[split_idx]
            if split_idx % 200 == 0:
                logger.info("Adding block {} split {} into index...".format(block_id, split_idx))
            index.add(cur_doc_embs)
            
            # print(f"passage: {cur_doc_embs[0]}")
            # print(f"query: {query_embs[0]}")
            
            # ann search
            tb = time.time()
            D, I = index.search(query_embs, args.top_n)
            
            # print(f"D: {D}")
            # print(f"I: {I}")
            
            elapse = time.time() - tb
            if split_idx % 200 == 0:
                logger.info({
                    'time cost': elapse,
                    'query num': query_embs.shape[0],
                    'time cost per query': elapse / query_embs.shape[0]
                })

            candidate_did_matrix = cur_eid2did[I] # doc embedding_idx -> real doc id
            D = D.tolist()
            candidate_did_matrix = candidate_did_matrix.tolist()
            candidate_matrix = []

            for score_list, doc_list in zip(D, candidate_did_matrix):
                candidate_matrix.append([])
                for score, doc in zip(score_list, doc_list):
                    candidate_matrix[-1].append((score, doc))
                assert len(candidate_matrix[-1]) == len(doc_list)
            assert len(candidate_matrix) == I.shape[0]

            index.reset()
            del cur_doc_embs
            del cur_eid2did

            if merged_candidate_matrix == None:
                merged_candidate_matrix = candidate_matrix
                continue
            
            # Merge
            merged_candidate_matrix_tmp = copy.deepcopy(merged_candidate_matrix)
            merged_candidate_matrix = []
            for merged_list, cur_list in zip(merged_candidate_matrix_tmp,
                                            candidate_matrix):
                p1, p2 = 0, 0
                merged_candidate_matrix.append([])
                while p1 < args.top_n and p2 < args.top_n:
                    if merged_list[p1][0] >= cur_list[p2][0]:
                        merged_candidate_matrix[-1].append(merged_list[p1])
                        p1 += 1
                    else:
                        merged_candidate_matrix[-1].append(cur_list[p2])
                        p2 += 1
                while p1 < args.top_n:
                    merged_candidate_matrix[-1].append(merged_list[p1])
                    p1 += 1
                while p2 < args.top_n:
                    merged_candidate_matrix[-1].append(cur_list[p2])
                    p2 += 1

    merged_D, merged_I = [], []

    for merged_list in merged_candidate_matrix:
        merged_D.append([])
        merged_I.append([])
        for candidate in merged_list:
            merged_D[-1].append(candidate[0])
            merged_I[-1].append(candidate[1])
    merged_D, merged_I = np.array(merged_D), np.array(merged_I)

    print(merged_D)
    logger.info(merged_D.shape)
    logger.info(merged_I.shape)

    return merged_D, merged_I

def search_one_by_one_with_faiss(args, passge_embeddings_dir, index, query_embeddings, topN):
    merged_candidate_matrix = None

    for block_id in range(args.passage_block_num):
        logger.info("Loading passage block " + str(block_id))
        
        # if block_id == 1:
        #     break
        
        passage_embedding = None
        passage_embedding2id = None
        try:
            with open(
                    oj(
                        passge_embeddings_dir,
                        "passage_emb_block_" + str(block_id) + ".pb"),
                    'rb') as handle:
                passage_embedding = pickle.load(handle)
            with open(
                    oj(
                        passge_embeddings_dir,
                        "passage_embid_block_" + str(block_id) + ".pb"),
                    'rb') as handle:
                passage_embedding2id = pickle.load(handle)
        except:
            break
        logger.info('passage embedding shape: ' + str(passage_embedding.shape))
        logger.info("query embedding shape: " + str(query_embeddings.shape))
        index.add(passage_embedding)

        print(f"passage: {passage_embedding[0]}")
        print(f"query: {query_embeddings[0]}")

        # ann search
        tb = time.time()
        D, I = index.search(query_embeddings, topN)
        
        print(f"D: {D}")
        print(f"I: {I}")
        
        elapse = time.time() - tb
        logger.info({
            'time cost': elapse,
            'query num': query_embeddings.shape[0],
            'time cost per query': elapse / query_embeddings.shape[0]
        })

        candidate_id_matrix = passage_embedding2id[I] # passage_idx -> passage_id
        D = D.tolist()
        candidate_id_matrix = candidate_id_matrix.tolist()
        candidate_matrix = []

        for score_list, passage_list in zip(D, candidate_id_matrix):
            candidate_matrix.append([])
            for score, passage in zip(score_list, passage_list):
                candidate_matrix[-1].append((score, passage))
            assert len(candidate_matrix[-1]) == len(passage_list)
        assert len(candidate_matrix) == I.shape[0]

        index.reset()
        del passage_embedding
        del passage_embedding2id

        if merged_candidate_matrix == None:
            merged_candidate_matrix = candidate_matrix
            continue
        
        # merge
        merged_candidate_matrix_tmp = copy.deepcopy(merged_candidate_matrix)
        merged_candidate_matrix = []
        for merged_list, cur_list in zip(merged_candidate_matrix_tmp,
                                         candidate_matrix):
            p1, p2 = 0, 0
            merged_candidate_matrix.append([])
            while p1 < topN and p2 < topN:
                if merged_list[p1][0] >= cur_list[p2][0]:
                    merged_candidate_matrix[-1].append(merged_list[p1])
                    p1 += 1
                else:
                    merged_candidate_matrix[-1].append(cur_list[p2])
                    p2 += 1
            while p1 < topN:
                merged_candidate_matrix[-1].append(merged_list[p1])
                p1 += 1
            while p2 < topN:
                merged_candidate_matrix[-1].append(cur_list[p2])
                p2 += 1

    merged_D, merged_I = [], []
    for merged_list in merged_candidate_matrix: # len(merged_candidate_matrix) = query_nums len([0]) = query_num * topk
        merged_D.append([])
        merged_I.append([])
        for candidate in merged_list: # len(merged_list) = query_num * topk
            merged_D[-1].append(candidate[0])
            merged_I[-1].append(candidate[1])
    
    merged_D, merged_I = np.array(merged_D), np.array(merged_I)
    logger.info(merged_I.shape)
    return merged_D, merged_I

def output_test_res(query_embedding2id,
                    retrieved_scores_mat, # score_mat: score matrix, test_query_num * (top_n * block_num)
                    retrieved_pid_mat, # pid_mat: corresponding passage ids
                    offset2pid,
                    args):
    

    qids_to_ranked_candidate_passages = {}
    topN = args.top_n

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
            pred_pid = offset2pid[idx]

            if not pred_pid in seen_pid:
                qids_to_ranked_candidate_passages[query_id][rank] = (pred_pid, score)
                rank += 1
                seen_pid.add(pred_pid)


    # for case study and more intuitive observation
    logger.info('Loading query and passages\' real text...')
    
    # query
    # qid2query = {}
    # qid2convid = {}
    # qid2turnid = {}
    # with open(args.test_file_path, 'r') as f:
    #     data = f.readlines()
    # for record in data:
    #     record = json.loads(record.strip())
        #qid2query[record["id"]] = record["query"]
        #qid2convid[record["id"]] = record["conv_id"]
        #qid2turnid[record["id"]] = record["turn_id"]
            
    
    # all passages
    #all_passages = load_collection(args.passage_collection_path)

    # write to file
    logger.info('begin to write the output...')

    #output_file = oj(args.qrel_output_path, "ANCE_QRIR_kd_prefix_oracle+answer_res.json")
    output_trec_file = oj(args.retrieval_output_path, "ANCE_t5_datasetrewrite_res.trec")
    merged_data = []
    #with open(output_file, "w") as f, open(output_trec_file, "w") as g:
    with open(output_trec_file, "w") as g:
        for qid, passages in qids_to_ranked_candidate_passages.items():
            #query = qid2query[qid]
            #conv_id = qid2convid[qid]
            #turn_id = qid2turnid[qid]
            #rank_list = []
            for i in range(topN):
                pid, score = passages[i]
                #passage = all_passages[pid]
                #rank_list.append(
                #    {
                #        "doc_id": str(pid),
                #        "rank": i+1,
                #        "retrieval_score": score,
                #    }
                #)
                g.write(str(qid) + " Q0 " + str(pid) + " " + str(i + 1) + " " + str(-i - 1 + 200) + ' ' + str(score) + " ance\n")
            
            #merged_data.append(
            #    {
            #        "query": query,
            #        "query_id": str(qid),
                    #"conv_id": str(conv_id),
                    #"turn_id": str(turn_id),
            #        "ctxs": rank_list,
            #    })

        #f.write(json.dumps(merged_data, indent=4) + "\n")

    logger.info("output file write ok at {}".format(args.retrieval_output_path))

    # print result   
    #res = print_res(output_file, args.gold_qrel_file_path)
    # trec_res = print_trec_res(output_trec_file, args.trec_gold_qrel_file_path, args.rel_threshold)
    # return trec_res

def dense_retrieval(args):
    query_embs, eid2sid = get_query_embs(args)
    score_mat, did_mat = faiss_flat_retrieval_one_by_one_and_finally_merge(args, query_embs)
    
    # write to file
    run_trec_file = os.path.join(args.retrieval_output_path, "original_ance_results.trec")
    run_trec_file = os.path.join(args.retrieval_output_path, "original_ance_results_p10.trec")
    with open(run_trec_file, "w") as f:
        for eid in range(len(did_mat)):
            sample_id = eid2sid[eid]
            retrieval_scores = score_mat[eid]
            retrieved_dids = did_mat[eid]
            for i in range(len(retrieval_scores)):
                rank = i + 1
                doc_id = retrieved_dids[i]
                rank_score = args.top_n - i # use the rank score for pytrec
                real_score = retrieval_scores[i] 
                f.write("{} {} {} {} {} {} {}".format(sample_id, "Q0", doc_id, rank, rank_score, real_score, "ance"))
                f.write('\n')
            
    # evaluation
    # trec_eval(run_trec_file, args.qrel_file_path, args.retrieval_output_path, args.rel_threshold)
    
def dense_retrieval_2(args):
    index = build_faiss_index(args)
    query_embeddings, query_embedding2id = get_query_embs(args)
    retrieved_scores_mat, retrieved_pid_mat = search_one_by_one_with_faiss(
                                                    args,
                                                    args.index_path, 
                                                    index, 
                                                    query_embeddings, 
                                                    args.top_n) 
    # print(f"retrieved_scores_mat: {retrieved_scores_mat}")
    
    with open(args.passage_offset2pid_path, "rb") as f:
        offset2pid = pickle.load(f)
    
    output_test_res(query_embedding2id,
                    retrieved_scores_mat,
                    retrieved_pid_mat,
                    offset2pid,
                    args)
            
    # evaluation
    # trec_eval(run_trec_file, args.qrel_file_path, args.retrieval_output_path, args.rel_threshold)
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # processed_datasets/TopiOCQA/dev_qrecc_format.json
    # component3_retriever/input_data/TopiOCQA/T5QR/t5_rewrite.json

    # pretrained_passage_encoder = "sentence-transformers/msmarco-roberta-base-ance-firstp"   # passage encoder!!!
    # pretrained_passage_encoder = "castorini/ance-msmarco-passage"

    parser.add_argument("--eval_file_path", type=str, default="component3_retriever/input_data/TopiOCQA/T5QR/t5_rewrite.json")
    parser.add_argument("--eval_field_name", type=str, default="t5_rewrite", help="Field name of the rewrite in the eval_file. E.g., t5_rewrite")
    parser.add_argument("--index_path", type=str, default="corpus/TopiOCQA/firstp/dense_embedded")
    parser.add_argument("--passage_offset2pid_path", type=str, default="corpus/TopiOCQA/dense_tokenized/offset2pid.pickle")
    parser.add_argument("--qrel_file_path", type=str, default="processed_datasets/TopiOCQA/test_gold_qrels.trec")
    parser.add_argument("--retriever_path", type=str, default="sentence-transformers/msmarco-roberta-base-ance-firstp")
    parser.add_argument("--retrieval_output_path", type=str, default="component3_retriever/output_results/TopiOCQA")
    parser.add_argument("--max_query_length", type=int, default=64, help="Max single query length") # 32
    parser.add_argument("--use_gpu_in_faiss", action="store_true", help="whether to use gpu in faiss or not.")
    parser.add_argument("--n_gpu_for_faiss", type=int, default=1, help="should be set if use_gpu_in_faiss")
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--passage_block_num", type=int, default=11)
    parser.add_argument("--rel_threshold", type=int, default=1, help="CAsT-20: 2, Others: 1")    
    parser.add_argument("--force_emptying_dir", action="store_true", help="Force to empty the (output) dir.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print(f"Device: {args.device}")
    
    logger.info("---------------------The arguments are:---------------------")
    pprint(args)
    
    set_seed(args) 
    dense_retrieval(args)
    # dense_retrieval_2(args)
    logger.info("Dense retrieval and evaluation finish!")
    
    # python component3_retriever/dense_ANCE/topiocqa_ance_retriever.py