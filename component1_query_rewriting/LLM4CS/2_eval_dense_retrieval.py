#!/usr/bin/env python3

### ==============================================================================
# Ref: https://github.com/kyriemao/LLM4CS/blob/main/evaluation/eval_dense_retrieval.py
# Ref: https://github.com/castorini/pyserini/blob/master/pyserini/encode/_ance.py
### ==============================================================================

import os
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from pyserini.dsearch import AnceQueryEncoder
from pyserini.search.faiss import FaissSearcher

from src.utils import set_seed, get_finished_sample_ids, get_has_qrel_label_sample_ids


# def set_seed(seed):
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

def batch_closest_candidate(embeddings, affiliated_embeddings=None):
    has_aff = False
    if affiliated_embeddings is not None:
        has_aff = True
        
    res = []
    res_aff = []    # corresponding affiliated_embeddings of embeddings.
    for i in range(embeddings.shape[0]):
        # Calculate the dot product of all pairs of embeddings in the batch
        dot_products = np.dot(embeddings[i], embeddings[i].T)

        # Calculate the sum of each row to get the total dot product for each candidate
        candidate_dots = np.sum(dot_products, axis=1)

        # Find the index of the candidate with the highest total dot product
        closest_idx = np.argmax(candidate_dots)

        # Return the embedding for the closest candidate
        res.append(embeddings[i][closest_idx].reshape(1, -1))

        if has_aff:
            res_aff.append(affiliated_embeddings[i][closest_idx].reshape(1, -1))

    return np.concatenate(res, axis=0), np.concatenate(res_aff, axis=0) if has_aff else None

def return_final_ebmbedding(
    query_embeddings,
    response_embeddings,
    n_query_candidate,
    n_response_candidate,
    query_eid2sid,
    response_eid2sid
):
    
    # filter out duplicate sample_ids
    eid2sid = query_eid2sid if query_eid2sid else response_eid2sid
    new_eid2sid = []
    eid2sid_set = set()
    for x in eid2sid:
        if x not in eid2sid_set:
            new_eid2sid.append(x)
            eid2sid_set.add(x)
    eid2sid = new_eid2sid
    
    torch.cuda.empty_cache()
    
    if n_query_candidate == 1 and n_response_candidate == 1:
        return (query_embeddings + response_embeddings) / 2, eid2sid  
    elif n_query_candidate >= 1 and n_response_candidate > 1:
        query_embeddings = query_embeddings.reshape(query_embeddings.shape[0] // n_query_candidate, n_query_candidate, query_embeddings.shape[1])
        response_embeddings = response_embeddings.reshape(response_embeddings.shape[0] // n_response_candidate, n_response_candidate, response_embeddings.shape[1])
        if args.aggregation_method == "maxprob":
            embeddings = (query_embeddings[:, 0, :] + response_embeddings[:, 0, :]) / 2
            return embeddings, eid2sid
        elif args.aggregation_method == "mean":
            embeddings = np.concatenate([query_embeddings, response_embeddings], axis = 1).mean(axis=1)
            return embeddings, eid2sid
        elif args.aggregation_method == "sc":
            if n_query_candidate == 1:
                query_embeddings = query_embeddings[:, 0, :]
                response_embeddings, _ = batch_closest_candidate(response_embeddings)
            else:
                query_embeddings, response_embeddings = batch_closest_candidate(query_embeddings, response_embeddings)
            return (query_embeddings + response_embeddings) / 2, eid2sid  
        else:
            raise NotImplementedError
    elif n_response_candidate == 0: # only query (rewrite)
        query_embeddings = query_embeddings.reshape(query_embeddings.shape[0] // n_query_candidate, n_query_candidate, query_embeddings.shape[1])
        if args.aggregation_method == "maxprob":
            query_embeddings = query_embeddings[:, 0, :]
        elif args.aggregation_method == "mean":
            query_embeddings = np.mean(query_embeddings, axis=1)
        elif args.aggregation_method == "sc":
            query_embeddings, _ = batch_closest_candidate(query_embeddings)
        else:
            raise NotImplementedError
        return query_embeddings, eid2sid
    else:   # only response
        response_emebddings = response_embeddings.reshape(response_embeddings.shape[0] // n_response_candidate, n_response_candidate, response_embeddings.shape[1])
        if args.aggregation_method == "maxprob":
            response_emebddings = response_emebddings[:, 0, :]
        elif args.aggregation_method == "mean":
            response_embeddings = np.mean(response_embeddings, axis=1)
        elif args.aggregation_method == "sc":
            response_embeddings, _ = batch_closest_candidate(response_embeddings)
        return response_emebddings, eid2sid

def llm4cs_retriever(args):
    
    print("Preprocessing files ...")
    index_dir = f"{args.index_dir_base_path}/{args.dataset_name}/ance_index"
    qrel_file_path = f"processed_datasets/{args.dataset_name}/test_gold_qrels.trec"
    rewriting_file = f"component3_retriever/input_data/{args.dataset_name}/LLM4CS/rewrites.jsonl"
    
    # === Query encoder model ===================
    encoder = AnceQueryEncoder(args.encoder, device=args.device)
    tokenizer = RobertaTokenizer.from_pretrained(args.encoder, do_lower_case=True)
    
    # === Defining functions ====================
    def query_encoding_collate_fn(batch):
        bt_sample_ids, bt_src_query = list(zip(*batch)) # unzip
        bt_src_query_encoding = tokenizer(bt_src_query, 
                                            padding="longest", 
                                            max_length=args.max_query_length, 
                                            truncation=True, 
                                            return_tensors="pt")
        
        bt_q_input_ids, bt_q_attention_mask = bt_src_query_encoding.input_ids, bt_src_query_encoding.attention_mask
        
        return {
                "bt_input": bt_src_query,
                "bt_sample_ids": bt_sample_ids, 
                "bt_input_ids":bt_q_input_ids, 
                "bt_attention_mask":bt_q_attention_mask}
    
    def response_encoding_collate_fn(batch):
        bt_sample_ids, bt_src_doc = list(zip(*batch))
        bt_src_doc_encoding = tokenizer(bt_src_doc, 
                                          padding="longest", 
                                          max_length=512, 
                                          truncation=True, 
                                          return_tensors="pt")
        bt_d_input_ids, bt_d_attention_mask = bt_src_doc_encoding.input_ids, bt_src_doc_encoding.attention_mask
        return {
                "bt_input": bt_src_doc,
                "bt_sample_ids": bt_sample_ids, 
                "bt_input_ids":bt_d_input_ids, 
                "bt_attention_mask":bt_d_attention_mask}
    
    def topic_encoding_collate_fn(batch):
        bt_sample_ids, bt_src_doc = list(zip(*batch))
        bt_src_doc_encoding = tokenizer(bt_src_doc, 
                                          padding="longest", 
                                          max_length=16, 
                                          truncation=True, 
                                          return_tensors="pt")
        bt_d_input_ids, bt_d_attention_mask = bt_src_doc_encoding.input_ids, bt_src_doc_encoding.attention_mask
        return {
                "bt_input": bt_src_doc,
                "bt_sample_ids": bt_sample_ids, 
                "bt_input_ids":bt_d_input_ids, 
                "bt_attention_mask":bt_d_attention_mask}
    
    def forward_pass(test_loader, encoder, has_qrel_label_sample_ids):
        embeddings = []
        eid2sid = []
        # encoder.zero_grad()
        with torch.no_grad():
            for batch in tqdm(test_loader):
                # encoder.eval()                
                bt_samples = batch["bt_input"]
                bt_sample_ids = batch["bt_sample_ids"]
                bt_input_ids = batch['bt_input_ids'].to(args.device)
                bt_attention_mask = batch['bt_attention_mask'].to(args.device)
                embs = encoder.encode(bt_samples[0])
                # embs = encoder.encode(bt_input_ids, bt_attention_mask)
                # embs = embs.detach().cpu().numpy()
                
                sifted_sample_ids = []
                sifted_embs = []
                for i in range(len(bt_sample_ids)):
                    if bt_sample_ids[i] not in has_qrel_label_sample_ids:
                        continue
                    sifted_sample_ids.append(bt_sample_ids[i])
                    # sifted_embs.append(embs[i].reshape(1, -1))
                    sifted_embs.append(embs.reshape(1, -1))
                    # sifted_embs.append(embs[i])

                if len(sifted_embs) > 0:
                    sifted_embs = np.concatenate(sifted_embs)
                    embeddings.append(sifted_embs)
                    eid2sid.extend(sifted_sample_ids)
                else:
                    continue

            embeddings = np.concatenate(embeddings, axis = 0)
        
        torch.cuda.empty_cache()
        return embeddings, eid2sid
    
    
    # ============================================
    # === Read topic =============================
    topics = {}
    topic_file = f"component3_retriever/input_data/{args.dataset_name}/baselines/{args.dataset_subsec}/original.jsonl"
    
    if args.expansion_info == "cur_topic":
        with open (topic_file, 'r') as file:
            for line in file:
                data = json.loads(line.strip())
                if args.dataset_name == "TopiOCQA":
                    topics[data['id']] = data["title"]
                    # topics[data['id']] = data["title"].split('[SEP]')[0]
                elif args.dataset_name == "INSCIT":
                    topics[data['id']] = data["topic"]
    
    elif args.expansion_info == "prev_topics":
        conversation_data = {}
        with open(topic_file, 'r') as in_file:
            for line in in_file:
                sample = json.loads(line)
                conversation_data[sample['id']] = sample
        
        for q_id, data in conversation_data.items():
            conv_id = data["conv_id"]
            turn_id = data["turn_id"]
            if turn_id == 1:
                topics[data['id']] = ''
            else:
                topics_list = []
                for tid in range(1, turn_id):
                    tit = conversation_data[f"{conv_id}_{tid}"]["title"].split('[SEP]')[0]
                    if tit not in topics_list:
                        topics_list.append(tit)
                
                topics[data['id']] = ' [SEP] '.join(topics_list)
    

    # == Get queries' embedding ==================
    print("Query embedding ...")
    with open(rewriting_file, "r") as f:
        data = f.readlines()
    
    query_encoding_dataset, response_encoding_dataset = [], []
    topic_encoding_dataset = []
    n_query_candidate, n_response_candidate = 0, 0
    
    for query_idx, line in enumerate(data):
        
        # if query_idx == 3:
        #     break
        
        record = json.loads(line)
        sample_id = record['sample_id']
        if args.include_query:
            query_list = record[args.eval_field_name]
            if isinstance(query_list, str):
                query_list = [query_list]
            n_query_candidate = len(query_list)   # all line's query_list has the same length
            for query in query_list:
                query_encoding_dataset.append([sample_id, query])
    
        if args.include_response:
            response_list = record['predicted_response']
            if isinstance(response_list, str):
                response_list = [response_list]
            n_response_candidate = len(response_list)
            for response in response_list:
                response_encoding_dataset.append([sample_id, response])
    
        # Create topic dataset
        topic = topics[sample_id]
        topic_encoding_dataset.append([sample_id, topic])
        
    
    has_qrel_label_sample_ids = get_has_qrel_label_sample_ids(qrel_file_path)
    if args.include_query:
        query_test_loader = DataLoader(query_encoding_dataset, batch_size = 1, shuffle=False, collate_fn=query_encoding_collate_fn)
        query_embeddings, query_eid2sid = forward_pass(query_test_loader, encoder, has_qrel_label_sample_ids)
                
    if args.include_response:
        response_test_loader = DataLoader(response_encoding_dataset, batch_size = 1, shuffle=False, collate_fn=response_encoding_collate_fn)
        response_embeddings, response_eid2sid = forward_pass(response_test_loader, encoder, has_qrel_label_sample_ids)

    embeddings, eid2sid = return_final_ebmbedding(
        query_embeddings,
        response_embeddings,
        n_query_candidate,
        n_response_candidate,
        query_eid2sid,
        response_eid2sid
    )
    
    # === Merge 
    if args.expansion_info in ["prev_topics", "cur_topic"]:
        topic_test_loader = DataLoader(topic_encoding_dataset, batch_size=1, shuffle=False, collate_fn=topic_encoding_collate_fn)
        topic_embeddings, topic_eid2sid = forward_pass(topic_test_loader, encoder, has_qrel_label_sample_ids)
        with_topic_embeddings = (embeddings + topic_embeddings) / 2
        

    # === Retrieve base on queries' embedding ====
    print("Retrieving ...")
    encoder = AnceQueryEncoder(args.encoder, device=args.device)
    searcher = FaissSearcher(index_dir, encoder)    
    if args.expansion_info in ["prev_topics", "cur_topic"]:
        hits = searcher.batch_search(with_topic_embeddings, eid2sid, k=args.top_k, threads=20)
    else:
        hits = searcher.batch_search(embeddings, eid2sid, k=args.top_k, threads=20)
    
    # === Write to output file ===================
    print("Writing to output file ...")
    os.makedirs(args.results_base_path, exist_ok=True)
    os.makedirs(f"{args.results_base_path}/{args.dataset_name}", exist_ok=True)
    if args.expansion_info in ["prev_topics", "cur_topic"]:
        output_res_file = f"{args.results_base_path}/{args.dataset_name}/{args.expansion_info}+LLM4CS_{args.retriever_model}_results.trec"
    else:
        output_res_file = f"{args.results_base_path}/{args.dataset_name}/LLM4CS_{args.retriever_model}_results.trec"
    
    with open(output_res_file, "w") as f:
        for qid in eid2sid:
            for i, item in enumerate(hits[qid]):
                result_line = f"{qid} Q0 {item.docid[3:]} {i+1} {item.score} {args.retriever_model}"
                f.write(result_line)
                f.write('\n')
    
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever_model", type=str, default="ance", choices=["bm25", "ance"])
    parser.add_argument("--index_dir_base_path", type=str, default="corpus")
    parser.add_argument("--results_base_path", type=str, default="component3_retriever/output_results")
    parser.add_argument("--dataset_name", type=str, default="TopiOCQA", choices=["TopiOCQA", "INSCIT"])
    parser.add_argument("--dataset_subsec", type=str, default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--encoder", type=str, default="castorini/ance-msmarco-passage")
    
    parser.add_argument("--aggregation_method", type=str, default="mean", choices=["sc", "mean", "maxprob"])
    parser.add_argument("--eval_field_name", type=str, default="predicted_rewrite", choices=["predicted_rewrite"])
    parser.add_argument("--max_query_length", type=int, default=64, help="Max single query length")
    parser.add_argument("--expansion_info", default="cur_topic", choices=["no", "cur_topic", "prev_topics"])
    
    parser.add_argument("--include_query", action="store_false")
    parser.add_argument("--include_response", action="store_false")
    parser.add_argument("--top_k", type=int, default="100")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    print("Available GPUs:", torch.cuda.device_count())
    args.device = 'cuda:0'
    
    set_seed(args)
    llm4cs_retriever(args)
    
    
    # python component1_query_rewriting/LLM4CS/2_eval_dense_retrieval.py





### == Get retrieval result by query embedding
### == Ref: https://github.com/castorini/pyserini/blob/master/docs/experiments-ance.md
# python -m pyserini.search.faiss \
#   --index msmarco-v1-doc.ance-maxp \
#   --topics msmarco-doc-dev \
#   --encoded-queries ance_maxp-msmarco-doc-dev \
#   --output runs/run.msmarco-doc.passage.ance-maxp.txt \
#   --output-format msmarco \
#   --batch-size 36 --threads 12 \
#   --hits 1000 --max-passage --max-passage-hits 100
    