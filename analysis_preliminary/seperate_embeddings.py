#!/usr/bin/env python3

import torch
import random
import numpy as np
from tqdm import tqdm
import argparse, logging, os, json
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer

from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher
from pyserini.dsearch import AnceQueryEncoder


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

def get_has_qrel_label_sample_ids(qrel_file):
    with open(qrel_file, 'r') as f:
        qrel_data = f.readlines()
    qids = set()
    for line in qrel_data:
        line = line.strip().split("\t")
        if len(line) == 1:
            line = line[0].strip().split(' ')
        qid = line[0]
        qids.add(qid)
    return qids

def pyserini_retriever_seperate_emb(args):
    print("Preprocessing files ...")
    index_dir = f"{args.index_dir_base_path}/{args.dataset_name}/{args.retriever_model}_index"
    qrel_file_path = f"processed_datasets/{args.dataset_name}/test_gold_qrels.trec"
    
    # ========================================
    # === Read topic =========================
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
    

    # ========================================
    # === Read doc ===========================
    docs = {}
    doc_file = "processed_datasets/TopiOCQA/ir_all_history_dev.json"
    with open (doc_file, 'r') as file:
        data = json.load(file)
    
    for item in data:
        query_id = f"{str(item['conv_id'])}_{str(item['turn_id'])}"
        doc = item["positive_ctxs"][0]["text"]
        docs[query_id] = doc


    # ========================================
    # === Read query file ====================
    queries = {}
    if args.query_format == "t5_rewritten":
        args.query_file = f"component3_retriever/input_data/{args.dataset_name}/T5QR/t5_rewrite.json"
        with open(args.query_file, 'r') as file:
            data = json.load(file)
        
        for item in data:
            query = item["t5_rewrite"]
            queries[item['sample_id']] = query

    else:
        args.query_file = f"component3_retriever/input_data/{args.dataset_name}/baselines/{args.dataset_subsec}/{args.query_format}.jsonl"
        with open (args.query_file, 'r') as file:
            for line in file:
                item = json.loads(line.strip())
                queries[item['id']] = item["query"]


    # = Select a subset of queries ===========
    if subset_percentage != 1.0:
        subset_size = int(len(queries) * subset_percentage)
        subset_keys = random.sample(list(queries.keys()), subset_size)
        subset_queries = {key: queries[key] for key in subset_keys}
    else:
        subset_queries = queries

    qid_list = list(subset_queries.keys())
    # query_list = [subset_queries[qid] for qid in qid_list]
    # topic_list = [topics[qid] for qid in qid_list]
    
    print(f"Query_id: {qid_list[1]}")
    print(f"Query: {queries[qid_list[0]]}")
    print(f"Topic: {topics[qid_list[0]]}")
    print(f"Doc: {docs[qid_list[0]]} \n")
    
    
    
    
    # = Get queries' embedding ===============
    print("Query embedding ...")
    encoder = AnceQueryEncoder(args.encoder, device=args.device)
    tokenizer = RobertaTokenizer.from_pretrained(args.encoder, do_lower_case=True)
    
    # Defining functions =====================
    def doc_encoding_collate_fn(batch):
        bt_sample_ids, bt_src_query = list(zip(*batch)) # unzip
        bt_src_query_encoding = tokenizer(bt_src_query, 
                                            padding="longest", 
                                            max_length=args.max_doc_length, 
                                            truncation=True, 
                                            return_tensors="pt")
        
        bt_q_input_ids, bt_q_attention_mask = bt_src_query_encoding.input_ids, bt_src_query_encoding.attention_mask
        
        return {
                "bt_input": bt_src_query,
                "bt_sample_ids": bt_sample_ids, 
                "bt_input_ids":bt_q_input_ids, 
                "bt_attention_mask":bt_q_attention_mask}
    
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
    
    def topic_encoding_collate_fn(batch):
        bt_sample_ids, bt_src_doc = list(zip(*batch))
        bt_src_doc_encoding = tokenizer(bt_src_doc, 
                                          padding="longest", 
                                          max_length=args.max_title_length, 
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
    
    def return_final_ebmbedding(
        query_embeddings,
        topic_embeddings,
        doc_embeddings,
        query_eid2sid,
        topic_eid2sid,
        doc_eid2sid
    ):
        # filter out duplicate sample_ids
        eid2sid = query_eid2sid if query_eid2sid else topic_eid2sid
        new_eid2sid = []
        eid2sid_set = set()
        for x in eid2sid:
            if x not in eid2sid_set:
                new_eid2sid.append(x)
                eid2sid_set.add(x)
        eid2sid = new_eid2sid
        
        torch.cuda.empty_cache()
        # return (query_embeddings + topic_embeddings) / 2, eid2sid
        return (doc_embeddings + topic_embeddings) / 2, eid2sid  
    # =======================================
    
    query_encoding_dataset, topic_encoding_dataset, doc_encoding_dataset = [], [], []
    
    for query_id in qid_list:
        query = queries[query_id]
        topic = topics[query_id]
        doc = docs[query_id]
        query_encoding_dataset.append([query_id, query])
        topic_encoding_dataset.append([query_id, topic])
        doc_encoding_dataset.append([query_id, doc])
    
    has_qrel_label_sample_ids = get_has_qrel_label_sample_ids(qrel_file_path)
    
    query_test_loader = DataLoader(query_encoding_dataset, batch_size=1, shuffle=False, collate_fn=query_encoding_collate_fn)
    query_embeddings, query_eid2sid = forward_pass(query_test_loader, encoder, has_qrel_label_sample_ids)
    topic_test_loader = DataLoader(topic_encoding_dataset, batch_size=1, shuffle=False, collate_fn=topic_encoding_collate_fn)
    topic_embeddings, topic_eid2sid = forward_pass(topic_test_loader, encoder, has_qrel_label_sample_ids)
    doc_test_loader = DataLoader(doc_encoding_dataset, batch_size=1, shuffle=False, collate_fn=doc_encoding_collate_fn)
    doc_embeddings, doc_eid2sid = forward_pass(doc_test_loader, encoder, has_qrel_label_sample_ids)
    
    embeddings, eid2sid = return_final_ebmbedding(
        query_embeddings,
        topic_embeddings,
        doc_embeddings,
        query_eid2sid,
        topic_eid2sid,
        doc_eid2sid
    )
    
    # === Retrieve base on queries' embedding ====
    print("Retrieving ...")
    encoder = AnceQueryEncoder(args.encoder, device=args.device)
    searcher = FaissSearcher(index_dir, encoder)
    hits = searcher.batch_search(embeddings, eid2sid, k=args.top_k, threads=20)
    
    # === Write to output file ===================
    print("Writing to output file ...")
    os.makedirs(args.results_base_path, exist_ok=True)
    os.makedirs(f"{args.results_base_path}/{args.dataset_name}", exist_ok=True)
    
    # if args.expansion_info in ["prev_topics", "cur_topic", "gen_topic"]:
    #     output_res_file = f"{args.results_base_path}/{args.dataset_name}/{args.expansion_info}+{args.query_format}_sep_embedding_{args.retriever_model}_results.trec"
    # else:
        # output_res_file = f"{args.results_base_path}/{args.dataset_name}/{args.query_format}_doc_topic_sep_embedding_{args.retriever_model}_results.trec"
    output_res_file = f"{args.results_base_path}/{args.dataset_name}/doc_topic_embedding_{args.retriever_model}_results.trec"
    
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
    parser.add_argument("--encoder", type=str, default="castorini/ance-msmarco-passage")
    parser.add_argument("--index_dir_base_path", type=str, default="corpus")
    parser.add_argument("--results_base_path", type=str, default="component3_retriever/output_results")
    parser.add_argument("--dataset_name", type=str, default="TopiOCQA", choices=["TopiOCQA", "INSCIT"])
    parser.add_argument("--dataset_subsec", type=str, default="dev", choices=["train", "dev", "test"])
    
    parser.add_argument("--query_format", type=str, default='original', choices=[
        'original', 'human_rewritten', 'all_history', 'same_topic',
        't5_rewritten'
    ])
    parser.add_argument("--expansion_info", default="cur_topic", choices=[
        "no", "cur_topic", "prev_topics", "gen_topic"
    ])
    parser.add_argument("--max_query_length", type=int, default=64, help="Max single query length")
    parser.add_argument("--max_doc_length", type=int, default=384, help="Max single doc length")
    parser.add_argument("--max_title_length", type=int, default=16, help="Max single doc length")
    
    parser.add_argument("--bm25_k1", type=float, default="0.9")
    parser.add_argument("--bm25_b", type=float, default="0.4")
    parser.add_argument("--top_k", type=int, default="100")
    parser.add_argument("--rel_threshold", type=int, default="1")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    print("Available GPUs:", torch.cuda.device_count())
    args.device = 'cuda:0'
    
    set_seed(args.seed)
    pyserini_retriever_seperate_emb(args)
    
    
    # python analysis_preliminary/seperate_embeddings.py