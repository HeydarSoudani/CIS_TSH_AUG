#!/usr/bin/env python3

### ==============================================================================
# Ref: https://github.com/fengranMark/HAConvDR/blob/main/bm25/bm25_topiocqa.py
# Ref: https://github.com/castorini/pyserini/blob/master/scripts/ance/encode_queries_msmarco_passage.py
### ==============================================================================

import torch
import random
import numpy as np
import argparse, logging, os, json
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

def pyserini_retriever(args):
    
    print("Preprocessing files ...")
    index_dir = f"{args.index_dir_base_path}/{args.dataset_name}/{args.retriever_model}_index"
    
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
    
    elif args.expansion_info == "gen_topic":
        topic_file = "processed_datasets/TopiOCQA/topic_generation.json"
        with open (topic_file, 'r') as file:
            for line in file:
                data = json.loads(line.strip())
                topics[data['query_id']] = data["output"]
      
    elif args.expansion_info == "gen_shift_topic":
        topic_file = "processed_datasets/TopiOCQA/cot_topic_gen_1.json"
        with open (topic_file, 'r') as file:
            for line in file:
                data = json.loads(line.strip())
                
                if "topic" in data["output"]:
                    topics[data['query_id']] = data["output"]["topic"]
                else:
                    topics[data['query_id']] = ""
    
    elif args.expansion_info == "gen_topic_100p_detector":
        # topic_file = "processed_datasets/TopiOCQA/topic_gen_100p_shift_detector_2.json"
        topic_file = "processed_datasets/TopiOCQA/topic_gen_100p_shift_detector_no_topic_2.json"
        with open (topic_file, 'r') as file:
            for line in file:
                data = json.loads(line.strip())
                
                if "topic" in data["output"]:
                    topics[data['query_id']] = data["output"]["topic"]
                else:
                    topics[data['query_id']] = ""
    
    
    # ========================================    
    # === Read nuggets =======================
    nuggets = {}
    # nuggets_file = "processed_datasets/TopiOCQA/dev_nuggets_2.json"
    nuggets_file = "processed_datasets/TopiOCQA/dev_nuggets_query_hist_1.json"
      
    if args.expansion_info == "rand_his_nug":
        # == Read all data =====
        conversation_data = {}
        with open (nuggets_file, 'r') as file:
            for line in file:
                sample = json.loads(line.strip())
                conversation_data[sample['query_id']] = sample
        
        for q_id, data in conversation_data.items():
            conv_id = q_id.split('_')[0]
            turn_id = int(q_id.split('_')[1])
            
            if turn_id == 1:
                nuggets[q_id] = ""
            else:
                nuggets_list = []
                for tid in range(1, turn_id):
                    nuggets_list.extend(conversation_data[f"{conv_id}_{tid}"]["nuggets"]) 
                   
                if args.nugget_num < len(nuggets_list):
                    nuggets[q_id] = ' [SEP] '.join(random.sample(nuggets_list, args.nugget_num))
                else:
                    nuggets[q_id] = ' [SEP] '.join(nuggets_list)
    
    elif args.expansion_info == "comb_his_cur_nug":
        conversation_data = {}
        with open (nuggets_file, 'r') as file:
            for line in file:
                sample = json.loads(line.strip())
                conversation_data[sample['query_id']] = sample
        
        for q_id, data in conversation_data.items():
            conv_id = q_id.split('_')[0]
            turn_id = int(q_id.split('_')[1])
            cur_nuggets = data["nuggets"]
            num_each_part = int(args.nugget_num/2)
            
            
            if turn_id == 1:
                if len(cur_nuggets) > args.nugget_num:
                    nuggets[q_id] = ' [SEP] '.join(random.sample(cur_nuggets, args.nugget_num))
                else:
                    nuggets[q_id] = ' [SEP] '.join(cur_nuggets)
                
            else:
                nuggets_list = []
                
                if len(cur_nuggets) > num_each_part:
                    nuggets_list.extend(random.sample(cur_nuggets, num_each_part))
                else:
                    num_each_part = args.nugget_num
                
                for tid in range(1, turn_id):
                    nuggets_list.extend(conversation_data[f"{conv_id}_{tid}"]["nuggets"]) 
                   
                if num_each_part < len(nuggets_list):
                    nuggets[q_id] = ' [SEP] '.join(random.sample(nuggets_list, num_each_part))
                else:
                    nuggets[q_id] = ' [SEP] '.join(nuggets_list)
    
    elif args.expansion_info == "same_top_nug":
        pass
    
    elif args.expansion_info in ["cur_turn_nug", "nug_v2"]:
        with open (nuggets_file, 'r') as file:
            for line in file:
                data = json.loads(line.strip())
                if args.nugget_num < len(data["nuggets"]):
                    nuggets[data['query_id']] = ' [SEP] '.join(random.sample(data["nuggets"], args.nugget_num))
                else:
                    nuggets[data['query_id']] = ' [SEP] '.join(data["nuggets"])
                    
    
    # ========================================
    # === Read query file ====================
    queries = {}
    if args.query_format == "t5_rewritten":
        args.query_file = f"component3_retriever/input_data/{args.dataset_name}/T5QR/t5_rewrite.json"
        with open(args.query_file, 'r') as file:
            data = json.load(file)
        
        for item in data:
            query = item["t5_rewrite"]
            if args.expansion_info in ["prev_topics", "cur_topic", "gen_topic", "gen_shift_topic", "gen_topic_100p_detector"]:
                queries[item['sample_id']] = topics[item['sample_id']] + ' [SEP] ' + query
            else:
                queries[item['sample_id']] = query
    
    elif args.query_format == "ConvGQR_rewritten":
        query_oracle_path = f"component3_retriever/input_data/{args.dataset_name}/ConvGQR/convgqr_rewrite_oracle_prefix_v2.json"
        query_expand_path = f"component3_retriever/input_data/{args.dataset_name}/ConvGQR/convgqr_rewrite_answer_prefix_v2.json"
        
        query_oracle_data = []
        query_expand_data = []
        with open(query_oracle_path, 'r') as file:
            for line in file:
                query_oracle_data.append(json.loads(line.strip()))
        with open(query_expand_path, 'r') as file:
            for line in file:
                query_expand_data.append(json.loads(line.strip()))
        
        for i, oracle_sample in enumerate(query_oracle_data):
            
            if args.expansion_info in ["prev_topics", "cur_topic"]:
                if args.query_type == "raw":
                    queries[oracle_sample['id']] = topics[item['sample_id']] + ' [SEP] ' + oracle_sample["query"]
                elif args.query_type == "rewrite":
                    queries[oracle_sample['id']] = topics[item['sample_id']] + ' [SEP] ' + oracle_sample['rewrite']

                elif args.query_type == "decode":
                    query = oracle_sample['oracle_utt_text']
                    if args.eval_type == "answer":
                        queries[oracle_sample['sample_id']] = topics[item['sample_id']] + ' [SEP] ' + query_expand_data[i]['answer_utt_text']
                    elif args.eval_type == "oracle+answer":
                        queries[oracle_sample['sample_id']] = topics[item['sample_id']] + ' [SEP] ' + query + ' ' + query_expand_data[i]['answer_utt_text']
                
            else:
                if args.query_type == "raw":
                    queries[oracle_sample['id']] = oracle_sample["query"]
                elif args.query_type == "rewrite":
                    queries[oracle_sample['id']] = oracle_sample['rewrite']

                elif args.query_type == "decode":
                    query = oracle_sample['oracle_utt_text']
                    if args.eval_type == "answer":
                        queries[oracle_sample['sample_id']] = query_expand_data[i]['answer_utt_text']
                    elif args.eval_type == "oracle+answer":
                        queries[oracle_sample['sample_id']] = query + ' ' + query_expand_data[i]['answer_utt_text']
    
    elif args.query_format == "top_qr":
        # args.query_file = "processed_datasets/TopiOCQA/topic_aware_query_rewriting_2.json"
        args.query_file = "processed_datasets/TopiOCQA/gen_topic_aware_query_rewriting_2.json"
        with open (args.query_file, 'r') as file:
            for line in file:
                item = json.loads(line.strip())
                if len(item["rewritten"]) == 0:
                    queries[item['query_id']] = item["question"]
                else:
                    queries[item['query_id']] = random.sample(item["rewritten"], 1)[0]
    
    else:
        args.query_file = f"component3_retriever/input_data/{args.dataset_name}/baselines/{args.dataset_subsec}/{args.query_format}.jsonl"
        with open (args.query_file, 'r') as file:
            for line in file:
                item = json.loads(line.strip())
                
                if args.expansion_info in ["prev_topics", "cur_topic", "gen_topic", "gen_shift_topic", "gen_topic_100p_detector"]:
                    queries[item['id']] = topics[item['id']] + ' [SEP] ' + item["query"]
                elif args.expansion_info in ["rand_his_nug", "same_top_nug", "cur_turn_nug", "comb_his_cur_nug", "nug_v2"]:
                    queries[item['id']] = nuggets[item['id']] + ' [SEP] ' + item["query"]
                elif args.expansion_info == "no":
                    queries[item['id']] = item["query"]
    
    
    # = Select a subset of queries ===========
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
    elif args.retriever_model == "ance":
        encoder = AnceQueryEncoder(args.query_encoder, device=args.device)
        searcher = FaissSearcher(index_dir, encoder)
        hits = searcher.batch_search(query_list, qid_list, k=args.top_k, threads=20)
    
    
    # === Write to output file ===============
    print("Writing to output file ...")
    os.makedirs(args.results_base_path, exist_ok=True)
    os.makedirs(f"{args.results_base_path}/{args.dataset_name}", exist_ok=True)
    
    if args.expansion_info in ["prev_topics", "cur_topic", "gen_topic", "gen_shift_topic", "gen_topic_100p_detector"]:
        output_res_file = f"{args.results_base_path}/{args.dataset_name}/{args.expansion_info}+{args.query_format}_{args.retriever_model}_results.trec"
    elif args.expansion_info in ["rand_his_nug", "same_top_nug", "cur_turn_nug", "comb_his_cur_nug", "nug_v2"]:
        output_res_file = f"{args.results_base_path}/{args.dataset_name}/{args.expansion_info}_{args.nugget_num}+{args.query_format}_{args.retriever_model}_results.trec"
    else:
        output_res_file = f"{args.results_base_path}/{args.dataset_name}/{args.query_format}_{args.retriever_model}_results.trec"
    
    with open(output_res_file, "w") as f:
        for qid in qid_list:
            for i, item in enumerate(hits[qid]):
                result_line = f"{qid} Q0 {item.docid[3:]} {i+1} {item.score} {args.retriever_model}"
                f.write(result_line)
                f.write('\n')
                
    print("Done!")
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever_model", type=str, default="ance", choices=["bm25", "ance"])
    parser.add_argument("--query_encoder", type=str, default="castorini/ance-msmarco-passage")
    parser.add_argument("--index_dir_base_path", type=str, default="corpus")
    parser.add_argument("--results_base_path", type=str, default="component3_retriever/output_results")
    parser.add_argument("--dataset_name", type=str, default="TopiOCQA", choices=["TopiOCQA", "INSCIT"])
    parser.add_argument("--dataset_subsec", type=str, default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--query_format", type=str, default='t5_rewritten', choices=[
        'original', 'human_rewritten', 'all_history', 'same_topic',
        't5_rewritten', 'ConvGQR_rewritten',
        "top_qr"
    ])
    parser.add_argument("--expansion_info", default="cur_topic", choices=[
        "no", "cur_topic", "prev_topics",
        "rand_his_nug", "same_top_nug", "cur_turn_nug", "comb_his_cur_nug", "nug_v2",
        "gen_topic", "gen_shift_topic", "gen_topic_100p_detector"
    ])
    parser.add_argument("--nugget_num", type=int, default=5)
    
    parser.add_argument("--query_type", type=str, default="decode", help="for ConvGQR")
    parser.add_argument("--eval_type", type=str, default="oracle+answer", help="for ConvGQR")
    
    parser.add_argument("--bm25_k1", type=float, default="0.9")
    parser.add_argument("--bm25_b", type=float, default="0.4")
    parser.add_argument("--top_k", type=int, default="100")
    parser.add_argument("--rel_threshold", type=int, default="1")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    print("Available GPUs:", torch.cuda.device_count())
    args.device = 'cuda:0'
    
    set_seed(args.seed)
    pyserini_retriever(args)
    
    # python component3_retriever/1_topiocqa_inscit_retriever.py
    