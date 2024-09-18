import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from pyserini.dsearch import AnceQueryEncoder

def get_queries_docs(conversation_num):
    
    his_file_path = "processed_datasets/TopiOCQA/ir_all_history_dev.json"
    rw_file_path = "processed_datasets/TopiOCQA/ir_rewrite_dev.json"
    
    with open(his_file_path, 'r') as file:
        history_data = json.load(file)
    with open(rw_file_path, 'r') as file:
        rewrite_data = json.load(file)
    
    cur_queries = []
    his_queries = []
    rw_queries = []
    t_queries = []
    docs = []
    titles = []
    
    for idx, item in enumerate(history_data):
        if item["conv_id"] == conversation_num:
            
            sample_id = f"{str(item['conv_id'])}_{str(item['turn_id'])}"
            
            current_doc = item['positive_ctxs'][0]["text"]
            title = item['positive_ctxs'][0]["title"].split('[SEP]')[0]
            
            current_query = item["question"].split('[SEP]')[-1]
            his_query = item["question"]
            rw_query = rewrite_data[idx]["question"]
            t_query = f"{title} [SEP] {current_query}"
            
            cur_queries.append([sample_id, current_query])
            his_queries.append([sample_id, his_query])
            rw_queries.append([sample_id, rw_query])
            t_queries.append([sample_id, t_query])
            docs.append([sample_id, current_doc])
            titles.append([sample_id, title])
    
    return docs, his_queries

def main(args):
    # == Load Model =========
    encoder = AnceQueryEncoder(args.encoder, device=args.device)
    tokenizer = RobertaTokenizer.from_pretrained(args.encoder, do_lower_case=True)
    
    # == Get Query & Doc Emb. =====
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
    
    def forward_pass(test_loader, encoder):
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
                    # if bt_sample_ids[i] not in has_qrel_label_sample_ids:
                    #     continue
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
    
    def forward_pass_prototype(test_dataset, encoder):
        hist_embeddings = []
        curr_embeddings = []
        
        eid2sid = []
        with torch.no_grad():
            for item in tqdm(test_dataset):
                sample_id, query = item
                split_segments = query.split(' [SEP] ')
                grouped_segments = [' [SEP] '.join(split_segments[i:i+2]) for i in range(0, len(split_segments), 2)]
                
                turn_embeddings = []
                for turn in grouped_segments:
                    embs = encoder.encode(turn)
                    turn_embeddings.append(embs.reshape(1, -1))
                
                turn_embeddings = np.concatenate(turn_embeddings, axis=0)
                
                if turn_embeddings.shape[0] == 1:
                    hist_mean = np.zeros((1, 768))
                    curr_vec = turn_embeddings.reshape(1, -1)
                elif turn_embeddings.shape[0] == 2:
                    hist_mean = turn_embeddings[0].reshape(1, -1)
                    curr_vec = turn_embeddings[1].reshape(1, -1)
                else:
                    hist_mean = np.mean(turn_embeddings[:-1], axis=0).reshape(1, -1)
                    curr_vec = turn_embeddings[-1].reshape(1, -1)
                
                hist_embeddings.append(hist_mean)
                curr_embeddings.append(curr_vec)                
            
            hist_embeddings = np.concatenate(hist_embeddings, axis=0)
            curr_embeddings = np.concatenate(curr_embeddings, axis=0)
            print(hist_embeddings.shape)
            print(curr_embeddings.shape)
        
        return hist_embeddings, curr_embeddings
        

    # == Read Data ================
    conversation_num = 1
    docs_dataset, his_queries_dataset = get_queries_docs(conversation_num)

    doc_test_loader = DataLoader(docs_dataset, batch_size=1, shuffle=False, collate_fn=doc_encoding_collate_fn)
    his_query_test_loader = DataLoader(his_queries_dataset, batch_size=1, shuffle=False, collate_fn=query_encoding_collate_fn)
    doc_embeddings, doc_eid2sid = forward_pass(doc_test_loader, encoder)
    his_query_embeddings, query_eid2sid = forward_pass(his_query_test_loader, encoder)
    sep_his_query_embeddings, sep_curr_query_embeddings = forward_pass_prototype(his_queries_dataset, encoder)


    # print(doc_embeddings.shape)
    # n_samples = doc_embeddings.shape[0]

    # dot_products = np.einsum('ij,ij->i', doc_embeddings, his_query_embeddings)
    
    # dp_sep_hist = np.einsum('ij,ij->i', doc_embeddings, sep_his_query_embeddings)
    # dp_sep_curr = np.einsum('ij,ij->i', doc_embeddings, sep_curr_query_embeddings)
    
    # for i in range(1, n_samples):
    #     print(f"Turn {i} -> All-Hist: {dot_products[i]} / Sep-Hist: {dp_sep_hist[i]}/{dp_sep_curr[i]}")
        
    
 
    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, default="castorini/ance-msmarco-passage")
    parser.add_argument("--max_query_length", type=int, default=64, help="Max single query length")
    parser.add_argument("--max_doc_length", type=int, default=384, help="Max single doc length")
    parser.add_argument("--max_title_length", type=int, default=8, help="Max single doc length")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    print("Available GPUs:", torch.cuda.device_count())
    args.device = 'cuda:0'
    
    main(args)
    
    # python analysis_representation/query_doc_score.py