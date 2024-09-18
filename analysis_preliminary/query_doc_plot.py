
import umap
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
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
            title = item['positive_ctxs'][0]["title"]
            
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
    
    return cur_queries, his_queries, rw_queries, t_queries, docs, titles

def main(args):
    # == Load Model ===============
    encoder = AnceQueryEncoder(args.encoder, device=args.device)
    tokenizer = RobertaTokenizer.from_pretrained(args.encoder, do_lower_case=True)

    # == Read Data ================
    conversation_num = 1
    mse_title = 0.0
    mse_tq = 0.0
    mse_rwq = 0.0
    
    for conversation_num in tqdm(range(1, 206)):
        queries_dataset, his_queries_dataset, rw_queries_dataset, t_queries_dataset, docs_dataset, titles_dataset = get_queries_docs(conversation_num)

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
        
        def title_encoding_collate_fn(batch):
            bt_sample_ids, bt_src_query = list(zip(*batch)) # unzip
            bt_src_query_encoding = tokenizer(bt_src_query, 
                                                padding="longest", 
                                                max_length=args.max_title_length, 
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
                for batch in test_loader:
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
        
        doc_test_loader = DataLoader(docs_dataset, batch_size=1, shuffle=False, collate_fn=doc_encoding_collate_fn)
        titles_test_loader = DataLoader(titles_dataset, batch_size=1, shuffle=False, collate_fn=title_encoding_collate_fn)
        query_test_loader = DataLoader(queries_dataset, batch_size=1, shuffle=False, collate_fn=query_encoding_collate_fn)
        his_query_test_loader = DataLoader(his_queries_dataset, batch_size=1, shuffle=False, collate_fn=query_encoding_collate_fn)
        rw_query_test_loader = DataLoader(rw_queries_dataset, batch_size=1, shuffle=False, collate_fn=query_encoding_collate_fn)
        t_query_test_loader = DataLoader(t_queries_dataset, batch_size=1, shuffle=False, collate_fn=query_encoding_collate_fn)
        
        doc_embeddings, doc_eid2sid = forward_pass(doc_test_loader, encoder)
        title_embeddings, title_eid2sid = forward_pass(titles_test_loader, encoder)
        query_embeddings, query_eid2sid = forward_pass(query_test_loader, encoder)
        rw_query_embeddings, query_eid2sid = forward_pass(rw_query_test_loader, encoder)
        his_query_embeddings, query_eid2sid = forward_pass(his_query_test_loader, encoder)
        t_query_embeddings, query_eid2sid = forward_pass(t_query_test_loader, encoder)
        
        # print(doc_embeddings.shape)
        # print(title_embeddings.shape)
        # print(query_embeddings.shape)
        # print(rw_query_embeddings.shape)
        
        n_samples = doc_embeddings.shape[0]
        
        # == MSE calc ===========
        mse_per_row_title = np.mean((doc_embeddings - title_embeddings) ** 2, axis=1).reshape(-1, 1)
        total_mse_title = np.sum(mse_per_row_title)
        
        mse_per_row_tq = np.mean((doc_embeddings - t_query_embeddings) ** 2, axis=1).reshape(-1, 1)
        total_mse_tq = np.sum(mse_per_row_tq)
        
        mse_per_row_rw = np.mean((doc_embeddings - rw_query_embeddings) ** 2, axis=1).reshape(-1, 1)
        total_mse_rw = np.sum(mse_per_row_rw)
        
        mse_title += total_mse_title
        mse_tq += total_mse_tq
        mse_rwq += total_mse_rw
        
    print("Total MSE title:", mse_title/205)
    print("Total MSE TQ:", mse_tq/205)
    print("Total MSE RW:", mse_rwq/205)
    

    # == Plot ===============
    # combined_array = np.vstack([
    #     doc_embeddings,
    #     title_embeddings,
    #     query_embeddings,
    #     rw_query_embeddings,
    #     # t_query_embeddings,
        
    # ])
    # # tsne = TSNE(n_components=2, perplexity=20, random_state=42)
    # umap_model = umap.UMAP(n_components=2, random_state=42)
    
    
    
    # print('Getting tsne embedding ...')
    # # tsne_results = tsne.fit_transform(combined_array)
    # tsne_results = umap_model.fit_transform(combined_array)
    
    # tsne_array1 = tsne_results[:n_samples]
    # tsne_array2 = tsne_results[n_samples:2*n_samples]
    # tsne_array3 = tsne_results[2*n_samples:3*n_samples]
    # tsne_array4 = tsne_results[3*n_samples:]
    # # tsne_array5 = tsne_results[4*n_samples:]
    
    # print('Plotting ...')
    # plt.figure(figsize=(10, 8))
    # colors = plt.cm.rainbow(np.linspace(0, 1, n_samples))
    # for i in range(n_samples):
    #     plt.scatter(tsne_array1[i, 0], tsne_array1[i, 1], color=colors[i], marker='o', label=f'T{i+1}, D')
    #     plt.scatter(tsne_array2[i, 0], tsne_array2[i, 1], color=colors[i], marker='x', label=f'T{i+1}, T')
    #     plt.scatter(tsne_array3[i, 0], tsne_array3[i, 1], color=colors[i], marker='^', label=f'T{i+1}, cq')
    #     plt.scatter(tsne_array4[i, 0], tsne_array4[i, 1], color=colors[i], marker='*', label=f'T{i+1}, rq')
    #     # plt.scatter(tsne_array5[i, 0], tsne_array5[i, 1], color=colors[i], marker='d', label=f'T{i+1}, tq')

    # plt.title('Docs & Queries Embedding')
    # plt.xlabel('Dim 1')
    # plt.ylabel('Dim 2')
    # plt.legend(loc='upper center', ncol=n_samples, fontsize=6, bbox_to_anchor=(0.5, 1.1))
    # plt.savefig('foo.png')
    # plt.show()
    


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
    
    # python analysis_representation/query_doc_plot.py

