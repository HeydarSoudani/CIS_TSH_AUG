import json
import torch
import random
import numpy as np
from tqdm import tqdm, trange

import torch.nn as nn
from transformers import AdamW
from torch.utils.data import Dataset


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def get_optimizer(args, model: nn.Module, weight_decay: float = 0.0, ) -> torch.optim.Optimizer:
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    return optimizer


def padding_seq_to_same_length(input_ids, max_pad_length, pad_token = 0):
    padding_length = max_pad_length - len(input_ids)
    padding_ids = [pad_token] * padding_length
    attention_mask = []

    if padding_length <= 0:
        attention_mask = [1] * max_pad_length
        input_ids = input_ids[:max_pad_length]
    else:
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        input_ids = input_ids + padding_ids
            
    assert len(input_ids) == max_pad_length
    assert len(attention_mask) == max_pad_length
  
    return input_ids, attention_mask


class Retrieval_topiocqa_new(Dataset):
    def __init__(self, args, tokenizer, filename, collection=None):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()

        n = len(data)

        for i in tqdm(trange(n)):
            record = json.loads(data[i])
            # <CUR> oracle_query <CTX> cq_{k-1} <SEP> ... cq_{1} (<BOS>)
            sample_id = record['sample_id']
            flat_qp_concat = []
            ctx_utts_text = record['cur_utt_text'].strip().split(" [SEP] ") # [q1, a1, q2, a2, ...]
            cur_utt_text = ctx_utts_text[-1] 
            ctx_utts_text = ctx_utts_text[:-1]
            last_response = record['last_response']
            pos_docs_text = record['pos_docs'][0]
            pos_docs_pids = record['pos_docs_pids'][0]
            if args.is_train:
                bm25_hard_neg_docs = record['bm25_hard_neg_docs'][0]
                pseudo_prepos_docs = record["pseudo_prepos_docs"]
                prepos_neg_docs = record["prepos_neg_docs"]
            rel_label = record['rel_label']

            cur_utt = tokenizer.encode(cur_utt_text, add_special_tokens = True, max_length = args.max_query_length)
            flat_qp_concat.extend(cur_utt)
            if args.use_PRL and 1 in rel_label:
                for index in range(len(rel_label) - 1, -1, -1):
                    if rel_label[index] == 1:
                        if not args.is_PRF:
                            rel_turn_passage = json.loads(data[i - (len(rel_label) - index)])['pos_docs'][0]
                            rel_turn_passage = tokenizer.encode(rel_turn_passage, add_special_tokens = True, max_length = args.max_doc_length)
                        else:
                            rel_turn_passage = json.loads(data[i - (len(rel_label) - index)])['PRF_pos_docs'][0]
                            rel_turn_passage = tokenizer.encode(rel_turn_passage, add_special_tokens = True, max_length = args.max_doc_length)
                        rel_turn_query = json.loads(data[i - (len(rel_label) - index)])['cur_utt_text'].strip().split(" [SEP] ")[-1]
                        rel_turn_query = tokenizer.encode(rel_turn_query, add_special_tokens = True, max_length = args.max_query_length)
                        if len(flat_qp_concat) + len(rel_turn_passage) > args.max_concat_length:
                            flat_qp_concat += rel_turn_passage[:args.max_concat_length - len(flat_qp_concat) - 1] + [rel_turn_passage[-1]]    # must ended with [SEP]
                            break
                        else:
                            flat_qp_concat.extend(rel_turn_passage)
                        if len(flat_qp_concat) + len(rel_turn_query) > args.max_concat_length:
                            flat_qp_concat += rel_turn_query[:args.max_concat_length - len(flat_qp_concat) - 1] + [rel_turn_query[-1]]    # must ended with [SEP]
                            break
                        else:
                            flat_qp_concat.extend(rel_turn_query)
            else: # including no PRL, all zero in rel_label and first-turn
                if len(last_response) > 0:
                    last_response_utt = tokenizer.encode(last_response, add_special_tokens = True, max_length = args.max_doc_length)
                    flat_qp_concat.extend(last_response_utt)
            
            for j in range(len(ctx_utts_text) - 1, -1, -1):
                if j % 2 == 1: # answer
                    max_length = args.max_response_length
                elif j % 2 == 0: # query
                    max_length = args.max_query_length
                utt = tokenizer.encode(ctx_utts_text[j], add_special_tokens=True, max_length=max_length, truncation=True) # not remove [CLS]
                if len(flat_qp_concat) + len(utt) > args.max_concat_length:
                    flat_qp_concat += utt[:args.max_concat_length - len(flat_qp_concat) - 1] + [utt[-1]]    # must ended with [SEP]
                    break
                else:
                    flat_qp_concat.extend(utt)
                    
            flat_qp_concat, flat_qp_concat_mask = padding_seq_to_same_length(flat_qp_concat, max_pad_length = args.max_concat_length)

            # doc 
            pos_docs, neg_docs, pos_docs_mask, neg_docs_mask, pseudo_prepos_docs, pseudo_prepos_docs_mask, prepos_neg_docs, prepos_neg_docs_mask = [], [], [], [], [], [], [], []
            if args.is_train:
                pos_docs.extend(tokenizer.encode(pos_docs_text, add_special_tokens=True, max_length=args.max_doc_length, truncation=True))
                neg_docs.extend(tokenizer.encode(bm25_hard_neg_docs, add_special_tokens=True, max_length=args.max_doc_length, truncation=True))
                pos_docs, pos_docs_mask = padding_seq_to_same_length(pos_docs, max_pad_length = args.max_doc_length)
                neg_docs, neg_docs_mask = padding_seq_to_same_length(neg_docs, max_pad_length = args.max_doc_length)
                if len(pseudo_prepos_docs) > 0:
                    pseudo_prepos_docs.extend(tokenizer.encode(random.choice(pseudo_prepos_docs), add_special_tokens=True, max_length=args.max_doc_length, truncation=True))
                    pseudo_prepos_docs, pseudo_prepos_docs_mask = padding_seq_to_same_length(pseudo_prepos_docs, max_pad_length = args.max_doc_length)
                if len(prepos_neg_docs) > 0:
                    prepos_neg_docs.extend(tokenizer.encode(random.choice(prepos_neg_docs), add_special_tokens=True, max_length=args.max_doc_length, truncation=True))
                    prepos_neg_docs, prepos_neg_docs_mask = padding_seq_to_same_length(prepos_neg_docss, max_pad_length = args.max_doc_length)
            
            self.examples.append([sample_id, flat_qp_concat, flat_qp_concat_mask, pos_docs, pos_docs_mask, neg_docs, neg_docs_mask, pseudo_prepos_docs, pseudo_prepos_docs_mask, prepos_neg_docs, prepos_neg_docs_mask])
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             "bt_conv_qp": [],
                             "bt_conv_qp_mask": [],
                             "bt_pos_docs":[],
                             "bt_pos_docs_mask":[],
                             "bt_neg_docs":[],
                             "bt_neg_docs_mask":[],
                             "bt_pseudo_prepos_docs":[],
                             "bt_pseudo_prepos_docs_mask":[],
                             "bt_prepos_neg_docs":[],
                             "bt_prepos_neg_docs_mask":[],
                            }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_conv_qp"].append(example[1])
                collated_dict["bt_conv_qp_mask"].append(example[2])
                collated_dict["bt_pos_docs"].append(example[3])
                collated_dict["bt_pos_docs_mask"].append(example[4])
                collated_dict["bt_neg_docs"].append(example[5])
                collated_dict["bt_neg_docs_mask"].append(example[6])
                collated_dict["bt_pseudo_prepos_docs"].append(example[7])
                collated_dict["bt_pseudo_prepos_docs_mask"].append(example[8])
                collated_dict["bt_prepos_neg_docs"].append(example[9])
                collated_dict["bt_prepos_neg_docs_mask"].append(example[10])

            not_need_to_tensor_keys = set(["bt_sample_ids", "bt_cur_utt_text", "bt_oracle_utt_text"])

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
            return collated_dict

        return collate_fn
