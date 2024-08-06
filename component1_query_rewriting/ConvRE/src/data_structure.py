
import json
import torch
import random
import logging
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, TensorDataset, IterableDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def pad_seq_ids_with_mask(input_ids,
                            max_length,
                            pad_on_left=False,
                            pad_token=0):
    padding_length = max_length - len(input_ids)
    padding_id = [pad_token] * padding_length

    attention_mask = []

    if padding_length <= 0:
        input_ids = input_ids[-max_length:]
        attention_mask = [1] * max_length
    else:
        if pad_on_left:
            input_ids = padding_id + input_ids
        else:
            attention_mask = [1] * len(input_ids) + [0] * padding_length
            input_ids = input_ids + padding_id

    assert len(input_ids) == max_length
    assert len(attention_mask) == max_length

    return input_ids, attention_mask

class ConvExample_topiocqa_rel:
    def __init__(self, sample_id,
                        conv_id,
                        turn_id, 
                        pair_query = None,
                        query_passage = None,
                        cur_query = None
                        ):
        self.sample_id = sample_id
        self.conv_id = conv_id
        self.turn_id = turn_id
        self.cur_query = cur_query
        self.pair_query = pair_query
        self.query_passage = query_passage

class ConvDataset_topiocqa_rel(Dataset):
    def __init__(self, args, query_tokenizer, filename, add_doc_info=True):
        self.examples = []
        
        with open(filename, 'r') as f:
            data = f.readlines()
        n = len(data)
        
        logger.info("Loading {} data file...".format(filename))
        for i in trange(n):
            # basic
            data[i] = json.loads(data[i])
            sample_id = data[i]['id']
            conv_id = data[i]['conv_id']
            turn_id = data[i]['turn_id']
            #topic = data[i]["topic"]
            #sub_topic = data[i]["sub_topic"]
            query = data[i]["query"] # str
            history_answer = data[i]["history_answer"]
            last_response = data[i]["last_response"]
            #answer = data[i]["answer"]
            #turn_pair_id = data[i]['turn_pair_id']
            query_pair = data[i]["query_pair"] # str

            # query
            pair_query = []
            query_passage = []
            cur_query = query_tokenizer.encode(query, add_special_tokens=True)
            pair_query.extend(cur_query)
            query_passage.extend(cur_query)
            if args.use_last_response and len(last_response) > 0:
                lp = []
                lp.append(query_tokenizer.cls_token_id)
                lp.extend(query_tokenizer.convert_tokens_to_ids(["<response>"]))
                lp.extend(query_tokenizer.convert_tokens_to_ids(query_tokenizer.tokenize(last_response)))
                lp = lp[:args.max_doc_length]
                lp.append(query_tokenizer.sep_token_id)
                #lp = query_tokenizer.encode(last_response, add_special_tokens=True)
                pair_query.extend(lp)
                query_passage.extend(lp)
            if args.use_answer and len(history_answer) > 0:
                last_answer = query_tokenizer.encode(history_answer[-1], add_special_tokens=True)
                pair_query.extend(last_answer)
            if len(query_pair) > 0:
                turn_query = query_tokenizer.encode(query_pair, add_special_tokens=True)
                pair_query.extend(turn_query)

            self.examples.append(ConvExample_topiocqa_rel(sample_id,
                                            conv_id,
                                            turn_id, 
                                            pair_query,
                                            query_passage,
                                            cur_query,
                                            )) 

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    @staticmethod
    def get_collate_fn(args, add_doc_info:bool):

        def collate_fn(batch: list):
            collated_dict = {
                "bt_sample_id": [],
                "bt_conv_id": [],
                "bt_turn_id": [],
                "bt_query":[],
                "bt_query_mask":[],
                "bt_pair_query":[],
                "bt_pair_query_mask":[],
                "bt_query_passage":[],
                "bt_query_passage_mask":[],
            }
            
            bt_sample_id = [] 
            bt_conv_id = [] 
            bt_turn_id = [] 
            #bt_query = []
            #bt_query_mask = []
            bt_pair_query = []
            bt_pair_query_mask = []
            bt_query_passage = []
            bt_query_passage_mask = []

            for example in batch:
                # padding
                pair_query, pair_query_mask = pad_seq_ids_with_mask(example.pair_query, max_length = args.max_concat_length)
                query_passage, query_passage_mask = pad_seq_ids_with_mask(example.query_passage, max_length = args.max_concat_length)
                bt_sample_id.append(example.sample_id)
                bt_conv_id.append(example.conv_id)
                bt_turn_id.append(example.turn_id)
                bt_pair_query.append(pair_query)
                bt_pair_query_mask.append(pair_query_mask)
                bt_query_passage.append(query_passage)
                bt_query_passage_mask.append(query_passage_mask)
                

            collated_dict["bt_sample_id"] = bt_sample_id
            collated_dict["bt_conv_id"] = bt_conv_id
            collated_dict["bt_turn_id"] = bt_turn_id
            collated_dict["bt_pair_query"] = bt_pair_query
            collated_dict["bt_pair_query_mask"] = bt_pair_query_mask
            collated_dict["bt_query_passage"] = bt_query_passage
            collated_dict["bt_query_passage_mask"] = bt_query_passage_mask
            
            # change to tensor
            for key in collated_dict:
                if key not in ["bt_sample_id", "bt_conv_id", "bt_turn_id"]:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
            return collated_dict

        return collate_fn


