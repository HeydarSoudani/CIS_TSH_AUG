#!/usr/bin/env python3

import os
import json
import torch
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm

from models import Generator_HF

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

subset_percentage = 0.01
print("Available GPUs:", torch.cuda.device_count())
device = 'cuda:0'

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prompt_creation():
    pass

def main(args):
    
    ### === Load model ===============================
    llama3_model = Generator_HF(args.model_name_or_path)
    
    ### === Load & prepare data ======================
    conversation_data = []
    conversation_data_obj = {}
    with open(args.test_file_path, 'r') as in_file:
        for line in in_file:
            sample = json.loads(line)
            conversation_data.append(sample)
            conversation_data_obj[sample["id"]] = sample
    
    # = Random subset selection
    if subset_percentage == 1.0:
        subset_conversation_data = conversation_data
    else:
        conversation_data_subset_size = int(subset_percentage * len(conversation_data))
        subset_conversation_data = random.sample(conversation_data, conversation_data_subset_size)
    
    ### === Read prompt file ========================= 
    generation_prompt = open(args.prompt_file_path, "r").read()
    
    
    ### === Loop on conversation samples =============
    with open(args.output_file_path, 'w') as out_file:
        for query_idx, conversation_sample in tqdm(enumerate(subset_conversation_data)):
            
            # if query_idx == 10:
            #     break
            
            query_id = conversation_sample["id"]
            conv_id = conversation_sample["conv_id"]
            turn_id = conversation_sample["turn_id"]
            current_query = conversation_sample["query"]
            answer = conversation_sample["answer"]
            gold_topic = conversation_sample["topic"]
            g_passage = conversation_sample["pos_docs"][0]    
            
            # == Query & history ======
            conversation_hist = ""
            for tid in range(1, turn_id):
                conversation_hist += f"turn {tid}: {conversation_data_obj[f'{conv_id}_{tid}']['query']} {conversation_data_obj[f'{conv_id}_{tid}']['answer']}\n"
            # conversation_turn = {'history': conversation_hist, 'query': current_query}
            
            # == Generate output =====
            prompt = generation_prompt.format(Conversation_history=conversation_hist, Query=current_query)
            prompt = llama3_model.formatter.format_prompt(prompt)
            completion = llama3_model.get_completion_from_prompt(prompt)
    
            if query_idx < 10 or query_idx % 100 == 0:
                logger.info(f"Prompt: {prompt}")
                logger.info(f"Query_id: {query_id}")
                logger.info(f"Gold topic: {gold_topic}")
                logger.info(f"Question: {current_query}")
                logger.info(f"Answer: {answer}")
                logger.info(f"Output: {completion}")
            
            item = {
                "query_id": query_id,
                "gold_topic": gold_topic,
                "question": current_query,
                "answer": answer,
                "output": completion
            }
            out_file.write(json.dumps(item) + '\n')
    
    

if __name__ == "__main__":
    
    # "meta-llama/Llama-2-7b-chat-hf"
    # "meta-llama/Meta-Llama-3-8B-Instruct"
    # "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--test_file_path", type=str, default="processed_datasets/TopiOCQA/dev_new.json")
    parser.add_argument("--output_file_path", type=str, default="processed_datasets/TopiOCQA/dev_topic.json")
    parser.add_argument("--prompt_file_path", type=str, default="component2_topic_generation/prompt.txt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    main(args)
    
    
    # python component2_topic_generation/generation.py