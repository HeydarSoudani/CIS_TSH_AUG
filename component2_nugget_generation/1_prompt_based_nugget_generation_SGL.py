#!/usr/bin/env python3

import os
import json
import torch
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm

from src.llm_model import Generator_HF, Generator_SGL
from src.llm_model import nugget_extraction_prompt, nugget_extraction_prompt_first_turn, nugget_extraction_prompt_v2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

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

def main(args):
    set_seed(args.seed)
    
    ### === Load model ===============================
    llama3_model = Generator_SGL(args.model_name_or_path)
    
    ### === Load & prepare data ======================
    conversation_data = []
    with open(args.test_file_path, 'r') as in_file:
        for line in in_file:
            sample = json.loads(line)
            conversation_data.append(sample)
    
    # = Random subset selection
    if subset_percentage == 1.0:
        subset_conversation_data = conversation_data
    else:
        conversation_data_subset_size = int(subset_percentage * len(conversation_data))
        subset_conversation_data = random.sample(conversation_data, conversation_data_subset_size)
    
    ### === Prepare prompts ======================
    prompts = []
    for query_idx, conversation_sample in tqdm(enumerate(subset_conversation_data)):
        query_id = conversation_sample["id"]
        conv_id = conversation_sample["conv_id"]
        turn_id = conversation_sample["turn_id"]
        current_query = conversation_sample["query"]
        answer = conversation_sample["answer"]
        g_passage = conversation_sample["pos_docs"][0]
        
        conversation_turn = {'query': current_query, 'answer': answer, 'passage': g_passage}
        prompt = nugget_extraction_prompt_v2(conversation_turn)    
        prompt = llama3_model.formatter.format_prompt(prompt)   
        prompts.append(prompt)       
        
    ### === Loop on conversation samples =======
    completions = llama3_model.get_completion_from_prompt(prompts)
      
    with open(args.output_file_path, 'w') as out_file:    
        for idx, completion in enumerate(completions):   
            query_id = conversation_sample[idx]["id"]
            current_query = conversation_sample[idx]["query"]
            answer = conversation_sample[idx]["answer"]
            item = {
                "query_id": query_id,
                "question": current_query,
                "answer": answer,
                "nuggets": completion
            }
            out_file.write(json.dumps(item) + '\n')
   
if __name__ == "__main__":
    
    # "meta-llama/Llama-2-7b-chat-hf"
    # "meta-llama/Meta-Llama-3-8B-Instruct"
    # "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--test_file_path", type=str, default="processed_datasets/TopiOCQA/dev_new.json")
    parser.add_argument("--output_file_path", type=str, default="processed_datasets/TopiOCQA/dev_nuggets_sgl.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    main(args)
    
    # python component2_nugget_generation/1_prompt_based_nugget_generation.py
