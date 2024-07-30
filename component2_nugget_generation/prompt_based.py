#!/usr/bin/env python3

import os
import json
import torch
import random
import argparse
import numpy as np

from llm_model import LLMModel
from llm_model import nugget_extraction_prompt, nugget_extraction_prompt_first_turn


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
    llama3_model = LLMModel(args.model_name_or_path)
    
    
    ### === Load & prepare data ======================
    conversation_data = {}
    with open(args.test_file, 'r') as in_file:
        for line in in_file:
            sample = json.loads(line)
            conversation_data[sample["id"]] = sample
    
    with open(args.output_results_file, 'w') as out_file:
        for query_idx, (query_id, conversation_sample) in enumerate(conversation_data.items()):
            
            if query_idx == 3:
                break
            
            conv_id = conversation_sample["conv_id"]
            turn_id = conversation_sample["turn_id"]
            current_query = conversation_sample["query"]
            answer = conversation_sample["answer"]
            
            conv_history = []
            if turn_id != 1:
                for tid in range(1, turn_id):
                    conv_his_turn = conversation_data[f"{conv_id}_{tid}"]
                    
                    conv_history.append({
                        "query": conv_his_turn["query"],
                        "answer": conv_his_turn["answer"],
                        "passage": conv_his_turn["pos_docs"][0],
                    })
                input_text = nugget_extraction_prompt(current_query, conv_history)
            else:
                input_text = nugget_extraction_prompt_first_turn(current_query)
                
            print(input_text) 
            print('\n')   
                
            # formatted_prompt = llama3_model.format_prompt(input_text)
            response = llama3_model.generate_text(input_text)
            print(response)
            # output_text = response[0].outputs[0]['text']
            # print(output_text)
            print('\n')
            
            
            item = {
                "query_id": query_id,
                "question": current_query,
                "answer": answer,
                "nuggets": ""
            }
            out_file.write(json.dumps(item) + '\n')
            
            
if __name__ == "__main__":
    
    # "meta-llama/Llama-2-7b-chat-hf"
    # "meta-llama/Meta-Llama-3-8B-Instruct"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--test_file", type=str, default="processed_datasets/TopiOCQA/dev_new.json")
    parser.add_argument("--output_results_file", type=str, default="processed_datasets/TopiOCQA/dev_nuggets.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    main(args)
    
    # python component2_nugget_generation/prompt_based.py
