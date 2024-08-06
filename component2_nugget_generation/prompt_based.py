#!/usr/bin/env python3

import os
import json
import torch
import random
import argparse
import numpy as np

from llm_model import LLMModel_hf
from llm_model import nugget_extraction_prompt, nugget_extraction_prompt_first_turn, nugget_extraction_prompt_v2


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

subset_percentage = 0.1
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
    # llama3_model = LLMModel_vllm(args.model_name_or_path)
    llama3_model = LLMModel_hf(args.model_name_or_path)
    
    
    ### === Load & prepare data ======================
    conversation_data = []
    with open(args.test_file, 'r') as in_file:
        for line in in_file:
            sample = json.loads(line)
            conversation_data.append(sample)
    
    # === Random subset selection
    if subset_percentage == 1.0:
        subset_conversation_data = conversation_data
    else:
        conversation_data_subset_size = int(subset_percentage * len(conversation_data))
        subset_conversation_data = random.sample(conversation_data, conversation_data_subset_size)
    
    
    ### === Loop on conversation samples =======
    max_try_num = 3
    with open(args.output_results_file, 'w') as out_file:
        for query_idx, conversation_sample in enumerate(subset_conversation_data):
            
            if query_idx == 1:
                break
            
            query_id = conversation_sample["id"]
            conv_id = conversation_sample["conv_id"]
            turn_id = conversation_sample["turn_id"]
            current_query = conversation_sample["query"]
            answer = conversation_sample["answer"]
            g_passage = conversation_sample["pos_docs"][0]
            
            # === V1
            # conv_history = []
            # if turn_id == 1:
            #     nuggets = []
            # else:
            #     for tid in range(1, turn_id):
            #         conv_his_turn = conversation_data[f"{conv_id}_{tid}"]
                    
            #         conv_history.append({
            #             "query": conv_his_turn["query"],
            #             "answer": conv_his_turn["answer"],
            #             "passage": conv_his_turn["pos_docs"][0]
            #         })
            #     input_text = nugget_extraction_prompt(current_query, conv_history)
                
            #     response = llama3_model.generate_text(input_text)
            #     output_text = response[0]["generated_text"].split('/INST]')[1]
            #     # output_text = response[0].outputs[0].text
            #     nuggets = llama3_model.pattern_extractor(output_text)
            #     print(output_text)
            
            # === V2
            conversation_turn = {'query': current_query, 'answer': answer, 'passage': g_passage}
            input_text = nugget_extraction_prompt_v2(conversation_turn)    
            
            print(input_text)
            
            # while True:
            for i in range (max_try_num):
                response = llama3_model.generate_text(input_text)
            
                print(f"response: {response}")
                output_text = response[0]["generated_text"].split('<|start_header_id|>assistant<|end_header_id|>')[-1]
                print(output_text)
                # output_text = response[0]["generated_text"].split('/INST]')[1]
                nuggets = llama3_model.pattern_extractor(output_text)
            
                print(f"Nuggets: {nuggets}")
                print('\n')
                if nuggets is not None:
                    print("JSON successfully parsed!")
                    break
                else:
                    print(f"LLM output is not correct for '{current_query}' in try: {i}")
            
            item = {
                "query_id": query_id,
                "question": current_query,
                "answer": answer,
                "nuggets": nuggets
            }
            out_file.write(json.dumps(item) + '\n')
   
if __name__ == "__main__":
    
    # "meta-llama/Llama-2-7b-chat-hf"
    # "meta-llama/Meta-Llama-3-8B-Instruct"
    # "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--test_file", type=str, default="processed_datasets/TopiOCQA/dev_new.json")
    parser.add_argument("--output_results_file", type=str, default="processed_datasets/TopiOCQA/dev_nuggets.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    main(args)
    
    # python component2_nugget_generation/prompt_based.py
