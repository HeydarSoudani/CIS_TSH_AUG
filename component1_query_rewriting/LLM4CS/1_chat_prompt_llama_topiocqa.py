#!/usr/bin/env python3

import time
from tqdm import tqdm
import os, json, argparse

from src.utils import set_seed, get_finished_sample_ids, get_has_qrel_label_sample_ids
from src.chat_prompter import RewriteAndResponsePromptor
from src_llama.llama_wrapper import Llama3HFWrapper


def main(args):
    
    # === Test_dataset ================= 
    output_file_path = f"{args.output_dir}/{args.dataset_name}/LLM4CS/rewrites_llama3.jsonl"
    os.makedirs(f"{args.output_dir}/{args.dataset_name}/LLM4CS", exist_ok=True)
    finished_samples = get_finished_sample_ids(output_file_path)
    has_qrel_labels_samples = get_has_qrel_label_sample_ids(args.qrel_file_path)
    
    
    # === Model and promptor setting ===
    model_wrapper = Llama3HFWrapper(max_tokens=args.max_new_tokens, sys_prompt=None) # None uses default
    # model_wrapper = Llama3SGLangWrapper(max_tokens=768, sys_prompt=None) # None uses default
    promptor = RewriteAndResponsePromptor(args.demo_file_path, enable_cot=True)
    
    # === Predict ======================
    begin_time = time.time()
    with open(args.test_file_path, "r") as in_f, open(output_file_path, "a+") as out_f:
        for idx, line in enumerate(in_f):
            
            if idx == 2:
                break
            
            conversation = json.loads(line.strip())
            sample_id = conversation["sample_id"]
            if sample_id in finished_samples or sample_id not in has_qrel_labels_samples:
                continue
            
            
            prompt = promptor.build_turn_prompt_topiocqa(conversation)
            prompt = model_wrapper.formatter.format_prompt(prompt)
            completion = model_wrapper.get_completion_from_prompt(prompt)
            print(completion)
        
        # record = {}
        # record['sample_id'] = sample_id
        # record['predicted_rewrite'] = rewrite_list
        # record['predicted_response'] = response_list
        # record['predicted_cot'] = cot_list
        
        # out_f.write(json.dumps(record))
        # out_f.write('\n')
        # out_f.flush()
    
    print("{} Generation ok!, time cost {}".format(args.output_dir, time.time() - begin_time))
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file_path", type=str, default="processed_datasets/TopiOCQA/dev_qrecc_format.json")
    parser.add_argument("--qrel_file_path", type=str, default="processed_datasets/TopiOCQA/test_gold_qrels.trec")
    parser.add_argument("--demo_file_path", type=str, default="component1_query_rewriting/LLM4CS/src/demonstrations.json")
    parser.add_argument("--output_dir", type=str, default="component3_retriever/input_data", help='output rewrite path.')
    parser.add_argument("--dataset_name", type=str, default="TopiOCQA", choices=["QReCC", "TopiOCQA", "INSCIT"])
    parser.add_argument("--max_new_tokens", type=int, default=256, help='')
    parser.add_argument("--n_generation", type=int, default=5, help='the number for generation')
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)   
    
    set_seed(args)
    main(args)
    
    # python component1_query_rewriting/LLM4CS/1_chat_prompt_llama_topiocqa.py
    
    
    
