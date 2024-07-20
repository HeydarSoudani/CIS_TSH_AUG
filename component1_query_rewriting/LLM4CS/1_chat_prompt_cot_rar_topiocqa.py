#!/usr/bin/env python3

import time
import os, json, argparse

from src.generator import ChatGenerator, OPENAI_KEYS
from src.llama3_generator import Llama3Generator
from src.chat_prompter import RewriteAndResponsePromptor
from src.utils import set_seed, get_finished_sample_ids, get_has_qrel_label_sample_ids


def main(args):
    
    # model and promptor setting
    promptor = RewriteAndResponsePromptor(args.demo_file_path, enable_cot=True)
    # model_kwargs = {"temperature": 0.7, "max_new_tokens": 256} # "stop_tokens": ["<|endoftext|>", "STOP"]
    # generator = Llama3Generator(args.n_generation, **model_kwargs)
    
    model_kwargs = {"temperature": 0.7, "max_tokens": 256, "stop": promptor.stop_tokens}
    api_key = OPENAI_KEYS[args.open_ai_key_id]
    generator = ChatGenerator(api_key, args.n_generation, **model_kwargs)

    # test_dataset    
    output_file_path = f"{args.output_dir}/{args.dataset_name}/LLM4CS/rewrites.jsonl"
    os.makedirs(f"{args.output_dir}/{args.dataset_name}/LLM4CS", exist_ok=True)
    finished_samples = get_finished_sample_ids(output_file_path)
    has_qrel_labels_samples = get_has_qrel_label_sample_ids(args.qrel_file_path)
    
    # predict
    begin_time = time.time()
    with open(args.test_file_path, "r") as in_f, open(output_file_path, "a+") as out_f:
        for idx, line in enumerate(in_f):
            
            if idx == 3:
                break
            
            conversation = json.loads(line.strip())
            sample_id = conversation["sample_id"]
            
            if sample_id in finished_samples or sample_id not in has_qrel_labels_samples:
                continue
            
            prompt = promptor.build_turn_prompt_topiocqa(conversation)
            print(prompt)
            n_outputs = generator.generate(prompt, promptor.parse_returned_text)
            print(n_outputs)
            
            cot_list, rewrite_list, response_list = list(zip(*n_outputs))
            
            # if idx < 10 or idx % 100 == 0:
            
            
            record = {}
            record['sample_id'] = sample_id
            record['predicted_rewrite'] = rewrite_list
            record['predicted_response'] = response_list
            record['predicted_cot'] = cot_list
            
            out_f.write(json.dumps(record))
            out_f.write('\n')
            out_f.flush()
    print("{} Generation ok!, time cost {}".format(args.output_dir, time.time() - begin_time))
            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file_path", type=str, default="processed_datasets/TopiOCQA/dev_qrecc_format.json")
    parser.add_argument("--qrel_file_path", type=str, default="processed_datasets/TopiOCQA/test_gold_qrels.trec")
    parser.add_argument("--demo_file_path", type=str, default="component1_query_rewriting/LLM4CS/src/demonstrations.json")
    parser.add_argument("--output_dir", type=str, default="component3_retriever/input_data", help='output rewrite path.')
    parser.add_argument("--dataset_name", type=str, default="TopiOCQA", choices=["QReCC", "TopiOCQA", "INSCIT"])
    parser.add_argument("--n_generation", type=int, default=5, help='the number for generation')
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--open_ai_key_id", type=int, choices=[0,1,2,3,4,5], default=0)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)   
    set_seed(args)
    
    main(args)
    
    # python component1_query_rewriting/LLM4CS/chat_prompt_cot_rar.py