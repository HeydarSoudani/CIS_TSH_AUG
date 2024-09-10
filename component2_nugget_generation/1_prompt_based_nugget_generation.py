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
from src.llm_model import nugget_extraction_prompt_v2,\
                        nugget_extraction_prompt_only_query,\
                        topic_aware_query_rewriting,\
                        topic_generation_prompt,\
                        topic_generation_prompt_cot,\
                        topic_generation_100p_shift_detector,\
                        topic_generation_100p_shift_detector_2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

subset_percentage = 1.0
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
    ### === Load model ===============================
    llama3_model = Generator_HF(args.model_name_or_path, args.max_tokens)
    
    ### === Load & prepare data ======================
    conversation_data = []
    conversation_data_obj = {}
    with open(args.test_file_path, 'r') as in_file:
        for line in in_file:
            sample = json.loads(line)
            conversation_data.append(sample)
            conversation_data_obj[sample["id"]] = sample
    
    ### === Load generated topic ======================
    gen_topic_file = "processed_datasets/TopiOCQA/topic_generation.json"
    generated_topic = {}
    with open(gen_topic_file, 'r') as in_file:
        for line in in_file:
            sample = json.loads(line)
            generated_topic[sample["query_id"]] = sample["output"]
    
    # === Load shifted turns (+ first) ================
    bucket_file = f"processed_datasets/TopiOCQA/turn_buckets/per_shift.json"
    with open(bucket_file, 'r') as f:
        bucket_data = json.load(f)
    first_turns = bucket_data["First"]
    shift_turns = bucket_data["Topic-shift"]
    all_turns = first_turns + shift_turns
    print(len(all_turns))
    
    
    # === Random subset selection =====================
    if subset_percentage == 1.0:
        subset_conversation_data = conversation_data
    else:
        conversation_data_subset_size = int(subset_percentage * len(conversation_data))
        subset_conversation_data = random.sample(conversation_data, conversation_data_subset_size)
    
    ### === Loop on conversation samples ==============
    with open(args.output_file_path, 'w') as out_file:
        for query_idx, conversation_sample in tqdm(enumerate(subset_conversation_data)):
            
            # if query_idx == 30:
            #     break
            
            query_id = conversation_sample["id"]
            conv_id = conversation_sample["conv_id"]
            turn_id = conversation_sample["turn_id"]
            current_query = conversation_sample["query"]
            answer = conversation_sample["answer"]
            g_passage = conversation_sample["pos_docs"][0]
            gold_topic = conversation_sample["topic"]
            gen_topic = generated_topic[query_id]
            
            conversation_hist = ""
            # if args.history_window == 0 or turn_id <= args.history_window+1:
            #     for tid in range(1, turn_id):
            #         item = conversation_data_obj[f'{conv_id}_{tid}']
            #         conversation_hist += f"Turn {tid}: {item['query']} {item['answer']}, Topic: {item['topic']}\n"
            # else:
            #     for tid in range(turn_id-args.history_window, turn_id):
            #         item = conversation_data_obj[f'{conv_id}_{tid}']
            #         conversation_hist += f"Turn {tid}: {item['query']} {item['answer']}, Topic: {item['topic']}\n"
    
            if args.history_window == 0 or turn_id <= args.history_window+1:
                for tid in range(1, turn_id):
                    item = conversation_data_obj[f'{conv_id}_{tid}']
                    conversation_hist += f"Turn {tid}: {item['query']} {item['answer']}\n"
            else:
                for tid in range(turn_id-args.history_window, turn_id):
                    item = conversation_data_obj[f'{conv_id}_{tid}']
                    conversation_hist += f"Turn {tid}: {item['query']} {item['answer']}\n"
    
            # == Query & passage ======
            # conversation_turn = {'query': current_query, 'answer': answer, 'passage': g_passage}
            # prompt = nugget_extraction_prompt_v2(conversation_turn)
            
            # == Query & history ======
            # conversation_turn = {'history': conversation_hist, 'query': current_query}
            # prompt = nugget_extraction_prompt_only_query(conversation_turn)
            
            # == Query rewriting with gold/generated topic ======
            # conversation_turn = {
            #     'history': conversation_hist,
            #     'query': current_query,
            #     'gold_topic': gold_topic,
            #     'gen_topic': gen_topic
            # }
            # prompt = topic_aware_query_rewriting(conversation_turn)
            
            # == Topic Generation =========================
            query = f"Turn {turn_id}: {current_query}"
            conversation_turn = {'history': conversation_hist, 'query': query}
            # prompt = topic_generation_prompt(conversation_turn)
            # prompt = topic_generation_prompt_cot(conversation_turn)
            # prompt = topic_generation_100p_shift_detector(conversation_turn) # Topic Generation, 100% shift detector
            prompt = topic_generation_100p_shift_detector_2(conversation_turn)
            
            # == Generate output =====
            prompt = llama3_model.formatter.format_prompt(prompt)
            
            if query_id in all_turns:
                completion = llama3_model.get_completion_from_prompt(prompt)
            else:
                completion = ""
            
            
            if query_idx < 10 or query_idx % 100 == 0:
                logger.info(f"Prompt: {prompt}")
                logger.info(f"Query_id: {query_id}")
                logger.info(f"Question: {current_query}")
                logger.info(f"Answer: {answer}")
                logger.info(f"Topic: {gold_topic}")
                logger.info(f"Output: {completion}")
            
            item = {
                "query_id": query_id,
                "question": current_query,
                "answer": answer,
                "topic": gold_topic,
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
    parser.add_argument("--output_file_path", type=str, default="processed_datasets/TopiOCQA/topic_gen_100p_shift_detector_no_topic.json")
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--history_window", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    main(args)
    
    # python component2_nugget_generation/1_prompt_based_nugget_generation.py
