#!/usr/bin/env python3

import os
import json
import torch
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm

from src.llm_model import Generator_HF
from src.llm_model import object_extraction_prompt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    
    llama3_model = Generator_HF(args.model_name_or_path, max_tokens=256)
    
    with open(args.empty_nugget_file, 'r') as in_file, open(args.output_file, 'w') as out_file:
        for query_idx, line in tqdm(enumerate(in_file)):
            sample = json.loads(line)
            query_id = sample["query_id"]
            nuggets = sample["nuggets"]
            
            prompt = object_extraction_prompt(nuggets)
            prompt = llama3_model.formatter.format_prompt(prompt)
            completion = llama3_model.get_completion_from_prompt(prompt)
            
            if query_idx < 10 or query_idx % 50 == 0:
                logger.info(f"query_id: {query_id}")
                logger.info(f"nuggets: {completion}")
            
            item = {
                "query_id": query_id,
                "nuggets": completion
            }
            out_file.write(json.dumps(item) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--empty_nugget_file", type=str, default="processed_datasets/TopiOCQA/empty_nuggets.json")
    parser.add_argument("--output_file", type=str, default="processed_datasets/TopiOCQA/out_empty_nuggets_2.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    main(args)
    
    # python component2_nugget_generation/2_postprocessing_nug_extraction_prompting.py
    