#!/usr/bin/env python3


### ==============================================================================
# Ref: https://github.com/kyriemao/T5QR/blob/main/inference_t5qr.py
### ==============================================================================


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
sys.path.append('..')
sys.path.append('.')

import json
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from utils import set_seed
from dataset import T5RewriterDataset, Collator

def inference_t5qr(args):
    model = T5ForConditionalGeneration.from_pretrained(args.model_checkpoint_path)
    tokenizer = T5Tokenizer.from_pretrained(args.model_checkpoint_path)
    assert tokenizer.sep_token == "[SEP]"
    args.tokenizer = tokenizer

    model.to(args.device)
    if args.n_gpu > 1:
        model = DDP(model, device_ids = [args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    # dist.barrier()

    test_dataset = T5RewriterDataset(args, args.test_file_path)
    collate_kwargs = {"tokenizer": args.tokenizer, 
                      "max_query_length": args.max_query_length, 
                      "max_seq_length": args.max_seq_length,
                      "collate_type": "test"}
    collate_fn = Collator(**collate_kwargs)
    args.batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # ddp_sampler = DistributedSampler(test_dataset, shuffle=False)
    test_dataloader = DataLoader(test_dataset, 
                                #   sampler=ddp_sampler,
                                  batch_size=args.batch_size, 
                                  collate_fn=collate_fn)
    
    results = []  
    with torch.no_grad():
        model.eval()
        for batch in tqdm(test_dataloader, desc="Step"):
            bt_input_ids = batch["bt_input_ids"].to(args.device)
            bt_attention_mask = batch["bt_attention_mask"].to(args.device)
            if args.n_gpu > 1:
                output_seqs = model.module.generate(input_ids=bt_input_ids, 
                                                    attention_mask=bt_attention_mask, 
                                                    do_sample=False,
                                                    max_length=args.max_query_length)
            else:
                output_seqs = model.generate(input_ids=bt_input_ids, 
                                                    attention_mask=bt_attention_mask, 
                                                    do_sample=False,
                                                    max_length=args.max_query_length)
            outputs = tokenizer.batch_decode(output_seqs, skip_special_tokens=True)
            for i in range(len(outputs)):
                record = {}
                record["sample_id"] = batch["bt_sample_ids"][i]
                record["utterance_context"] = batch["bt_ctx_utts_text"][i]
                record["response_context"] = batch["bt_ctx_resps_text"][i]
                record["current_utterance"] = batch["bt_cur_utt_text"][i]
                record["oracle_rewrite"] = batch["bt_oracle_utt_text"][i]
                record["t5_rewrite"] = outputs[i]
                record["human_judge"] = ""
                results.append(record)

    # We use "a+" for DDP, but note that it will cause the output json file cannot be loaded with json.load(f)
    # if using multiple GPUs, because there will be n_gpu "results" dumped into the output file.
    with open(args.output_file_path, "a+") as f:    
        f.write(json.dumps(results, indent=4))

    logger.info("Inference finsh!")

def get_args():
    
    # === QReCC
    # test_file_path: processed_datasets/QReCC/new_test.json
    # output_file_path: component3_retriever/input_data/QReCC/T5QR/t5_rewrite.json
    # === TopiOCQA
    # test_file_path: processed_datasets/TopiOCQA/dev_qrecc_format.json
    # output_file_path: component3_retriever/input_data/TopiOCQA/T5QR/t5_rewrite.json
    # === INSCIT
    # test_file_path:
    # output_file_path:
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint_path", type=str, default="component1_query_rewriting/T5QR/query_rewriter_models/QReCC_checkpoints/epoch-4")
    parser.add_argument("--test_file_path", type=str, default="processed_datasets/TopiOCQA/dev_qrecc_format.json")
    parser.add_argument('--output_file_path', type=str, default="component3_retriever/input_data/TopiOCQA/T5QR/t5_rewrite.json")
    parser.add_argument('--local_rank', type=int, default=1, metavar='N', help='Local process rank.')  # you need this argument in your scripts for DDP to work
    parser.add_argument("--per_gpu_eval_batch_size", type=int, default=64)
    parser.add_argument("--use_data_percent", type=float, default=1.0)
    parser.add_argument("--max_query_length", type=int, default=32, help="Max single query length")
    parser.add_argument("--max_response_length", type=int, default=128, help="Max response token length")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Max concatenation length of the session.")
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()

    # pytorch parallel gpu
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu", args.local_rank)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    # dist.init_process_group(backend='nccl', init_method='env://')
    # torch.cuda.set_device(args.local_rank)

    return args

if __name__ == '__main__':
    args = get_args()
    set_seed(args)

    logger.info("---------------------The arguments are:---------------------")
    logger.info(args)

    inference_t5qr(args)

# python component1_query_rewriting/T5QR/test_t5qr.py