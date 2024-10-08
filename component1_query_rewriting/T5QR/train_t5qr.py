#!/usr/bin/env python3


### ==============================================================================
# Ref: https://github.com/kyriemao/T5QR/blob/main/train_t5qr.py
### ==============================================================================


# from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.append('..')
sys.path.append('.')
import time
import numpy as np
import argparse
from os.path import join as oj
from tqdm import tqdm, trange

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from transformers import get_linear_schedule_with_warmup, AdamW, T5Tokenizer, T5ForConditionalGeneration

from utils import check_dir_exist_or_build, set_seed, json_dumps_arguments, train_dev_split_qrecc
from dataset import T5RewriterDataset, Collator


def save_model(output_checkpoint_path, model, query_tokenizer, epoch):
    output_dir = oj(output_checkpoint_path, 'epoch-{}'.format(epoch))
    check_dir_exist_or_build([output_dir], True)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    query_tokenizer.save_pretrained(output_dir)
    logger.info("Save checkpoint at {}".format(output_dir))

def train_t5qr(args):
    if args.need_output:
        # check_dir_exist_or_build([args.log_path], args.force_emptying_dir)
        check_dir_exist_or_build([args.log_path], True)
        log_writer = SummaryWriter(log_dir = args.log_path)
    else:
        log_writer = None

    # model
    model = T5ForConditionalGeneration.from_pretrained(args.model_path)
    model.to(args.device)
    tokenizer = T5Tokenizer.from_pretrained(args.model_path)
    special_token_dict = {"cls_token":"[CLS]", "sep_token": "[SEP]"}
    tokenizer.add_special_tokens(special_token_dict)
    model.resize_token_embeddings(len(tokenizer))
    args.tokenizer = tokenizer

    # training data and optimizer
    train_dataset = T5RewriterDataset(args, args.train_file_path)
    collate_kwargs = {"tokenizer": args.tokenizer, 
                      "max_query_length": args.max_query_length, 
                      "max_seq_length": args.max_seq_length,
                      "collate_type": "train"}
    collate_fn = Collator(**collate_kwargs)
    train_dataloader = DataLoader(train_dataset, 
                                  shuffle=True,
                                  batch_size=args.train_batch_size, 
                                  collate_fn=collate_fn)



    dev_dataset = T5RewriterDataset(args, args.dev_file_path)
    dev_dataloader = DataLoader(dev_dataset, 
                                  shuffle=False,
                                  batch_size=args.dev_batch_size, 
                                  collate_fn=collate_fn)
    total_training_steps = args.num_train_epochs * (len(train_dataset) // args.train_batch_size + int(bool(len(train_dataset) % args.train_batch_size)))    

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_training_steps)

    # saving/log prepare
    num_steps_per_epoch = total_training_steps // args.num_train_epochs
    if isinstance(args.model_save_steps, float):
        args.model_save_steps = int(args.model_save_steps * num_steps_per_epoch)
        args.model_save_steps = max(1, args.model_save_steps)
    if isinstance(args.log_print_steps, float):
        args.log_print_steps = int(args.log_print_steps * num_steps_per_epoch)
        args.log_print_steps = max(1, args.log_print_steps)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", total_training_steps)

    cur_step = 0
    min_dev_loss = float('inf')
    break_count = 0

    epoch_iterator = trange(args.num_train_epochs, desc="Epoch")
    for epoch in epoch_iterator:
        if break_count == 2:
            break
        for batch in tqdm(train_dataloader,  desc="Step"):
            if break_count == 2:
                logger.info("Trigger early stop at epoch-{}. Current step-{}.".format(epoch, cur_step))
                break

            model.zero_grad()
            model.train()
            
            bt_input_ids, bt_attention_mask, bt_labels = (batch["bt_input_ids"], batch["bt_attention_mask"], batch["bt_labels"])
            bt_input_ids = bt_input_ids.to(args.device)
            bt_attention_mask = bt_attention_mask.to(args.device)
            bt_labels = bt_labels.to(args.device)
    
            loss = model(input_ids=bt_input_ids, 
                         attention_mask=bt_attention_mask, 
                         labels=bt_labels).loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            if (args.log_print_steps > 0) and (cur_step % args.log_print_steps == 0):
                logger.info("Epoch = {}, Cur Step = {}, Train Loss = {}".format(
                                epoch,
                                cur_step,
                                loss.item())
                            )
            if log_writer:
                log_writer.add_scalar("train_t5_qr_loss", loss, cur_step)
            cur_step += 1    # avoid saving the model of the first step.
            
            
            # if args.need_output and cur_step % args.model_save_steps == 0:
            if cur_step % args.model_save_steps == 0:
                save_model(args.output_checkpoint_path, model, tokenizer, epoch)
                
                # Validation
                dev_loss = validation(model, dev_dataloader)
                # log_writer.add_scalar("dev_t5_qr_loss", dev_loss, cur_step)
                dev_loss = dev_loss.item()
                logger.info("Epoch-{}-Step-{}, Dev loss: {}.".format(epoch, cur_step, dev_loss))
                if dev_loss < min_dev_loss:
                    min_dev_loss = dev_loss
                    break_count = 0
                else:
                    break_count += 1
    
    logger.info("Training finish!")

def validation(model, dev_dataloader):
    dev_loss = 0.0
    with torch.no_grad():
        model.eval()
        for batch in tqdm(dev_dataloader,  desc="Step"):
            bt_input_ids, bt_attention_mask, bt_labels = (batch["bt_input_ids"], batch["bt_attention_mask"], batch["bt_labels"])
            bt_input_ids = bt_input_ids.to(args.device)
            bt_attention_mask = bt_attention_mask.to(args.device)
            bt_labels = bt_labels.to(args.device)
    
            loss = model(input_ids=bt_input_ids, 
                         attention_mask=bt_attention_mask, 
                         labels=bt_labels).loss
            dev_loss += loss
            
    return dev_loss

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="google-t5/t5-base")
    parser.add_argument("--train_file_path", type=str, default="processed_datasets/QReCC/new_train_qrecc.json")
    parser.add_argument("--dev_file_path", type=str, default="processed_datasets/QReCC/new_dev_qrecc.json")
    parser.add_argument("--log_path", type=str, default="component1_query_rewriting/T5QR/outputs/log")
    parser.add_argument("--output_checkpoint_path", type=str, default="component1_query_rewriting/T5QR/outputs/checkpoints")
    parser.add_argument("--output_dir_path", type=str, default="component1_query_rewriting/T5QR/outputs")
    parser.add_argument("--need_output", action="store_true", help="Whether need to output logs and models (creating the dirs)")
    parser.add_argument("--force_emptying_dir", action="store_true", help="Force to empty the (output) dir.")

    parser.add_argument("--log_print_steps", type=int, default=250, help="Percent of steps per epoch to print once.")
    parser.add_argument("--model_save_steps", type=int, default=500)

    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--use_data_percent", type=float, default=1.0, help="Percent of samples to use. Faciliating the debugging.")
    parser.add_argument("--num_train_epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--train_batch_size", type=int, default=48, help="train batch size, only one GPU")
    parser.add_argument("--dev_batch_size", type=int, default=48, help="train batch size, only one GPU")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="adam epsilon")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm.")
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Warm up steps.")

    parser.add_argument("--max_query_length", type=int, default=32, help="Max single query length")
    parser.add_argument("--max_response_length", type=int, default=100, help="Max response token length")
    parser.add_argument("--max_seq_length", type=int, default=384, help="Max concatenation length of the session.")
    

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.start_running_time = time.asctime(time.localtime(time.time()))
    logger.info("---------------------The arguments are:---------------------")
    logger.info(args)
    
    # if args.need_output:
    check_dir_exist_or_build([args.output_dir_path], force_emptying=True)
    json_dumps_arguments(oj(args.output_dir_path, "parameters.txt"), args)
        
    return args


if __name__ == '__main__':
    args = get_args()
    set_seed(args)
    train_t5qr(args)
    
# python component1_query_rewriting/T5QR/train_t5qr.py
