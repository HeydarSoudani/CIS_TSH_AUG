#!/usr/bin/env python3

### ==============================================================================
### This code is based on the following files: 
### (1) https://github.com/fengranMark/HAConvDR/blob/main/gen_tokenized_doc.py
### (2) https://github.com/fengranMark/HAConvDR/blob/main/gen_doc_embeddings.py
### (3) https://github.com/fengranMark/HAConvDR/blob/main/Config/gen_tokenized_doc.toml
### (4) https://github.com/fengranMark/HAConvDR/blob/main/Config/gen_doc_embeddings.toml
### ==============================================================================


import os
import gc
import sys
import json
import pickle
import logging
import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing import Process

import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import TensorDataset, DataLoader

torch.multiprocessing.set_sharing_strategy('file_system')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.models import load_model
from src.utils import (
    barrier_array_merge,
    StreamingDataset,
    EmbeddingCache,
)

WIKI_FILE = "corpus/TopiOCQA/full_wiki_segments.tsv"
TOKENIZED_DOC_DIR = "corpus/TopiOCQA/dense_tokenized"
EMBEDDED_DOC_DIR = "corpus/TopiOCQA/dense_embedded"
model_type = "ANCE"
# pretrained_passage_encoder = "sentence-transformers/msmarco-roberta-base-ance-firstp"   # passage encoder!!!
pretrained_passage_encoder = "castorini/ance-msmarco-passage"
max_seq_length = 384    # The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
max_doc_character = 10000   # used before tokenizer to save tokenizer latency
per_gpu_eval_batch_size = 250 # defualt= 250
local_rank = -1 # Not use distributed training
disable_tqdm = False
n_gpu = 1


def pad_input_ids(input_ids, max_length, pad_on_left=False, pad_token=0):
    padding_length = max_length - len(input_ids)
    padding_id = [pad_token] * padding_length

    # attention_mask = [1] * len(input_ids) + [0] * padding_length

    if padding_length <= 0:
        input_ids = input_ids[:max_length]
    else:
        if pad_on_left:
            input_ids = padding_id + input_ids
        else:
            input_ids = input_ids + padding_id

    return input_ids

class EmbeddingCache:
    def __init__(self, base_path, seed=-1):
        self.base_path = base_path
        with open(base_path + '_meta', 'r') as f:
            meta = json.load(f)
            self.dtype = np.dtype(meta['type'])
            self.total_number = meta['total_number']
            self.record_size = int(
                meta['embedding_size']) * self.dtype.itemsize + 4
        if seed >= 0:
            self.ix_array = np.random.RandomState(seed).permutation(
                self.total_number)
        else:
            self.ix_array = np.arange(self.total_number)
        self.f = None

    def open(self):
        self.f = open(self.base_path, 'rb')

    def close(self):
        self.f.close()

    def read_single_record(self):
        record_bytes = self.f.read(self.record_size)
        passage_len = int.from_bytes(record_bytes[:4], 'big')
        passage = np.frombuffer(record_bytes[4:], dtype=self.dtype)
        return passage_len, passage

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __getitem__(self, key):
        if key < 0 or key > self.total_number:
            raise IndexError(
                "Index {} is out of bound for cached embeddings of size {}".
                format(key, self.total_number))
        self.f.seek(key * self.record_size)
        return self.read_single_record()

    def __iter__(self):
        self.f.seek(0)
        for i in range(self.total_number):
            new_ix = self.ix_array[i]
            yield self.__getitem__(new_ix)

    def __len__(self):
        return self.total_number

def numbered_byte_file_generator(base_path, file_no, record_size):
    print(f"file number: {file_no}")
    for i in range(file_no):
        if os.path.exists('{}_split{}'.format(base_path, i)):
            with open('{}_split{}'.format(base_path, i), 'rb') as f:
                while True:
                    b = f.read(record_size)
                    if not b:
                        # eof
                        break
                    yield b

def tokenize_to_file(args, i, num_process, in_path, out_path, line_fn):

    tokenizer, _ = load_model(args.model_type + "_Passage", args.pretrained_passage_encoder)

    with open(in_path, 'r', encoding='utf-8') if in_path[-2:] != "gz" else gzip.open(in_path, 'rt', encoding='utf8') as in_f,\
            open('{}_split{}'.format(out_path, i), 'wb') as out_f:
        first_line = False # tsv with first line
        for idx, line in enumerate(in_f):
            if idx % num_process != i or first_line:
                first_line = False
                continue
            try:
                res = line_fn(args, line, tokenizer)
            except ValueError:
                print("Bad passage.")
            else:
                out_f.write(res)

def multi_file_process(args, num_process, in_path, out_path, line_fn):
    processes = []
    for i in range(num_process):
        p = Process(target=tokenize_to_file,
                    args=(
                        args,
                        i,
                        num_process,
                        in_path,
                        out_path,
                        line_fn,
                    ))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

def PassagePreprocessingFn(args, line, tokenizer, title = False):
    line = line.strip()
    ext = args.raw_collection_path[args.raw_collection_path.rfind("."):]
    passage = None
    if ext == ".jsonl":
        obj = json.loads(line)
        p_id = int(obj["id"])
        p_text = obj["text"]
        p_title = obj["title"]

        full_text = p_text[:args.max_doc_character]

        passage = tokenizer.encode(
            p_title, 
            text_pair=full_text,
            add_special_tokens=True,
            truncation=True,
            max_length=args.max_seq_length,
        )
    elif ext == ".tsv":
        try:
            line_arr = line.split('\t')
            p_id = int(line_arr[0])
            if title == True:
                p_text = line_arr[2].rstrip().replace(' [SEP] ', ' ') + ' ' + line_arr[1].rstrip()
            else:
                p_text = line_arr[1].rstrip()
        except IndexError:  # split error
            raise ValueError  # empty passage
        else:
            full_text = p_text[:args.max_doc_character]
            passage = tokenizer.encode(
                full_text,
                add_special_tokens=True,
                truncation=True,
                max_length=args.max_seq_length,
            )
            
    else:
        raise TypeError("Unrecognized file type")

    passage_len = min(len(passage), args.max_seq_length)
    input_id_b = pad_input_ids(passage, args.max_seq_length)

    return p_id.to_bytes(8,'big') + passage_len.to_bytes(4,'big') + np.array(input_id_b,np.int32).tobytes()

def QueryPreprocessingFn(args, line, tokenizer):
    line_arr = line.split('\t')
    q_id = int(line_arr[0])

    passage = tokenizer.encode(
        line_arr[1].rstrip(),
        add_special_tokens=True,
        truncation=True,
        max_length=args.max_query_length)
    passage_len = min(len(passage), args.max_query_length)
    input_id_b = pad_input_ids(passage, args.max_query_length)

    return q_id.to_bytes(8,'big') + passage_len.to_bytes(4,'big') + np.array(input_id_b,np.int32).tobytes()

def GetProcessingFn(args, query=False):
    def fn(vals, i):
        passage_len, passage = vals
        max_len = args.max_query_length if query else args.max_seq_length

        pad_len = max(0, max_len - passage_len)
        token_type_ids = ([0] if query else [1]) * passage_len + [0] * pad_len
        attention_mask = [1] * passage_len + [0] * pad_len

        passage_collection = [(i, passage, attention_mask, token_type_ids)]

        query2id_tensor = torch.tensor(
            [f[0] for f in passage_collection], dtype=torch.long)
        all_input_ids_a = torch.tensor(
            [f[1] for f in passage_collection], dtype=torch.int)
        all_attention_mask_a = torch.tensor(
            [f[2] for f in passage_collection], dtype=torch.bool)
        all_token_type_ids_a = torch.tensor(
            [f[3] for f in passage_collection], dtype=torch.uint8)

        dataset = TensorDataset(
            all_input_ids_a,
            all_attention_mask_a,
            all_token_type_ids_a,
            query2id_tensor)

        return [ts for ts in dataset]

    return fn

def GetTrainingDataProcessingFn(args, query_cache, passage_cache):
    def fn(line, i):
        line_arr = line.split('\t')
        qid = int(line_arr[0])
        pos_pid = int(line_arr[1])
        neg_pids = line_arr[2].split(',')
        neg_pids = [int(neg_pid) for neg_pid in neg_pids]

        all_input_ids_a = []
        all_attention_mask_a = []

        query_data = GetProcessingFn(
            args, query=True)(
            query_cache[qid], qid)[0]
        pos_data = GetProcessingFn(
            args, query=False)(
            passage_cache[pos_pid], pos_pid)[0]

        pos_label = torch.tensor(1, dtype=torch.long)
        neg_label = torch.tensor(0, dtype=torch.long)

        for neg_pid in neg_pids:
            neg_data = GetProcessingFn(
                args, query=False)(
                passage_cache[neg_pid], neg_pid)[0]
            yield (query_data[0], query_data[1], query_data[2], pos_data[0], pos_data[1], pos_data[2], pos_label)
            yield (query_data[0], query_data[1], query_data[2], neg_data[0], neg_data[1], neg_data[2], neg_label)

    return fn

def GetTripletTrainingDataProcessingFn(args, query_cache, passage_cache):
    def fn(line, i):
        line_arr = line.split('\t')
        qid = int(line_arr[0])
        pos_pid = int(line_arr[1])
        neg_pids = line_arr[2].split(',')
        neg_pids = [int(neg_pid) for neg_pid in neg_pids]

        all_input_ids_a = []
        all_attention_mask_a = []

        query_data = GetProcessingFn(
            args, query=True)(
            query_cache[qid], qid)[0]
        pos_data = GetProcessingFn(
            args, query=False)(
            passage_cache[pos_pid], pos_pid)[0]

        for neg_pid in neg_pids:
            neg_data = GetProcessingFn(
                args, query=False)(
                passage_cache[neg_pid], neg_pid)[0]
            yield (query_data[0], query_data[1], query_data[2], pos_data[0], pos_data[1], pos_data[2],
                   neg_data[0], neg_data[1], neg_data[2])

    return fn

def InferenceEmbeddingFromStreamDataLoader(
    args,
    model,
    train_dataloader,
    is_query_inference=True,
):
    # expect dataset from ReconstructTrainingSet
    results = {}
    eval_batch_size = max(1, args.n_gpu) * args.per_gpu_eval_batch_size

    # Inference!
    logger.info("***** Running ANN Embedding Inference *****")
    logger.info("Batch size = %d", eval_batch_size)

    embedding = []
    embedding2id = []

    if args.local_rank != -1:
        dist.barrier()
    model.eval()

    tmp_n = 0
    expect_per_block_passage_num = 2500000 # 54573064 38636512
    block_size = expect_per_block_passage_num // eval_batch_size # 1000
    block_id = 0
    total_write_passages = 0

    for bt_idx, batch in enumerate(tqdm(train_dataloader,
                    desc="Inferencing",
                    disable=args.disable_tqdm,
                    position=0,
                    leave=True)):

        # if bt_idx == 100:
        #     break

        #if batch[3][-1] <= 19999999:
        #    logger.info("Current {} ".format(batch[3][-1]))
        #    continue

        idxs = batch[3].detach().numpy()  # [#B]

        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0].long(),
                "attention_mask": batch[1].long()
            }
            embs = model(inputs["input_ids"], inputs["attention_mask"])

        embs = embs.detach().cpu().numpy()
    
        # check for multi chunk output for long sequence
        if len(embs.shape) == 3:
            for chunk_no in range(embs.shape[1]):
                embedding2id.append(idxs)
                embedding.append(embs[:, chunk_no, :])
        else:
            embedding2id.append(idxs)
            embedding.append(embs)
        
        tmp_n += 1
        if tmp_n % 500 == 0:
            logger.info("Have processed {} batches...".format(tmp_n))

        if tmp_n % block_size == 0:
            embedding = np.concatenate(embedding, axis=0)
            embedding2id = np.concatenate(embedding2id, axis=0)
            emb_block_path = os.path.join(args.embedded_output_path, "passage_emb_block_{}.pb".format(block_id))
            with open(emb_block_path, 'wb') as handle:
                pickle.dump(embedding, handle, protocol=4)
            embid_block_path = os.path.join(args.embedded_output_path, "passage_embid_block_{}.pb".format(block_id))
            with open(embid_block_path, 'wb') as handle:
                pickle.dump(embedding2id, handle, protocol=4)
            total_write_passages += len(embedding)
            block_id += 1

            logger.info("Have written {} passages...".format(total_write_passages))
            embedding = []
            embedding2id = []
            gc.collect()

    if len(embedding) > 0:   
        embedding = np.concatenate(embedding, axis=0)
        embedding2id = np.concatenate(embedding2id, axis=0)

        emb_block_path = os.path.join(args.embedded_output_path, "passage_emb_block_{}.pb".format(block_id))
        embid_block_path = os.path.join(args.embedded_output_path, "passage_embid_block_{}.pb".format(block_id))
        with open(emb_block_path, 'wb') as handle:
            pickle.dump(embedding, handle, protocol=4)
        with open(embid_block_path, 'wb') as handle:
            pickle.dump(embedding2id, handle, protocol=4)
        total_write_passages += len(embedding)
        block_id += 1

    logger.info("total write passages {}".format(total_write_passages))
    # return embedding, embedding2id

def StreamInferenceDoc(args,
                       model,
                       fn,
                       prefix,
                       f,
                       is_query_inference=True,
                       merge=True):
    inference_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    inference_dataset = StreamingDataset(f, fn)
    inference_dataloader = DataLoader(inference_dataset,
                                      batch_size=inference_batch_size)
    
    num_batches = len(inference_dataloader)
    logger.info(f'The number of batches: {num_batches}')

    if args.local_rank != -1:
        dist.barrier()  # directory created

    InferenceEmbeddingFromStreamDataLoader(
        args,
        model,
        inference_dataloader,
        is_query_inference=is_query_inference,
        )
    logger.info("merging embeddings")

def generate_new_ann(args):

    _, model = load_model(args.model_type + "_Passage", args.pretrained_passage_encoder)
    model = model.to(args.device)
    if args.n_gpu > 1:
        model = nn.DataParallel(model, device_ids = list(range(args.n_gpu)))

    merge = False

    logger.info("***** inference of passages *****")
    passage_collection_path = os.path.join(args.tokenized_output_path, "passages")
    passage_cache = EmbeddingCache(passage_collection_path)
    with passage_cache as emb:
        StreamInferenceDoc(
            args,
            model,
            GetProcessingFn(args, query=False),
            "passage_",
            emb,
            is_query_inference=False,
            merge=merge)
    logger.info("***** Done passage inference *****")


# === Main Functions ==========
def gen_tokenized_doc(args):
    pid2offset = {}
    offset2pid = []
    in_passage_path = args.raw_collection_path

    out_passage_path = os.path.join(
        args.tokenized_output_path,
        "passages",
    )

    if os.path.exists(out_passage_path):
        print("preprocessed data already exist, exit preprocessing")
        return

    out_line_count = 0

    print('start passage file split processing')
    multi_file_process(
        args,
        32,
        in_passage_path,
        out_passage_path,
        PassagePreprocessingFn)

    print('start merging splits')
    with open(out_passage_path, 'wb') as f:
        for idx, record in enumerate(numbered_byte_file_generator(
                out_passage_path, 32, 8 + 4 + args.max_seq_length * 4)):
            p_id = int.from_bytes(record[:8], 'big')
            f.write(record[8:])
            pid2offset[p_id] = idx
            offset2pid.append(p_id)
            if idx < 3:
                print(str(idx) + " " + str(p_id))
            out_line_count += 1

    print("Total lines written: " + str(out_line_count))
    meta = {
        'type': 'int32',
        'total_number': out_line_count,
        'embedding_size': args.max_seq_length}
    with open(out_passage_path + "_meta", 'w') as f:
        json.dump(meta, f)
    embedding_cache = EmbeddingCache(out_passage_path)
    print("First line")
    with embedding_cache as emb:
        print(emb[0])

    pid2offset_path = os.path.join(
        args.tokenized_output_path,
        "pid2offset.pickle",
    )
    offset2pid_path = os.path.join(
        args.tokenized_output_path,
        "offset2pid.pickle",
    )
    with open(pid2offset_path, 'wb') as handle:
        pickle.dump(pid2offset, handle, protocol=4)
    with open(offset2pid_path, "wb") as handle:
        pickle.dump(offset2pid, handle, protocol=4)
    print("done saving pid2offset")

def gen_doc_embeddings(args):
    logger.info("start generate ann data")
    generate_new_ann(args)

    if args.local_rank != -1:
        dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_collection_path", type=str, default=WIKI_FILE)
    parser.add_argument("--tokenized_output_path", type=str, default=TOKENIZED_DOC_DIR)
    parser.add_argument("--embedded_output_path", type=str, default=EMBEDDED_DOC_DIR)
    args = parser.parse_args()
    
    args.model_type = model_type
    args.pretrained_passage_encoder = pretrained_passage_encoder
    args.max_seq_length = max_seq_length
    args.max_doc_character = max_doc_character
    args.per_gpu_eval_batch_size = per_gpu_eval_batch_size
    args.local_rank = local_rank
    args.disable_tqdm = disable_tqdm
    args.n_gpu = n_gpu
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", args.device)
    
    os.makedirs(args.tokenized_output_path, exist_ok=True)
    os.makedirs(args.embedded_output_path, exist_ok=True)
    
    
    # gen_tokenized_doc(args)
    gen_doc_embeddings(args)
    
    # python component0_preprocessing/corpus_indexing/dense_corpus_indexing.py
    
    