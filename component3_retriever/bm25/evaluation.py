### ==============================================================================
# Ref: https://github.com/fengranMark/HAConvDR/blob/main/bm25/bm25_topiocqa.py
### ==============================================================================

import torch
import random
import numpy as np
from tqdm import tqdm, trange
import argparse, logging, os, json
# from utils import str2bool
from pyserini.search.lucene import LuceneSearcher


logging.basicConfig(level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    handlers=[logging.StreamHandler()]
)
os.environ["WANDB_MODE"] = "offline"
print("Available GPUs:", torch.cuda.device_count())
device = 'cuda:0'

subset_percentage = 0.1

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# def gen_topiocqa_qrel(raw_dev_file_path, output_qrel_file_path):
def gen_topiocqa_qrel():
    '''
    raw_dev_file_path = "gold_dev.json"
    output_qrel_file_path = "topiocqa_qrel.trec"
    '''
    raw_dev_file_path = 'datasets/TopiOCQA/ir_all_history_dev.json'
    output_qrel_file_path = 'component3_retriever/data/topiocqa/dev/qrel_gold.trec'
    
    with open(raw_dev_file_path, "r") as f:
        data = json.load(f)
    
    with open(output_qrel_file_path, "w") as f:
        for line in tqdm(data):
            sample_id = "{}_{}".format(line["conv_id"], line["turn_id"])
            for pos in line["positive_ctxs"]:
                #pid = int(pos["passage_id"]) - 1
                pid = int(pos["passage_id"])
                f.write("{} {} {} {}".format(sample_id, 0, pid, 1))
                f.write('\n')



def main(args):
    
    # === Create output directory ============
    os.makedirs(args.output_dir_path, exist_ok=True)
    
    
    # === Read query file ====================
    # TODO: Create the different versions of query file
    query_path = f"component3_retriever/data/{args.dataset_name}/dev/{args.query_format}.jsonl"
    queries = {}
    with open (query_path, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            queries[data['id']] = data

    # = Select a subset of queries ===========
    if subset_percentage != 1.0:
        subset_size = int(len(queries) * subset_percentage)
        subset_keys = random.sample(list(queries.keys()), subset_size)
        subset_queries = {key: queries[key] for key in subset_keys}
    else:
        subset_queries = queries

    qid_list = list(subset_queries.keys())
    query_list = [subset_queries[qid]['query'] for qid in qid_list]
    
    
    # === Retriever Model: pyserini search ===
    searcher = LuceneSearcher(args.index_dir_path)
    searcher.set_bm25(args.bm25_k1, args.bm25_b)
    hits = searcher.batch_search(query_list, qid_list, k=args.top_k, threads=20)

    total = 0
    with open(os.path.join(args.output_dir_path, "dev_bm25_res.trec"), "w") as f:
        for qid in qid_list:
            for i, item in enumerate(hits[qid]):
                f.write("{} {} {} {} {} {} {}".format(qid,
                                                "Q0",
                                                item.docid[3:],
                                                i+1,
                                                -i - 1 + 200,
                                                item.score,
                                                "bm25"
                                                ))
                f.write('\n')
                total += 1
    print(total)
    
    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir_path", type=str, required=True)
    parser.add_argument("--output_dir_path", type=str, required=True)
    parser.add_argument("--gold_qrel_file_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="topiocqa", choices=["topiocqa", "inscit", "qrecc"])
    parser.add_argument("--query_format", type=str, default="original", choices=['original', 'human_rewritten', 'all_history', 'same_topic'])
    
    parser.add_argument("--bm25_k1", type=float, default="0.9")
    parser.add_argument("--bm25_b", type=float, default="0.4")
    parser.add_argument("--top_k", type=int, default="100")
    parser.add_argument("--rel_threshold", type=int, default="1")
    
    parser.add_argument("--seed", type=int)
    
    args = parser.parse_args()
    # gen_topiocqa_qrel()
    main(args)