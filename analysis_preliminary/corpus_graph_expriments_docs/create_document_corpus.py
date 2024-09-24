#!/usr/bin/env python3

import re
import torch
import random
import pytrec_eval
import numpy as np
from tqdm import tqdm
import argparse, json, logging, os
from transformers import AutoTokenizer
from pyserini.search.lucene import LuceneSearcher

# from org.apache.lucene.search import BooleanQuery
# BooleanQuery.setMaxClauseCount(10000)  # Set this to a larger value, e.g., 10000

logging.basicConfig(level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    handlers=[logging.StreamHandler()]
)
os.environ["WANDB_MODE"] = "offline"
subset_percentage = 1.0

more_than_1024 = 0


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def truncate_text(text, max_token_num, model_name="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_text = tokenizer.encode(text, truncation=False)
    if len(tokenized_text) > 1024:
        tokenized_text += 1
    
    truncated_tokenized_text = tokenized_text[:max_token_num]
    truncated_text = tokenizer.decode(truncated_tokenized_text, skip_special_tokens=True)
    return truncated_text

def remove_links(text):
    cleaned_text = re.sub(r'<a href="[^"]+">(.*?)</a>', r'\1', text)
    return cleaned_text

def create_document_corpus(wiki_file, output_file):
    all_context = {}
    doc_id_counter = 1
    with open(wiki_file, 'r') as in_file:
        for idx, line in tqdm(enumerate(in_file), desc="Reading wiki ..."):
            item = json.loads(line)
            
            # if idx == 10000:
            #     break
            
            doc_title = item["title"].split('##')[0]
            doc_subtitle = item["title"].split('##')[1]
            doc_context = remove_links(item['text'])
            
            if doc_title not in all_context:
                all_context[doc_title] = {"id": f"doc{str(doc_id_counter)}", "contents": ""}
                doc_id_counter += 1
            all_context[doc_title]["contents"] += f"{doc_context} "
            
    with open(output_file, 'w') as out_file:
        for doc_title, doc_context in tqdm(all_context.items(), desc="Writing wiki ..."):
            record = {
                "id": doc_context['id'],
                "topic": doc_title,
                "contents": doc_context["contents"]
            }
            json_str = json.dumps(record)
            out_file.write(json_str + '\n')
    
def create_document_train_dev(input_file, output_file, doc_corpus_file):
    
    def build_corpus_index(corpus_file):
        """Builds a dictionary for fast topic lookups."""
        corpus_index = {}
        with open(corpus_file, 'r') as corp_file:
            for line in corp_file:
                item = json.loads(line)
                topic = item["topic"]
                if topic not in corpus_index:
                    corpus_index[topic] = {
                        "id": item["id"],
                        "contents": item["contents"]
                    }
        return corpus_index
    
    print("Corpus indexing ...")
    corpus_index = build_corpus_index(doc_corpus_file)

    with open(input_file, 'r') as in_file:
        passage_based_data = json.load(in_file)

    output_data = {}
    for idx, turn in tqdm(enumerate(passage_based_data), desc="Reading turns ..."):
        # if idx == 10:
        #     break
        topic = turn["Topic"]
        if topic not in output_data and topic in corpus_index:
            conv_no = turn["Conversation_no"]
            id = corpus_index[topic]["id"]
            context = corpus_index[topic]["contents"]

            output_data[topic] = {
                "conv_no": conv_no,
                "id": id,
                "contents": context
            }  
            
    with open(output_file, 'w') as out_file:
        for topic, topic_context in tqdm(output_data.items(), desc="Writing docs ..."):   
            record = {
                "conv_no": topic_context["conv_no"],
                "id": topic_context['id'],
                "topic": topic,
                "contents": topic_context["contents"]
            }
            json_str = json.dumps(record)
            out_file.write(json_str + '\n')
 
def document_retriever_pyserini(args):
    print("Preprocessing files ...")
    index_dir = "corpus/TopiOCQA/doc_corpus_bm25_index"
    
    all_queries = 0
    queries = {}
    query_file = "processed_datasets/TopiOCQA/corpus_doc/topiocqa_doc_dev.jsonl"
    with open (query_file, 'r') as file:
        for line in file:
            all_queries += 1
            item = json.loads(line.strip())
            # queries[item['id']] = truncate_text(item["context"], 1024)
            queries[item['id']] = item["context"]
    
    print(f"# More than 1024: {more_than_1024}")
    print(f"# All data: {all_queries}")
    
    # Select a subset of queries
    if subset_percentage != 1.0:
        subset_size = int(len(queries) * subset_percentage)
        subset_keys = random.sample(list(queries.keys()), subset_size)
        subset_queries = {key: queries[key] for key in subset_keys}
    else:
        subset_queries = queries

    qid_list = list(subset_queries.keys())
    query_list = [subset_queries[qid] for qid in qid_list]
    print(f"Query_id: {qid_list[1]}\nQuery: {query_list[1]}\n")

    print(f"Retrieving using {args.retriever_model} ...")
    if args.retriever_model == "bm25":
        searcher = LuceneSearcher(index_dir)
        searcher.set_bm25(args.bm25_k1, args.bm25_b)
        hits = searcher.batch_search(query_list, qid_list, k=args.top_k, threads=20)
    
    ### === Write to output file =====================================
    print("Writing to output file ...")
    os.makedirs(args.results_base_path, exist_ok=True)
    output_res_file = f"{args.results_base_path}/{args.query_format}_{args.retriever_model}_results.trec"
    
    with open(output_res_file, "w") as f:
        for qid in qid_list:
            for i, item in enumerate(hits[qid]):
                result_line = f"{qid[3:]} Q0 {item.docid[3:]} {i+1} {item.score} {args.retriever_model}"
                f.write(result_line)
                f.write('\n')
    
    print("Done!")

def create_gold_trec_file(args):
    gold_trec_file = f"processed_datasets/TopiOCQA/corpus_doc/gold_trec_{args.dataset_name}_{args.dataset_subsec}_doc.trec"

    all_topics = []
    doc_based_file = f"corpus/TopiOCQA/topiocqa_doc_{subsec}.jsonl"
    with open(doc_based_file, 'r') as in_file:
        for line in in_file:
            turn_obj = json.loads(line)
            all_topics.append(turn_obj)
 
    with open(gold_trec_file, 'w') as out_file:
        for idx, turn in tqdm(enumerate(all_topics)):
            cur_conv = turn["conv_no"]
            if idx < len(all_topics)-1:
                nxt_conv = all_topics[idx+1]["conv_no"]
            
                if cur_conv == nxt_conv:
                    cur_turn = turn["id"][3:]
                    nxt_turn = all_topics[idx+1]["id"][3:]
                    out_file.write(f"{cur_turn} Q0 {nxt_turn} 1\n")

def retriever_eval(args):
    all_gold_trec_file = "processed_datasets/TopiOCQA/corpus_doc/gold_trec_TopiOCQA_dev_doc.trec"
    # first_gold_trec_file = f"analysis_preliminary/corpus_graph_expriments/gold_trec_first_{args.dataset_name}_{args.dataset_subsec}.trec"
    # concentrated_gold_trec_file = f"analysis_preliminary/corpus_graph_expriments/gold_trec_concentrated_{args.dataset_name}_{args.dataset_subsec}.trec"
    # shifted_gold_trec_file = f"analysis_preliminary/corpus_graph_expriments/gold_trec_shifted_{args.dataset_name}_{args.dataset_subsec}.trec"
    with open(all_gold_trec_file, 'r') as f:
        qrel_data = f.readlines()
    
    qrels = {}
    qrels_ndcg = {}
    query_id = []
    for line in qrel_data:
        line = line.strip().split()
        query = line[0]
        passage = line[2]
        rel = int(line[3])
        if query not in qrels:
            qrels[query] = {}
            query_id.append(query)
        if query not in qrels_ndcg:
            qrels_ndcg[query] = {}

        # for NDCG
        qrels_ndcg[query][passage] = rel
        # for MAP, MRR, Recall
        if rel >= args.rel_threshold:
            rel = 1
        else:
            rel = 0
        qrels[query][passage] = rel
    
    runs = {}
    results_file = "processed_datasets/TopiOCQA/corpus_doc/original_bm25_results.trec"
    with open(results_file, 'r') as f:
        run_data = f.readlines()
    for line in run_data:
        line = line.split(" ")
        query = line[0]
        passage = line[2]
        rel = int(float(line[4]))
        if query not in runs:
            runs[query] = {}
        runs[query][passage] = rel

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "recip_rank", "recall.5", "recall.10", "recall.100", "recall.1000"})
    res = evaluator.evaluate(runs)
    map_list = [v['map'] for v in res.values()]
    mrr_list = [v['recip_rank'] for v in res.values()]
    recall_1000_list = [v['recall_1000'] for v in res.values()]
    recall_100_list = [v['recall_100'] for v in res.values()]
    recall_10_list = [v['recall_10'] for v in res.values()]
    recall_5_list = [v['recall_5'] for v in res.values()]

    evaluator = pytrec_eval.RelevanceEvaluator(qrels_ndcg, {"ndcg_cut.3"})
    res = evaluator.evaluate(runs)
    ndcg_3_list = [v['ndcg_cut_3'] for v in res.values()]
    res = {
            "MAP": np.average(map_list),
            "MRR": np.average(mrr_list),
            "NDCG@3": np.average(ndcg_3_list),
            "Recall@5": np.average(recall_5_list),
            "Recall@10": np.average(recall_10_list),
            "Recall@100": np.average(recall_100_list), 
            "Recall@1000": np.average(recall_1000_list),
            
        }
    print("---------------------Evaluation results:---------------------")    
    print(res)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever_model", type=str, default="bm25", choices=["bm25", "ance"])
    parser.add_argument("--query_encoder", type=str, default="castorini/ance-msmarco-passage")
    parser.add_argument("--index_dir_base_path", type=str, default="corpus")
    parser.add_argument("--results_base_path", type=str, default="processed_datasets/TopiOCQA/corpus_doc")
    parser.add_argument("--dataset_name", type=str, default="TopiOCQA", choices=["TopiOCQA", "INSCIT"])
    parser.add_argument("--dataset_subsec", type=str, default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--query_format", type=str, default='original', choices=[
        'original', 'human_rewritten', 'all_history', 'same_topic', 't5_rewritten',
    ])
    parser.add_argument("--bm25_k1", type=float, default="0.9")
    parser.add_argument("--bm25_b", type=float, default="0.4")
    parser.add_argument("--top_k", type=int, default="1000")
    parser.add_argument("--rel_threshold", type=int, default="1")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    print("Available GPUs:", torch.cuda.device_count())
    args.device = 'cuda:0'
    set_seed(args.seed)
    
    ### === Create doc-based corpus =========== 
    # wget https://zenodo.org/records/6173228/files/data/wikipedia_split/full_wiki.jsonl -O corpus/TopiOCQA/full_wiki.jsonl
    wiki_file = "corpus/TopiOCQA/full_wiki.jsonl"
    wiki_document_corpus_file = "corpus/TopiOCQA/doc_corpus/wiki_document_corpus.jsonl"
    # create_document_corpus(wiki_file, wiki_document_corpus_file)
    # Indexing ====
    # python -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator -threads 20 -input "corpus/TopiOCQA/doc_corpus" -index "corpus/TopiOCQA/doc_corpus_bm25_index" -storePositions -storeDocvectors -storeRaw
    
    ### === Create doc-based train dev ========
    subsec = "dev"
    passage_based_file = f"corpus/TopiOCQA/topiocqa_{subsec}.json"
    doc_based_file = f"corpus/TopiOCQA/topiocqa_doc_{subsec}.jsonl"
    # create_document_train_dev(passage_based_file, doc_based_file, wiki_document_corpus_file)
    
    ### === retreival step ==================== 
    document_retriever_pyserini(args)
    
    ### === Evaluation ========================
    # create_gold_trec_file(args)
    # retriever_eval(args)
    
    
    
    
    
    
    # python component0_preprocessing/topiocqa_files_preprocessing/3_create_document_corpus.py
    
    
    
    # with open(wiki_document_corpus_file, 'r') as in_file, open("corpus/TopiOCQA/doc_corpus/wiki_document_corpus_new.jsonl", 'w') as out_file:
    #     for idx, line in tqdm(enumerate(in_file), desc="Reading wiki ..."):
    #         item = json.loads(line)
            
    #         new_item = {
    #             "id": item['id'],
    #             "topic": item['title'],
    #             "contents": item["context"]
    #         }
    #         json_str = json.dumps(new_item)
    #         out_file.write(json_str + '\n')
    