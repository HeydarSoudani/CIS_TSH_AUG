import argparse
import os
import json
from tqdm import tqdm
import pickle
# from IPython import embed
import csv
import random
from sklearn.model_selection import train_test_split



id_col= 0
text_col= 1
title_col = 2


# # def gen_topiocqa_qrel(raw_dev_file_path, output_qrel_file_path):
# def gen_topiocqa_qrel():
#     '''
#     raw_dev_file_path = "gold_dev.json"
#     output_qrel_file_path = "topiocqa_qrel.trec"
#     '''
#     raw_dev_file_path = 'datasets/TopiOCQA/ir_all_history_dev.json'
#     output_qrel_file_path = 'component3_retriever/data/topiocqa/dev/qrel_gold.trec'
    
#     with open(raw_dev_file_path, "r") as f:
#         data = json.load(f)
    
#     with open(output_qrel_file_path, "w") as f:
#         for line in tqdm(data):
#             sample_id = "{}_{}".format(line["conv_id"], line["turn_id"])
#             for pos in line["positive_ctxs"]:
#                 #pid = int(pos["passage_id"]) - 1
#                 pid = int(pos["passage_id"])
#                 f.write("{} {} {} {}".format(sample_id, 0, pid, 1))
#                 f.write('\n')


# .tsv -> .jsonl
def convert_collection(collection_tsv, collection_json):
    with open(collection_tsv, 'r') as input, open(collection_json, 'w') as output:
        reader = csv.reader(input, delimiter="\t") # passage_nums = 25700592
        for i, row in enumerate(tqdm(reader)):
            if row[id_col] == "id":
                # ['id', 'text', 'title'] id from 1
                continue
            title = row[title_col]
            text = row[text_col]
            title = ' '.join(title.split(' [SEP] '))
            break
                #obj = {"contents": " ".join([title, text]), "id": f"doc{i}"} # doc10
                #output.write(json.dumps(obj, ensure_ascii=False) + '\n')

def load_collection(collection_file, title = False):
    all_passages = ["[INVALID DOC ID]"] * 5000_0000
    ext = collection_file[collection_file.rfind(".") + 1:]
    if ext not in ["jsonl", "tsv"]:
        raise TypeError("Unrecognized file type")
    print("begin load")
    with open(collection_file, "r") as f:
        if ext == "jsonl":
            for line in f:
                line = line.strip()
                obj = json.loads(line)
                pid = int(obj["id"][3:]) # remove "doc" keyword from doc_id 
                #passage = obj["title"] + "[SEP]" + obj["text"]
                passage = obj["title"] + obj["text"]
                all_passages[pid] = passage
        else:
            first_line = True
            for line in tqdm(f):
                if first_line:
                    first_line = False
                    continue
                line = line.strip()
                try:
                    line_arr = line.split("\t")
                    pid = int(line_arr[0])
                    if title == True:
                        passage = line_arr[2].rstrip().replace(' [SEP] ', ' ') + ' ' + line_arr[1].rstrip()
                    else:
                        passage = line_arr[1].rstrip()
                    all_passages[pid] = passage
                except IndexError:
                    print("bad passage")
                except ValueError:
                    print("bad pid")
    return all_passages

# combine original data and gold ir data for training
def combine_data_train(inputs, inputs_gold, inputs_rewrite, output, collection):
    with open(inputs, "r") as f, open(inputs_gold, "r") as gf, open(inputs_rewrite, "r") as rw, open(output, "w") as g:
        obj = json.load(f)
        obj_g = json.load(gf)
        obj_rw = json.load(rw)
        assert len(obj) == len(obj_g)
        assert len(obj) == len(obj_rw)
        total_nums = len(obj)
        all_passages = load_collection(collection)
        print("loading collection finish!")
        history_rewrite = []
        for i in range(total_nums):
            query = obj[i]["Question"]
            rewrite = obj_rw[i]["question"]
            answer = obj[i]["Answer"]
            conv_id = obj_g[i]["conv_id"]
            turn_id = obj_g[i]["turn_id"]
            history_query = []
            if int(turn_id) == 1:
                history_rewrite = []
                last_response = ""
            elif int(turn_id) > 1 and i > 0:
                history_rewrite.append(obj_rw[i - 1]["question"])
                last_response = ' '.join(obj_g[i - 1]["positive_ctxs"][0]["title"].split(' [SEP] ')) + ' ' + obj_g[i - 1]["positive_ctxs"][0]["text"]
            history_answer = []
            idx = 0
            for key in obj[i]["Context"]:
                if idx % 2 == 0:
                    history_query.append(key)
                else:
                    history_answer.append(key)
                idx += 1
            topic = obj[i]["Topic"]
            sub_topic = obj[i]["Topic_section"]
            rationale = obj[i]["Rationale"]
            #additional_answers = obj[i]["Additional_answers"] # only dev
            is_nq = obj[i]["is_nq"]
            pos_docs = []
            pos_docs_id = []
            pos_docs.append(' '.join(obj_g[i]["positive_ctxs"][0]["title"].split(' [SEP] ')) + ' ' + obj_g[i]["positive_ctxs"][0]["text"])
            pos_docs_id.append(int(obj_g[i]["positive_ctxs"][0]["passage_id"]))

            # random negatives
            neg_nums = 1
            neg_docs = []
            neg_docs_id = random.sample(range(0, 25700592), neg_nums)
            pos_id = pos_docs_id[0]
            if (pos_id - 1) in neg_docs_id:
                replace = True
                pos = pos_id - 1
                while replace:
                    neg_new = random.randint(0, 25700592)
                    neg_docs_id.remove(pos)
                    neg_docs_id.append(neg_new)
                    if neg_new != pos:
                        replace = False

            for j in range(len(neg_docs_id)):
                idx = neg_docs_id[j] + 1
                neg_docs.append(all_passages[idx])
            #print(len(neg_docs))
            #print(len(neg_docs_id))
            assert len(neg_docs) == len(neg_docs_id)

            # BM25 hard_neg
            hard_neg_docs = []
            hard_neg_docs_id = []
            
            g.write(
                    json.dumps({
                        "id": str(conv_id) + '_' + str(turn_id),
                        "conv_id": conv_id,
                        "turn_id": turn_id,
                        "is_nq": is_nq,
                        "query": query,
                        "rewrite": rewrite,
                        "answer": answer,
                        "history_query": history_query,
                        "history_rewrite": history_rewrite,
                        "history_answer": history_answer,
                        "last_response": last_response,
                        "topic": topic,
                        "sub_topic": sub_topic,
                        "pos_docs": pos_docs,
                        "pos_docs_id": pos_docs_id,
                        "neg_docs": neg_docs,
                        "neg_docs_id": neg_docs_id,
                        "hard_neg_docs": hard_neg_docs,
                        "hard_neg_docs_id": hard_neg_docs_id,
                    }) + "\n")
        print(total_nums)

def combine_data_test(inputs, inputs_gold, inputs_rewrite, output):
    with open(inputs, "r") as f, open(inputs_gold, "r") as gf, open(inputs_rewrite, "r") as rw, open(output, "w") as g:
        obj = json.load(f)
        obj_g = json.load(gf)
        total_nums = len(obj)
        obj_rw = json.load(rw)
        assert len(obj) == len(obj_g)
        assert len(obj) == len(obj_rw)
        history_rewrite = []
        for i in range(total_nums):
            query = obj[i]["Question"]
            rewrite = obj_rw[i]["question"]
            answer = obj[i]["Answer"]
            conv_id = obj_g[i]["conv_id"]
            turn_id = obj_g[i]["turn_id"]
            history_query = []
            if int(turn_id) == 1:
                history_rewrite = []
                last_response = ""
            elif int(turn_id) > 1 and i > 0:
                history_rewrite.append(obj_rw[i - 1]["question"])
                last_response = ' '.join(obj_g[i - 1]["positive_ctxs"][0]["title"].split(' [SEP] ')) + ' ' + obj_g[i - 1]["positive_ctxs"][0]["text"]

            history_answer = []
            idx = 0
            for key in obj[i]["Context"]:
                if idx % 2 == 0:
                    history_query.append(key)
                else:
                    history_answer.append(key)
                idx += 1
            topic = obj[i]["Topic"]
            sub_topic = obj[i]["Topic_section"]
            rationale = obj[i]["Rationale"]
            additional_answers = obj[i]["Additional_answers"] # only test
            is_nq = obj[i]["is_nq"]
            pos_docs = []
            pos_docs_id = []
            pos_docs.append(' '.join(obj_g[i]["positive_ctxs"][0]["title"].split(' [SEP] ')) + ' ' + obj_g[i]["positive_ctxs"][0]["text"])
            pos_docs_id.append(int(obj_g[i]["positive_ctxs"][0]["passage_id"]))

            g.write(
                    json.dumps({
                        "id": str(conv_id) + '_' + str(turn_id),
                        "conv_id": conv_id,
                        "turn_id": turn_id,
                        "is_nq": is_nq,
                        "query": query,
                        "rewrite": rewrite, 
                        "answer": answer,
                        "history_query": history_query,
                        "history_rewrite": history_rewrite,
                        "history_answer": history_answer,
                        "last_response": last_response,
                        "topic": topic,
                        "sub_topic": sub_topic,
                        #"rationale": rationale,
                        #"additional_answers": additional_answers, # "Topic", "Topic_section"
                        "pos_docs": pos_docs,
                        "pos_docs_id": pos_docs_id,
                    }) + "\n")
        print(total_nums)

def convert_gold_to_trec(gold_file, trec_file):
    with open(gold_file, "r") as f, open(trec_file, "w") as g:
        data = f.readlines()
        for line in data:
            line = json.loads(line)
            qid = line["id"]
            #query = line["query"]
            doc_id = line["pos_docs_id"][0]
            g.write("{} {} {} {}".format(qid,
                                        "Q0",
                                        doc_id,
                                        1,
                                        ))
            g.write('\n')

def test_file_to_qrecc_format(input_file, output_file):
    with open(input_file, 'r') as file, open(output_file, "w") as of:
        for line in file:
            x = json.loads(line.strip())
            
            item = {
                "sample_id": x["id"],
                "cur_utt_text": x["query"],
                "oracle_utt_text": x["rewrite"],
                "cur_response_text": x["answer"],
                "ctx_utts_text": x["history_query"],
                "ctx_resps_text": x["history_answer"],
                "pos_docs_pids": x["pos_docs_id"]
            }
            of.write(json.dumps(item) + '\n')
            

if __name__ == "__main__":
    
    # ===== Step 1) download files ==============
    # wget https://zenodo.org/records/6149599/files/data/wikipedia_split/full_wiki_segments.tsv -O corpus/TopiOCQA/full_collection_segments.tsv
    # wget https://zenodo.org/records/7709644/files/topiocqa_train.json -O corpus/TopiOCQA/topiocqa_train.json
    # wget https://zenodo.org/records/6151011/files/data/retriever/all_history/train.json -O corpus/TopiOCQA/ir_all_history_train.json
    # wget https://zenodo.org/records/6151011/files/data/retriever/rewrites_t5_qrecc/train.json -O corpus/TopiOCQA/ir_rewrite_train.json
    # wget https://zenodo.org/records/7709644/files/topiocqa_dev.json -O corpus/TopiOCQA/topiocqa_dev.json
    # wget https://zenodo.org/records/6151011/files/data/retriever/all_history/dev.json -O corpus/TopiOCQA/ir_all_history_dev.json
    # wget https://zenodo.org/records/6151011/files/data/retriever/rewrites_t5_qrecc/dev.json -O corpus/TopiOCQA/ir_rewrite_dev.json
    
    # ===== Step 2) Process files ===============
    # = Input files ======
    collection_tsv = "corpus/TopiOCQA/full_wiki_segments.tsv"
    train = "corpus/TopiOCQA/topiocqa_train.json"
    dev = "corpus/TopiOCQA/topiocqa_dev.json"
    train_gold = "processed_datasets/TopiOCQA/ir_all_history_train.json"
    dev_gold = "corpus/TopiOCQA/ir_all_history_dev.json"
    train_rewrite = "processed_datasets/TopiOCQA/ir_rewrite_train.json"
    dev_rewrite = "corpus/TopiOCQA/ir_rewrite_dev.json"
    # = Output files =====
    collection_json = "corpus/TopiOCQA/full_collection_segments.jsonl"
    train_new = "processed_datasets/TopiOCQA/train_new.json"
    dev_new = "processed_datasets/TopiOCQA/dev_new.json"
    train_trec_gold = "processed_datasets/TopiOCQA/train_gold.trec"
    dev_trec_gold = "processed_datasets/TopiOCQA/dev_gold.trec"
    
    # convert_collection(collection_tsv, collection_json)
    combine_data_train(train, train_gold, train_rewrite, train_new, collection_tsv)
    # convert_gold_to_trec(train_new, train_trec_gold)
    # combine_data_test(dev, dev_gold, dev_rewrite, dev_new)
    # convert_gold_to_trec(dev_new, dev_trec_gold)
    
    # dev_qrecc_format = "processed_datasets/TopiOCQA/dev_qrecc_format.json"
    # test_file_to_qrecc_format(dev_new, dev_qrecc_format)
    
# python component0_preprocessing/topiocqa_files_preprocessing/1_corpus_preprocessing.py

