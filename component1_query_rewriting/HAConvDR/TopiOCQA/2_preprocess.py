#!/usr/bin/env python3

import csv
import json
import random
from tqdm import tqdm, trange


input_base_path = "processed_datasets/TopiOCQA"
output_base_path = "processed_datasets/TopiOCQA/for_HAConvDR"

train_new = f"{input_base_path}/train_new.json"
dev_new = f"{input_base_path}/dev_new.json"

train_trec_gold = f"{output_base_path}/train_gold.trec"
dev_trec_gold = f"{output_base_path}/dev_gold.trec"
train_rel = f"{output_base_path}/train_rel_1.json"
dev_rel = f"{output_base_path}/dev_rel.json"
train_rel_gold = f"{output_base_path}/train_rel_gold_1.trec"
dev_rel_gold = f"{output_base_path}/dev_rel_gold.trec"


id_col= 0
text_col= 1
title_col = 2

def merge_rel_label_info(rel_file, orig_file, new_file):
    # rel_file: train/dev_rel_label_rawq.json
    # orig_file: train/test.json
    # new_file: train/test_with_gold_rel.json
    with open(rel_file, "r") as f:
        rel_labels = f.readlines()

    with open(orig_file, 'r') as f, open(new_file, 'w') as g:
        lines = f.readlines()
        for i in range(len(lines)):
            line_dict = json.loads(lines[i])
            sample_id = line_dict['sample_id']
            if sample_id.split('-')[-1] != '1':
                assert sample_id == json.loads(rel_labels[i])['id']
                rel_label = json.loads(rel_labels[i])['rel_label']
                line_dict['rel_label'] = rel_label
            else:
                line_dict['rel_label'] = []
            json.dump(line_dict, g)
            g.write('\n')

def create_label_rel_turn(inputs, output):
    with open(inputs, "r") as f, open(output, "w") as g:
        obj = f.readlines()
        total_nums = len(obj)
        query_pair_nums = 0
        for i in range(total_nums):
            obj[i] = json.loads(obj[i])
            sample_id = obj[i]['id']
            conv_id = obj[i]['conv_id']
            turn_id = obj[i]['turn_id']
            history_query = obj[i]["history_query"]
            history_rewrite = obj[i]["history_rewrite"]
            history_answer = obj[i]["history_answer"]
            last_response = obj[i]["last_response"]
            topic = obj[i]["topic"]
            sub_topic = obj[i]["sub_topic"]
            query = obj[i]["query"]
            rewrite = obj[i]["rewrite"]
            answer = obj[i]["answer"]
            pos_docs = obj[i]["pos_docs"]
            pos_docs_id = obj[i]["pos_docs_id"]

            if int(turn_id) > 1: # if first turn
                g.write(
                    json.dumps({
                        "id": str(conv_id) + '-' + str(turn_id) + '-0',
                        "conv_id": conv_id,
                        "turn_id": turn_id,
                        "query": query,
                        "rewrite": rewrite,
                        "query_pair": "",
                        "rewrite_query_pair": "",
                        "history_answer": history_answer,
                        "last_response": last_response,
                        "topic": topic,
                        "sub_topic": sub_topic,
                        "pos_docs": pos_docs,
                        "pos_docs_id": pos_docs_id,
                    }) + "\n")
                query_pair_nums += 1

                for tid in range(0, int(turn_id) - 1):
                    query_pair = history_query[tid]
                    rewrite_query_pair = history_rewrite[tid]
                    #turn_pair_id = str(turn_id) + '-' + str(tid + 1)
                    g.write(
                        json.dumps({
                            "id": str(conv_id) + '-' + str(turn_id) + '-' + str(tid + 1),
                            "conv_id": conv_id,
                            "turn_id": turn_id,
                            "query": query,
                            "rewrite": rewrite,
                            "query_pair": query_pair,
                            "rewrite_query_pair": rewrite_query_pair,
                            "history_answer": history_answer,
                            "last_response": last_response,
                            "topic": topic,
                            "sub_topic": sub_topic,
                            "pos_docs": pos_docs,
                            "pos_docs_id": pos_docs_id,
                        }) + "\n")
                    query_pair_nums += 1
        print(query_pair_nums)

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

def create_PRJ(label_file, query_file, output):
    with open(label_file, "r") as f1, open(query_file, "r") as f2, open(output, "w") as g:
        obj_1 = f1.readlines()
        obj_2 = f2.readlines()
        one = 0
        zero = 0
        total_nums = len(obj_1)
        assert len(obj_1) == len(obj_2)
        for i in range(total_nums):
            obj_1[i] = json.loads(obj_1[i])
            obj_2[i] = json.loads(obj_2[i])
            sample_id = obj_1[i]['id']
            conv_id = obj_1[i]['conv_id']
            turn_id = obj_1[i]['turn_id']
            rel_label = obj_1[i]['rel_label']
            cur_query = obj_2[i]['query']
            history_query = obj_2[i]["history_query"]
            last_response = obj_2[i]["last_response"]
            assert len(history_query) == len(rel_label)
            if len(history_query) > 0:
                for idx in range(len(history_query)):
                    if rel_label[idx] == 1:
                        one += 1
                    else:
                        zero += 1
                    g.write(
                        json.dumps({
                            "id": sample_id + '-' + str(idx + 1),
                            "query": cur_query,
                            "rel_query": history_query[idx],
                            "rel_label": rel_label[idx],
                            #"last_response": last_response
                        }) + "\n")
        print("one", one)
        print("zero", zero)

if __name__ == "__main__":
    
    merge_rel_label_info(train_rel_file, output_train_file_path, train_new_file)
    merge_rel_label_info(test_rel_file, output_test_file_path, test_new_file)

    
    # create_label_rel_turn(train_new, train_rel)
    # create_label_rel_turn(dev_new, dev_rel)
    # convert_gold_to_trec(train_rel, train_rel_gold)
    # convert_gold_to_trec(dev_rel, dev_rel_gold)
    create_PRJ(f"{output_base_path}/train_rel_label.json", train_new, f"{output_base_path}/topiocqa_train_qp_1.json")
    create_PRJ(f"{output_base_path}/dev_rel_label.json", dev_new, f"{output_base_path}/topiocqa_dev_qp_1.json")
   
    
    
    # python component1_query_rewriting/HAConvDR/TopiOCQA/1_preprocess.py