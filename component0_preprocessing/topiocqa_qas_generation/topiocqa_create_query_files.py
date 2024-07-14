import json, os


# === Input files ==============
original = "datasets/topiocqa/topiocqa_{}.json"
rewrite = "datasets/topiocqa/ir_rewrite_{}.json"
all_history = "datasets/topiocqa/ir_all_history_{}.json" # contains pos passage id

# === Output files ==============
dataset_name = "topiocqa" # ["topiocqa", "inscit", "qrecc"]
output_base_dir = f"component3_retriever/data/{dataset_name}"
output_train_dir = f"{output_base_dir}/train"
output_test_dir = f"{output_base_dir}/dev"
os.makedirs(output_train_dir, exist_ok=True)
os.makedirs(output_test_dir, exist_ok=True)

original_train_output = f"{output_train_dir}/original.jsonl"
original_test_output = f"{output_test_dir}/original.jsonl"
human_rewritten_train_output = f"{output_train_dir}/human_rewritten.jsonl"
human_rewritten_test_output = f"{output_test_dir}/human_rewritten.jsonl"
all_history_train_output = f"{output_train_dir}/all_history.jsonl"
all_history_test_output = f"{output_test_dir}/all_history.jsonl"
same_topic_train_output = f"{output_train_dir}/same_topic.jsonl"
same_topic_test_output = f"{output_test_dir}/same_topic.jsonl"


def get_passage_information(subsection="train"):
    passage_info = {}
    with open(all_history.format(subsection), "r") as in_file:
        data = json.load(in_file)
    
    for item in data:
        conv_id = item["conv_id"]
        turn_id = item["turn_id"]
        passage_info[f"{conv_id}_{turn_id}"] = item
    
    return passage_info

def prepare_quries_original():
    
    ### === For train data =======
    subsec = "train"
    passage_info_train = get_passage_information(subsec)
    with open(original.format(subsec), "r") as in_file:
        data = json.load(in_file)
    
    with open(original_train_output, "w") as out_file:
        for item in data:
            conv_id = item['Conversation_no']
            turn_id = item['Turn_no']
            query_id = f"{conv_id}_{turn_id}"
            query = item['Question']
            answer = item['Answer']
            is_nq = item['is_nq']
            title = f"{item['Topic']} [SEP] {item['Topic_section']}"
            passage_id = passage_info_train[query_id]['positive_ctxs'][0]['passage_id']
            passage_text = passage_info_train[query_id]['positive_ctxs'][0]['text']
            
            out_file.write(json.dumps({
                "id": query_id,
                "conv_id": conv_id,
                "turn_id": turn_id,
                "is_nq": is_nq,
                "title": title,
                "query": query,
                "answer": answer,
                "passage_id": passage_id,
                "passage_text": passage_text    
            }) + "\n")
    

    ### === For dev data ==========
    subsec = "dev"
    passage_info_test = get_passage_information(subsec)
    with open(original.format(subsec), "r") as in_file:
        data = json.load(in_file)
    with open(original_test_output, "w") as out_file:
        for item in data:
            conv_id = item['Conversation_no']
            turn_id = item['Turn_no']
            query_id = f"{conv_id}_{turn_id}"
            query = item['Question']
            answer = item['Answer']
            is_nq = item['is_nq']
            title = f"{item['Topic']} [SEP] {item['Topic_section']}"
            passage_id = passage_info_test[query_id]['positive_ctxs'][0]['passage_id']
            passage_text = passage_info_test[query_id]['positive_ctxs'][0]['text']
            
            out_file.write(json.dumps({
                "id": query_id,
                "conv_id": conv_id,
                "turn_id": turn_id,
                "is_nq": is_nq,
                "title": title,
                "query": query,
                "answer": answer,
                "passage_id": passage_id,
                "passage_text": passage_text    
            }) + "\n")
     
def prepare_quries_human_rewritten():
    
    ### === For train data =======
    subsec = "train"
    passage_info_train = get_passage_information(subsec)
    with open(rewrite.format(subsec), "r") as in_file:
        data = json.load(in_file)
    
    with open(human_rewritten_train_output, "w") as out_file:
        for item in data:
            conv_id = item['conv_id']
            turn_id = item['turn_id']
            query_id = f"{conv_id}_{turn_id}"
            query = item['question']
            answer = item['short_answers'][0]
            title = item['title']
            passage_id = passage_info_train[query_id]['positive_ctxs'][0]['passage_id']
            passage_text = passage_info_train[query_id]['positive_ctxs'][0]['text']
            
            out_file.write(json.dumps({
                "id": query_id,
                "conv_id": conv_id,
                "turn_id": turn_id,
                "title": title,
                "query": query,
                "answer": answer,
                "passage_id": passage_id,
                "passage_text": passage_text    
            }) + "\n")
    
    
    ### === For dev data =======
    subsec = "dev"
    passage_info_test = get_passage_information(subsec)
    with open(rewrite.format(subsec), "r") as in_file:
        data = json.load(in_file)
    
    with open(human_rewritten_test_output, "w") as out_file:
        for item in data:
            conv_id = item['conv_id']
            turn_id = item['turn_id']
            query_id = f"{conv_id}_{turn_id}"
            query = item['question']
            answer = item['short_answers'][0]
            title = item['title']
            passage_id = passage_info_test[query_id]['positive_ctxs'][0]['passage_id']
            passage_text = passage_info_test[query_id]['positive_ctxs'][0]['text']
            
            out_file.write(json.dumps({
                "id": query_id,
                "conv_id": conv_id,
                "turn_id": turn_id,
                "title": title,
                "query": query,
                "answer": answer,
                "passage_id": passage_id,
                "passage_text": passage_text    
            }) + "\n")
    
def prepare_quries_all_history():
    ### === For train data =========
    subsec = "train"
    with open(all_history.format(subsec), "r") as in_file:
        data = json.load(in_file)
    with open(all_history_train_output, "w") as out_file:
        for item in data:
            conv_id = item['conv_id']
            turn_id = item['turn_id']
            query_id = f"{conv_id}_{turn_id}"
            query = item['question']
            answer = item['answers'][0]
            title = item['positive_ctxs'][0]['title']
            passage_id = item['positive_ctxs'][0]['passage_id']
            passage_text = item['positive_ctxs'][0]['text']
            
            out_file.write(json.dumps({
                "id": query_id,
                "conv_id": conv_id,
                "turn_id": turn_id,
                "title": title,
                "query": query,
                "answer": answer,
                "passage_id": passage_id,
                "passage_text": passage_text    
            }) + "\n")
    
    ### === For dev data =========
    subsec = "dev"
    with open(all_history.format(subsec), "r") as in_file:
        data = json.load(in_file)
    with open(all_history_test_output, "w") as out_file:
        for item in data:
            conv_id = item['conv_id']
            turn_id = item['turn_id']
            query_id = f"{conv_id}_{turn_id}"
            query = item['question']
            answer = item['answers'][0]
            title = item['positive_ctxs'][0]['title']
            passage_id = item['positive_ctxs'][0]['passage_id']
            passage_text = item['positive_ctxs'][0]['text']
            
            out_file.write(json.dumps({
                "id": query_id,
                "conv_id": conv_id,
                "turn_id": turn_id,
                "title": title,
                "query": query,
                "answer": answer,
                "passage_id": passage_id,
                "passage_text": passage_text    
            }) + "\n")

def prepare_quries_same_topic():
    ### === For train data =======
    subsec = "train"
    passage_info_train = get_passage_information(subsec)
    
    conversations = {}
    with open(original.format(subsec), "r") as in_file:
        data = json.load(in_file)
    
    for item in data:
        conv_id = item['Conversation_no']
        if conv_id not in conversations:
            conversations[conv_id] = []
        conversations[conv_id].append(item)   
    
    with open(same_topic_train_output, "w") as out_file:
        for item in data:
            conv_id = item['Conversation_no']
            turn_id = item['Turn_no']
            query_id = f"{conv_id}_{turn_id}"
            answer = item['Answer']
            is_nq = item['is_nq']
            title = f"{item['Topic']} [SEP] {item['Topic_section']}"
            passage_id = passage_info_train[query_id]['positive_ctxs'][0]['passage_id']
            passage_text = passage_info_train[query_id]['positive_ctxs'][0]['text']
            
            query = ""
            for conv_item in conversations[conv_id]:
                if conv_item['Turn_no'] < turn_id and conv_item['Topic'] == item['Topic']:
                    query += f"{conv_item['Question']} [SEP] {conv_item['Answer']} [SEP] "
            query += item['Question']
            
            out_file.write(json.dumps({
                "id": query_id,
                "conv_id": conv_id,
                "turn_id": turn_id,
                "is_nq": is_nq,
                "title": title,
                "query": query,
                "answer": answer,
                "passage_id": passage_id,
                "passage_text": passage_text    
            }) + "\n")
    
    
    
    ### === For test data =======
    subsec = "dev"
    passage_info_test = get_passage_information(subsec)
    
    conversations = {}
    with open(original.format(subsec), "r") as in_file:
        data = json.load(in_file)
    
    for item in data:
        conv_id = item['Conversation_no']
        if conv_id not in conversations:
            conversations[conv_id] = []
        conversations[conv_id].append(item)   
    
    with open(same_topic_test_output, "w") as out_file:
        for item in data:
            conv_id = item['Conversation_no']
            turn_id = item['Turn_no']
            query_id = f"{conv_id}_{turn_id}"
            answer = item['Answer']
            is_nq = item['is_nq']
            title = f"{item['Topic']} [SEP] {item['Topic_section']}"
            passage_id = passage_info_test[query_id]['positive_ctxs'][0]['passage_id']
            passage_text = passage_info_test[query_id]['positive_ctxs'][0]['text']
            
            query = ""
            for conv_item in conversations[conv_id]:
                if conv_item['Turn_no'] < turn_id and conv_item['Topic'] == item['Topic']:
                    query += f"{conv_item['Question']} [SEP] {conv_item['Answer']} [SEP] "
            query += item['Question']
            
            out_file.write(json.dumps({
                "id": query_id,
                "conv_id": conv_id,
                "turn_id": turn_id,
                "is_nq": is_nq,
                "title": title,
                "query": query,
                "answer": answer,
                "passage_id": passage_id,
                "passage_text": passage_text    
            }) + "\n")


if __name__ == "__main__":
    # prepare_quries_original()
    # prepare_quries_human_rewritten()
    # prepare_quries_all_history()
    prepare_quries_same_topic()