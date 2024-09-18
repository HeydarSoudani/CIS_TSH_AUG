import argparse, logging, os, json

def create_mini_corpus():
    train_file = "processed_datasets/TopiOCQA/ir_all_history_train.json"
    dev_file = "processed_datasets/TopiOCQA/ir_all_history_dev.json"
        
    with open (train_file, 'r') as file:
        data1 = json.load(file)
    with open (dev_file, 'r') as file:
        data2 = json.load(file)
    all_data = data1 + data2
    
    docs = {}
    for item in all_data:
        positive_ctxs = item["positive_ctxs"]
        for ctx in positive_ctxs:
            passage_id = ctx["passage_id"]
            title = ctx["title"]
            text = ctx["text"]
            docs[passage_id] = {"title": title, "text": text}
    
    
    
    
    
    
    
    
            
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    
    # 1) 
    create_mini_corpus()
    