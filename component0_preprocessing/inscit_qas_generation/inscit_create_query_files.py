import json, os, csv
from tqdm import tqdm

csv.field_size_limit(10**6)


def prepare_qrels_gold():
    gold_file = "component3_retriever/data/INSCIT/dev/original.jsonl"
    trec_file = "component3_retriever/data/INSCIT/dev/qrel_gold.trec"
    
    with open(gold_file, 'r') as f, open(trec_file, 'w') as g:
        for line in f:
            turn_obj = json.loads(line.strip())
            qid = turn_obj["id"]
            for psg in turn_obj["passages"]:
                g.write("{} {} {} {}".format(qid, "Q0", psg["passage_id"], 1))
                g.write('\n')


# Src: https://github.com/ellenmellon/INSCIT/blob/main/models/DPR/prepare_data.py
def prepare_quries_original():
    subsec = "test"
    data_file = f'datasets/INSCIT/{subsec}.json'
    output_file_path = f'component3_retriever/data/INSCIT/{subsec}/original.jsonl'
    os.makedirs(f'component3_retriever/data/INSCIT', exist_ok=True)
    os.makedirs(f'component3_retriever/data/INSCIT/{subsec}', exist_ok=True)
    
    with open(data_file, 'rb') as fin, open(output_file_path, 'w') as out_file:
        content = json.load(fin)
        for cid, cname in enumerate(content):
            for tid, turn in enumerate(content[cname]['turns']):
                conv_id = cid + 1
                conv_name = cname
                turn_id = tid + 1
                query_id = f"{conv_id}_{turn_id}"
                query = turn['context'][-1] # Only current turn
                # query = ' [SEP] '.join([' '.join(t.split()) for t in turn['context']]) # All history
                answers = [' '.join(l['response'].split()) for l in turn['labels']]
                
                pos_ctxs = []
                added = set()
                for label in turn['labels']:
                    for e in label['evidence']:
                        pos_ctx = {}
                        titles = [t[:-1] if t.endswith('.') else t for t in e['passage_titles']]
                        pos_ctx['title'] = ' [SEP] '.join(titles)
                        pos_ctx['text'] = e["passage_text"]
                        pos_ctx["score"] = 1000
                        pos_ctx["title_score"] = 1
                        pid = e['passage_id']
                        pos_ctx['passage_title_id'] = pid
                        if pid in added:
                            continue
                        added.add(pid)
                        pos_ctxs += [pos_ctx]
                passages = pos_ctxs

                out_file.write(json.dumps({
                    "id": query_id,
                    "conv_name": conv_name,
                    "conv_id": conv_id,
                    "turn_id": turn_id,
                    "query": query,
                    "answers": answers,
                    "passages": passages
                }) + "\n")

def prepare_quries_all_history():
    subsec = "test"
    data_file = f'datasets/INSCIT/{subsec}.json'
    output_file_path = f'component3_retriever/data/INSCIT/{subsec}/all_history.jsonl'
    os.makedirs(f'component3_retriever/data/INSCIT', exist_ok=True)
    os.makedirs(f'component3_retriever/data/INSCIT/{subsec}', exist_ok=True)
    
    with open(data_file, 'rb') as fin, open(output_file_path, 'w') as out_file:
        content = json.load(fin)
        for cid, cname in enumerate(content):
            for tid, turn in enumerate(content[cname]['turns']):
                conv_id = cid + 1
                conv_name = cname
                turn_id = tid + 1
                query_id = f"{conv_id}_{turn_id}"
                # query = turn['context'][-1] # Only current turn
                query = ' [SEP] '.join([' '.join(t.split()) for t in turn['context']]) # All history
                answers = [' '.join(l['response'].split()) for l in turn['labels']]
                
                pos_ctxs = []
                added = set()
                for label in turn['labels']:
                    for e in label['evidence']:
                        pos_ctx = {}
                        titles = [t[:-1] if t.endswith('.') else t for t in e['passage_titles']]
                        pos_ctx['title'] = ' [SEP] '.join(titles)
                        pos_ctx['text'] = e["passage_text"]
                        pos_ctx["score"] = 1000
                        pos_ctx["title_score"] = 1
                        pid = e['passage_id']
                        pos_ctx['passage_title_id'] = pid
                        if pid in added:
                            continue
                        added.add(pid)
                        pos_ctxs += [pos_ctx]
                passages = pos_ctxs

                out_file.write(json.dumps({
                    "id": query_id,
                    "conv_name": conv_name,
                    "conv_id": conv_id,
                    "turn_id": turn_id,
                    "query": query,
                    "answers": answers,
                    "passages": passages
                }) + "\n")

def prepare_quries_same_topic():
    subsec = "test"
    data_file = f'datasets/INSCIT/{subsec}.json'
    output_file_path = f'component3_retriever/data/INSCIT/{subsec}/same_topic.jsonl'
    os.makedirs(f'component3_retriever/data/INSCIT', exist_ok=True)
    os.makedirs(f'component3_retriever/data/INSCIT/{subsec}', exist_ok=True)
    
    with open(data_file, 'rb') as fin, open(output_file_path, 'w') as out_file:
        content = json.load(fin)
        for cid, cname in enumerate(content):
            for tid, turn in enumerate(content[cname]['turns']):
                conv_id = cid + 1
                conv_name = cname
                turn_id = tid + 1
                query_id = f"{conv_id}_{turn_id}"
                answers = [' '.join(l['response'].split()) for l in turn['labels']]
                
                # == Get query =============          
                prev_pids = []
                if len(turn["prevEvidence"]) > 0:
                    for e in turn["prevEvidence"][-1]:
                        prev_pids.append(e['passage_id'])
                
                current_pids = []
                for label in turn['labels']:
                    for e in label['evidence']:
                        current_pids.append(e['passage_id'])
                    
                query = ""
                for pturn_idx, pe in enumerate(turn["prevEvidence"]):
                    for e in pe:
                        if e['passage_id'] in current_pids:
                            query += f"{turn['context'][2*pturn_idx]} [SEP] {turn['context'][2*pturn_idx+1]} [SEP] "
                            break
                query += turn['context'][-1]
                
                
                # == Get passages ==========
                pos_ctxs = []
                added = set()
                for label in turn['labels']:
                    for e in label['evidence']:
                        pos_ctx = {}
                        titles = [t[:-1] if t.endswith('.') else t for t in e['passage_titles']]
                        pos_ctx['title'] = ' [SEP] '.join(titles)
                        pos_ctx['text'] = e["passage_text"]
                        pos_ctx["score"] = 1000
                        pos_ctx["title_score"] = 1
                        pid = e['passage_id']
                        pos_ctx['passage_title_id'] = pid
                        if pid in added:
                            continue
                        added.add(pid)
                        pos_ctxs += [pos_ctx]
                passages = pos_ctxs
                
                out_file.write(json.dumps({
                    "id": query_id,
                    "conv_name": conv_name,
                    "conv_id": conv_id,
                    "turn_id": turn_id,
                    "query": query,
                    "answers": answers,
                    "passages": passages
                }) + "\n")

def prepare_quries_human_rewritten():
    pass

def title2id():
    id_col= 0
    text_col= 1
    title_col = 2
    corpus_file = "corpus/INSCIT/full_wiki_segments.tsv"
    title2id_file = "corpus/INSCIT/title2ids.json"
    
    t2id_obj = {}
    with open(corpus_file, 'r') as input:
        reader = csv.reader(input, delimiter="\t")
        for i, row in enumerate(tqdm(reader)):
            if row[id_col] == "id":
                continue
            
            psg_id = row[id_col]
            t2id_obj[psg_id] = f"doc{i}"
    
    with open(title2id_file, 'w') as file:
        json.dump(t2id_obj, file, indent=2)

def add_pid_to_data_files():
    subsec = "dev"
    query_format = "original" # original, human_rewritten, all_history, same_topic
    data_file = f'component3_retriever/data/INSCIT/{subsec}/{query_format}.jsonl'
    output_file = f'component3_retriever/data/INSCIT/{subsec}/{query_format}_new.jsonl'
    
    print("Loading t2id file ...")
    with open("corpus/INSCIT/title2ids.json", 'r') as file:
        title2ids_data = json.load(file)
    
    def get_value_with_exception_handling(key):
        try:
            return title2ids_data[key]
        except KeyError:
            print(f"Key {key} not found")
            return 0
    
    
    print("Merging id to data ...")
    with open(data_file, 'r') as in_file, open(output_file, 'w') as out_file:
        
        for i, line in enumerate(tqdm(in_file)):
            # if i == 5:
            #     break
            turn_object = json.loads(line.strip())
            passages = turn_object["passages"]
            
            new_passages_list = []
            for psg in passages:
                # psg_id = title2ids_data(psg['passage_title_id'])
                psg_id = get_value_with_exception_handling(psg['passage_title_id'])
                new_psg = {
                    "title": psg["title"],
                    "text": psg["text"],
                    "score": psg["score"],
                    "title_score": psg["title_score"],
                    "passage_title_id": psg["passage_title_id"],
                    "passage_id": psg_id[3:]
                }
                new_passages_list.append(new_psg)
            
            out_file.write(json.dumps({
                    "id": turn_object["id"],
                    "conv_name": turn_object["conv_name"],
                    "conv_id": turn_object["conv_id"],
                    "turn_id": turn_object["turn_id"],
                    "query": turn_object["query"],
                    "answers": turn_object["answers"],
                    "passages": new_passages_list
                }) + "\n")
    

if __name__ == "__main__":
        
    # prepare_qrels_gold()
    # prepare_quries_original()
    # prepare_quries_all_history()
    prepare_quries_same_topic()
    # prepare_quries_human_rewritten()
    
    
    # === Add passage id to the data files ========
    # title2id()
    # add_pid_to_data_files()

    
    # python component0_preprocessing/inscit_qas_generation/inscit_create_query_files.py
    

