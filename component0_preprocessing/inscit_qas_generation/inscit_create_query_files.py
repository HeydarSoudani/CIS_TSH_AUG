import json, os

def prepare_qrels_gold():
    pass

# Src: https://github.com/ellenmellon/INSCIT/blob/main/models/DPR/prepare_data.py
def prepare_quries_original():
    subsec = "dev"
    data_file = f'datasets/INSCIT/{subsec}.json'
    output_file_path = f'component3_retriever/data/INSCIT/{subsec}/original.jsonl'
    os.makedirs(f'component3_retriever/data/INSCIT', exist_ok=True)
    os.makedirs(f'component3_retriever/data/INSCIT/{subsec}', exist_ok=True)
    
    with open(data_file, 'rb') as fin, open(output_file_path, 'w') as out_file:
        content = json.load(fin)
        for cid, cname in enumerate(content):
            for tid, turn in enumerate(content[cname]['turns']):
                conv_id = cid
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
                        pos_ctx['passage_id'] = pid
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

def prepare_quries_all_history():
    pass

def prepare_quries_same_topic():
    pass


if __name__ == "__main__":
        
    # prepare_qrels_gold()
    prepare_quries_original()
    # prepare_quries_human_rewritten()
    # prepare_quries_all_history()
    # prepare_quries_same_topic()
    
    

