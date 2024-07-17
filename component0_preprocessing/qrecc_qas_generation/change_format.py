import os, json
from sklearn.model_selection import train_test_split


def change_format(input_file, output_file):
    
    with open(input_file, 'r') as file:
        data = json.load(file)
    
    with open(output_file, 'w') as ofile:
        for i, turn in enumerate(data):
            
            conv_id = turn["Conversation_no"]
            turn_id = turn["Turn_no"]
            cur_query = turn["Question"]
            orc_query = turn["Rewrite"]
            prev_queries = turn["Context"][::2]
            prev_res = turn["Context"][1::2]
            
            item = {
                "sample_id": f"{conv_id}_{turn_id}",
                "cur_utt_text": cur_query,
                "oracle_utt_text": orc_query,
                "ctx_utts_text": prev_queries,
                "ctx_resps_text": prev_res
            }
            ofile.write(json.dumps(item) + '\n')
    
def train_dev_split_qrecc():
    base_dir = "processed_datasets/QReCC"
    
    with open(f"{base_dir}/train_qrecc.jsonl", "r") as f:
        data = f.readlines()

    train_data, dev_data = train_test_split(data, test_size=0.1, random_state=42)
    with open(f"{base_dir}/new_train_qrecc.json", "w") as ft, open(f"{base_dir}/new_dev_qrecc.json", "w") as fd:
        for x in train_data:
            ft.write(x)
        for x in dev_data:
            fd.write(x)

if __name__ == "__main__":
    dataset_subsec = "train" # train, test
    input_file = f"datasets/QReCC/qrecc_{dataset_subsec}.json"
    output_file = f"processed_datasets/QReCC/{dataset_subsec}_qrecc.jsonl"
    os.makedirs("processed_datasets/QReCC", exist_ok=True)
    
    # change_format(input_file, output_file)
    train_dev_split_qrecc()
    
    # python component0_preprocessing/qrecc_qas_generation/change_format.py