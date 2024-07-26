import json

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def write_json_file(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def main(input_file, output_file):
    conversations = read_json_file(input_file)
    splits = {}
    
    for turn in conversations:
        turn_info = {"Conversation_no": turn["Conversation_no"], "Turn_no": turn["Turn_no"]}
        conv_num = turn["Conversation_no"]
        turn_num = turn["Turn_no"]
        
        thtrshold = 12
        if turn_num < thtrshold:
            split_label  = turn_num
        else:
            split_label = f"{thtrshold}<"
        
        if split_label not in splits:
            splits[split_label] = []
        
        splits[split_label].append(turn_info)
        
    
    all_turns = 0
    for key, value in splits.items():
        all_turns += len(value)
        print(f"{key}: {len(value)}")
    print(f"Total turns: {all_turns}")
    
    write_json_file(splits, output_file)
        
    
    


if __name__ == "__main__":
    
    dataset_section = "train"
    input_file = f'data/topiocqa/topiocqa_{dataset_section}.json'
    output_file = f'analysis/files/topiocqa_{dataset_section}_turns_num.jsonl'
    main(input_file, output_file)
    
    
    
    
    