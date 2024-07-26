import json

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def write_json_file(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def split_conversations_by_topic_shifting(conversations):
    splits = {
        "start": [],
        "t_concentrated": []    
    }
    conversation_no = 0
    shift_counter = 0

    for turn in conversations:
        turn_info = {"Conversation_no": turn["Conversation_no"], "Turn_no": turn["Turn_no"]}
        
        # New conversation, First turn
        if turn["Conversation_no"] != conversation_no:
            conversation_no = turn["Conversation_no"]
            current_split = "start"
            splits[current_split].append(turn_info)
            
            shift_counter = 1
            current_split = f"{current_split}-sh{shift_counter}"
            if current_split not in splits:
                splits[current_split] = []
            
            previous_topic = turn["Topic"]
            
    
        elif turn["Topic"] == previous_topic:
            if shift_counter == 1:
                splits[current_split].append(turn_info)
            else:
                splits["t_concentrated"].append(turn_info)
            
            
        # Detected a topic shift
        elif turn["Topic"] != previous_topic:
            
            if shift_counter < 9:
                shift_split = f"sh{shift_counter}"
                if shift_split not in splits:
                    splits[shift_split] = []
                splits[shift_split].append(turn_info)
                shift_counter += 1
            else:
                shift_counter = 10
                shift_split = f"sh{shift_counter}<"
                if shift_split not in splits:
                    splits[shift_split] = []
                splits[shift_split].append(turn_info)
            
            # current_split = f"{shift_split}-sh{shift_counter+1}"
            # if current_split not in splits:
            #     splits[current_split] = []
        
    
    return splits


if __name__ == "__main__":
    
    dataset_section = "train"
    input_file = f'data/topiocqa/topiocqa_{dataset_section}.json'
    output_file = f'analysis/files/topiocqa_{dataset_section}_topic_shift_turns.jsonl'

    conversations = read_json_file(input_file)
    result = split_conversations_by_topic_shifting(conversations)
    
    all_turns = 0
    for key, value in result.items():
        all_turns += len(value)
        print(f"{key}: {len(value)}")
    print(f"Total turns: {all_turns}")
    
    write_json_file(result, output_file)
    
