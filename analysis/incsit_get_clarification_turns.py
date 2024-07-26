import json


def get_turns_by_type():
    turn_type = "clarification"
    input_file = 'data/INSCIT/dev.json'
    output_file = f'analysis/files/inscit_dev_{turn_type}_turns.jsonl'
    with open(input_file, "r") as file:
        conversation_data = json.load(file)
    
    
    with open (output_file, "w") as ofile:
        for conv_id, conv in conversation_data.items():
            print(f"Processing conversation {conv_id} ...")
            
            turns = conv["turns"]
            
            for turn_number, turn in enumerate(turns):
                labels = turn["labels"]
                for label_idx, label in enumerate(labels):
                    if label["responseType"] == turn_type:
                        item = {
                            'conv_id': conv_id,
                            'turn_number': turn_number,
                            'label_idx': label_idx,
                            'response': label['response']
                        }
                        ofile.write(json.dumps(item) + '\n')

def get_turns_in_results():
    turn_type = "clarification"
    input_file = f'analysis/files/inscit_dev_{turn_type}_turns.jsonl'
    baseline_results_file = 'baselines/INSCIT/results/dialki_fid_dev.json'
    output_file = f'analysis/files/inscit_dev_baseline_result_{turn_type}_turns.jsonl'
    
    clarification_turns = []
    with open (input_file, 'r') as in_file:
        for line in in_file:
            data = json.loads(line.strip())
            clarification_turns.append(data) 

    with open(baseline_results_file, "r") as file:
        baseline_results_data = json.load(file)
        
    with open (output_file, "w") as ofile:
        for turn in clarification_turns:
            conv_id, turn_number = turn['conv_id'], turn['turn_number']
            turn_id = turn_number + 1
            
            for result_turn in baseline_results_data:
                if result_turn['conv_id'] == conv_id and result_turn['turn_id'] == str(turn_id):
                    item = {
                        'conv_id': conv_id,
                        'turn_number': turn_number,
                        'response': result_turn['output']['response']
                    }
                    ofile.write(json.dumps(item) + '\n')
                    break
        
        

if __name__ == "__main__":
    # get_turns_by_type()
    
    get_turns_in_results()