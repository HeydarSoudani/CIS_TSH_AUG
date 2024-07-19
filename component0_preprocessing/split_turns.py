
import json
import argparse


def per_turn_number(args):
    test_data_file = "corpus/TopiOCQA/topiocqa_dev.json"
    output_file = f"processed_datasets/{args.dataset_name}/turn_buckets/per_turn_number.json"
    
    with open(test_data_file, 'r', encoding='utf-8') as infile:
        row_data = json.load(infile)
    
    turn_buckets = {}
    for item in row_data:
        key = f"turn_{item['Turn_no']}"
        turn_id = f"{item['Conversation_no']}_{item['Turn_no']}"
        if key not in turn_buckets:
            turn_buckets[key] = []
        turn_buckets[key].append(turn_id)
        
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(turn_buckets, outfile, indent=4)
    

def per_shift_type(args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="TopiOCQA", choices=["QReCC", "TopiOCQA", "INSCIT"])
    args = parser.parse_args()
    
    per_turn_number(args)
    per_shift_type(args)
    

# python component0_preprocessing/split_turns.py