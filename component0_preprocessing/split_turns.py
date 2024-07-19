
import json
import argparse


def per_turn_number(args):
    
    # == Read files ==============
    if args.dataset_name == "QReCC":
        test_data_file = "corpus/QReCC/scai-qrecc21-test-turns.json"
    elif args.dataset_name == "TopiOCQA":
        test_data_file = "corpus/TopiOCQA/topiocqa_dev.json"
    elif args.dataset_name == "INSCIT":
        test_data_file = "corpus/INSCIT/test.json"
    
    with open(test_data_file, 'r', encoding='utf-8') as infile:
        row_data = json.load(infile)
    
    # == Split turns =============
    turn_buckets = {}
    if args.dataset_name in ["QReCC", "TopiOCQA"]:
        for item in row_data:
            key = f"turn_{item['Turn_no']}"
            turn_id = f"{item['Conversation_no']}_{item['Turn_no']}"
            if key not in turn_buckets:
                turn_buckets[key] = []
            turn_buckets[key].append(turn_id)
    
    elif args.dataset_name == "INSCIT":
        for conv_id, (conv_label, conv_data) in enumerate(row_data.items()):
            turns = conv_data["turns"]
            for turn_no, turn in enumerate(turns):
                key = f"turn_{turn_no+1}"
                turn_id = f"{conv_id+1}_{turn_no+1}"
                if key not in turn_buckets:
                    turn_buckets[key] = []
                turn_buckets[key].append(turn_id)
    
    # == Write output =============
    output_file = f"processed_datasets/{args.dataset_name}/turn_buckets/per_turn_number.json"
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(turn_buckets, outfile, indent=4)
    

def per_shift_type(args):
    # == Read files ==============
    if args.dataset_name == "QReCC":
        test_data_file = "corpus/QReCC/scai-qrecc21-test-turns.json"
    elif args.dataset_name == "TopiOCQA":
        test_data_file = "corpus/TopiOCQA/topiocqa_dev.json"
    elif args.dataset_name == "INSCIT":
        test_data_file = "corpus/INSCIT/test.json"
    
    with open(test_data_file, 'r', encoding='utf-8') as infile:
        row_data = json.load(infile)
        
    
    # == Split turns =============
    turn_buckets = {
        "First": [],
        "Topic-concentrated": [],
        "Topic-shift": []
    }
    if args.dataset_name in ["TopiOCQA"]:
        for item_idx, item in enumerate(row_data):
            turn_id = f"{item['Conversation_no']}_{item['Turn_no']}"
            if item['Turn_no'] == 1:
                turn_buckets["First"].append(turn_id)
            else:
                cur_topic = item["Topic"]
                prev_topic = row_data[item_idx-1]["Topic"]
                if cur_topic == prev_topic:
                    turn_buckets["Topic-concentrated"].append(turn_id)
                else:
                    turn_buckets["Topic-shift"].append(turn_id)
    
    elif args.dataset_name == "INSCIT":
        for conv_id, (conv_label, conv_data) in enumerate(row_data.items()):
            turns = conv_data["turns"]
            for turn_no, turn in enumerate(turns):
                turn_id = f"{conv_id+1}_{turn_no+1}"
                
                if turn_no == 0:
                    turn_buckets["First"].append(turn_id)
                else:
                    
                    cur_topic = None
                    if len(turn["labels"]) > 0:
                        if len(turn["labels"][0]["evidence"]) > 0:
                            cur_topic = turn["labels"][0]["evidence"][0]["passage_id"].split(':')[0]
                    
                    prev_topic = None
                    if len(turn["prevEvidence"]) > 0:
                        if len(turn["prevEvidence"][-1]) > 0:
                            prev_topic = turn["prevEvidence"][-1][0]["passage_id"].split(':')[0]
                    
                    if cur_topic == prev_topic:
                        turn_buckets["Topic-concentrated"].append(turn_id)
                    else:
                        turn_buckets["Topic-shift"].append(turn_id)
                    
        
    
    # == Write output =============
    output_file = f"processed_datasets/{args.dataset_name}/turn_buckets/per_shift.json"
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(turn_buckets, outfile, indent=4)
        




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="INSCIT", choices=["QReCC", "TopiOCQA", "INSCIT"])
    args = parser.parse_args()
    
    # per_turn_number(args)
    per_shift_type(args)
    

# python component0_preprocessing/split_turns.py