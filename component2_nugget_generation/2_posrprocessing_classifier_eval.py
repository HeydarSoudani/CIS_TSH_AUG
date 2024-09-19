
import argparse, json


def main(args):

    bucket_file = f"processed_datasets/{args.dataset_name}/turn_buckets/per_shift.json"
    with open(bucket_file, 'r') as f:
        bucket_data = json.load(f)
    shift_turns = bucket_data["Topic-shift"]
    concentrated_turns = bucket_data["Topic-concentrated"]    
    
    input_data = {}
    with open(args.input_file_path, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            input_data[data["query_id"]] = data["output"]
       
    # 
    counter_sh = 0
    for turn_id in shift_turns:
        # shift = input_data[turn_id]["shift"]
        
        if "shift" in input_data[turn_id]:
            shift = input_data[turn_id]["shift"]
        else:
            shift = "NO"
        
        if shift == "YES":
            counter_sh += 1
    
    acc = (counter_sh / len(shift_turns))*100
    print(f"Accuracy (shift): {acc}")

    # 
    counter_c = 0
    for turn_id in concentrated_turns:
        
        if "shift" in input_data[turn_id]:
            shift = input_data[turn_id]["shift"]
        else:
            shift = "YES"
        
        if shift == "NO":
            counter_c += 1
    
    acc = (counter_c / len(concentrated_turns))*100
    print(f"Accuracy (concentrated): {acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset_name", type=str, default="TopiOCQA", choices=["QReCC", "TopiOCQA", "INSCIT"])
    parser.add_argument("--input_file_path", type=str, default="processed_datasets/TopiOCQA/cot_topic_gen_1.json")
    args = parser.parse_args()
    
    main(args)
    
# python component2_nugget_generation/2_posrprocessing_classifier_eval.py