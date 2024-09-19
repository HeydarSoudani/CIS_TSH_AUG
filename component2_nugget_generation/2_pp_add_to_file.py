
import json


def main():
    
    main_file = "processed_datasets/TopiOCQA/dev_nuggets_1.json"
    exul_file = "processed_datasets/TopiOCQA/out_empty_nuggets_1.json"
    output_file = "processed_datasets/TopiOCQA/dev_nuggets_2.json"
    
    exul_data = {}
    with open(exul_file, 'r') as infile:
        for line in infile:
            data = json.loads(line)
            exul_data[data["query_id"]] = data
            
    with open(main_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line)
            print(data["query_id"])
            if data['nuggets'] == []:
                temp_new_nuggets = exul_data[data["query_id"]]["nuggets"]
                if len(temp_new_nuggets) > 0:
                    if type(temp_new_nuggets[0]) == str:
                        new_nuggets = temp_new_nuggets
                    else:
                        temp = []
                        for item in temp_new_nuggets:
                            temp.append(list(item.values())[0])
                        new_nuggets = temp
                else:
                    new_nuggets = []
            else:
                new_nuggets = data["nuggets"]
            
            item = {
                "query_id": data["query_id"],
                "question": data["question"],
                "answer": data["answer"],
                "nuggets": new_nuggets
            }
            outfile.write(json.dumps(item) + '\n')
        print(f"Successfully processed and written to {output_file}")

def main_2():
    main_file = "processed_datasets/TopiOCQA/gen_topic_aware_query_rewriting_1.json"
    output_file = "processed_datasets/TopiOCQA/gen_topic_aware_query_rewriting_2.json"
    
    with open(main_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line)
            
            if data['rewritten'] != []:
                if type(data['rewritten'][0]) != str:
                    temp = []
                    for item in data['rewritten']:
                        
                        if "query" in item.keys():
                            temp.append(item["query"])
                    
                    item = {
                        "query_id": data["query_id"],
                        "topic": data["topic"],
                        "question": data["question"],
                        "answer": data["answer"],
                        "rewritten": temp
                    }
                else:
                    item = {
                        "query_id": data["query_id"],
                        "topic": data["topic"],
                        "question": data["question"],
                        "answer": data["answer"],
                        "rewritten": data["rewritten"]
                    }
                
            else:
                item = {
                    "query_id": data["query_id"],
                    "topic": data["topic"],
                    "question": data["question"],
                    "answer": data["answer"],
                    "rewritten": data["rewritten"]
                }
            
            outfile.write(json.dumps(item) + '\n')
    

if __name__ == "__main__":
    # main()
    main_2()
    
    # python component2_nugget_generation/2_pp_add_to_file.py
