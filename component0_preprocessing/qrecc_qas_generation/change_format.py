import os, json

def main(input_file, output_file):
    
    with open('data.json', 'r') as file:
        data = json.load(file)
    
    
    
    pass


if __name__ == "__main__":
    dataset_subsec = "train" # train, test
    input_file = f"datasets/QReCC/qrecc_{dataset_subsec}.json"
    output_file = f"processed_datasets/QReCC/{dataset_subsec}_qrecc.jsonl"
    os.makedirs("processed_datasets/QReCC", exist_ok=True)
    
    main()