import json
import matplotlib.pyplot as plt

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def recall_metric(bucket_value, ret_results, k):
    bucket_recall_conter = 0
    for turn in bucket_value:
        conv_id = turn['Conversation_no']
        turn_id = turn['Turn_no']
        if f"{conv_id}_{turn_id}" in ret_results:
            ctxs_list = ret_results[f"{conv_id}_{turn_id}"]['ctxs'][:k]
            for ctx in ctxs_list:
                if ctx['has_answer']:
                    bucket_recall_conter += 1
                    break
    
    return (bucket_recall_conter/len(bucket_value))*100


def main(bucket_file, result_file, k):
    
    # === Read retriever results file ===========
    retriever_result = read_json_file(result_file)
    ret_results = {}
    for turn in retriever_result:
        ret_results[f"{turn['conv_id']}_{turn['turn_id']}"] = turn
    
    
    buckets_recall = {}
    bucket_data = read_json_file(bucket_file)
    
    for bucket_id, bucket_value in bucket_data.items():
        if len(bucket_value) != 0:
            bucket_recall = recall_metric(bucket_value, ret_results, k)
            buckets_recall[bucket_id] = bucket_recall
            print(f"{bucket_id}: {bucket_recall:.2f}")
        else:
            print(f"{bucket_id}: 0")
    
    return buckets_recall
        

def plot_results(buckets_recall, k):
    x_ticks = list(buckets_recall.keys())
    y_values = list(buckets_recall.values())
    
    
    plt.figure(figsize=(10, 5))
    plt.plot(x_ticks, y_values, marker='o', linestyle='-')
    plt.ylim(20, 55)
    plt.xticks(rotation=45)
    plt.ylabel(f'Recall@{k} (%)')
    
    plt.show()
    



if __name__ == "__main__":
    k = 10
    dataset_section = "dev"
    
    # === per topic shift
    # bucket_file = f'analysis/files/topiocqa_{dataset_section}_topic_shift_turns.jsonl'
    # === per Turn number
    bucket_file = f'analysis/files/topiocqa_{dataset_section}_turns_num.jsonl'
    
    retriever_result_file = f'baselines/topiocqa/baseline_result_files/dpr_retriever_all_history_{dataset_section}.json'
    
    buckets_recall = main(bucket_file, retriever_result_file, k)
    plot_results(buckets_recall, k)
    
    
    