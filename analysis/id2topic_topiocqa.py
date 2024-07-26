import csv, json
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt


def create_id2topic_file():
    input_path = "corpus/TopiOCQA/full_wiki_segments.tsv"    
    output_file = "corpus/TopiOCQA/id2topic.json"
    
    output_obj = {}
    with open(input_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        
        column_titles = next(reader)
        print("Column Titles:", column_titles)
        
        for i, row in enumerate(tqdm(reader)):
            id, text, title = row[0], row[1], row[2]
            output_obj[id] = title.split('[SEP]')[0]
    
    with open(output_file, 'w') as json_file:
        json.dump(output_obj, json_file, indent=2)
        
def get_retrieved_topics_per_conversations():
    # ===
    print("Loading files ...")
    gold_qrels_file = "component3_retriever/input_data/TopiOCQA/baselines/qrel_gold.trec"
    with open(gold_qrels_file, 'r') as f:
        gold_qrel_data = f.readlines()
    
    # result_qrels_file = "component3_retriever/output_results/TopiOCQA/t5_rewritten_bm25_results.trec"
    result_qrels_file = "component3_retriever/output_results/TopiOCQA/single_query_bm25_results.trec"
    with open(result_qrels_file, 'r') as f:
        result_qrel_data = f.readlines()
        
    id2topic_file = "corpus/TopiOCQA/id2topic.json"
    with open(id2topic_file, 'r') as json_file:
        id2topic = json.load(json_file)
    
    
    conv_num = 6
    # for conv_num in range(3, 206):
    print(f"== Getting topics for conversation number {conv_num} ...")
    # === Get gold topics
    # print("Getting gold topics ...")
    gold_topics = {}
    for line in gold_qrel_data:
        item = line.strip().split()
        
        query_id = item[0]
        if int(query_id.split('_')[0]) == conv_num:
            passage_id = item[2]
            passage_topic = id2topic[passage_id]
            gold_topics[query_id] = passage_topic

    # print(gold_topics)
    gold_topics_file = f"processed_datasets/TopiOCQA/topic_analysis/gold_topics_{conv_num}.json"
    with open(gold_topics_file, 'w') as json_file:
        json.dump(gold_topics, json_file, indent=2)
    
    # === Get retrieved topics
    print("Getting retrieved topics ...")
    retrieved_topics = {}
    for line in result_qrel_data:
        item = line.strip().split()
        
        query_id = item[0]
        if int(query_id.split('_')[0]) == conv_num:
            passage_id = item[2]
            passage_topic = id2topic[passage_id]
            
            if query_id not in retrieved_topics:
                retrieved_topics[query_id] = []
            retrieved_topics[query_id].append(passage_topic) 
    
    # print(retrieved_topics)
    retrieved_topics_file = f"processed_datasets/TopiOCQA/topic_analysis/single_retrieved_topics_{conv_num}.json"
    with open(retrieved_topics_file, 'w') as json_file:
        json.dump(retrieved_topics, json_file, indent=2)
    
def plot_topics():
    conv_num = 1
    gold_topics_file = f"processed_datasets/TopiOCQA/topic_analysis/gold_topics_{conv_num}.json"
    retrieved_topics_file = f"processed_datasets/TopiOCQA/topic_analysis/retrieved_topics_{conv_num}.json"

    with open(gold_topics_file, 'r') as json_file:
        gold_topics = json.load(json_file)
        
    with open(retrieved_topics_file, 'r') as json_file:
        retrieved_topics = json.load(json_file)

    all_retrieved_topics = set()
    for turn_topics in retrieved_topics.values():
        all_retrieved_topics.update(turn_topics)
    all_retrieved_topics = sorted(all_retrieved_topics)
    
    retrieved_counts = {topic: [] for topic in all_retrieved_topics}
    for turn, topics in retrieved_topics.items():
        topic_counter = Counter(topics)
        for topic in all_retrieved_topics:
            retrieved_counts[topic].append(topic_counter[topic])


    turns = list(gold_topics.keys())
    plt.figure(figsize=(15, 8))
    for topic, counts in retrieved_counts.items():
        plt.plot(turns, counts, label=topic)

    # Highlight the gold topics
    gold_labels = list(gold_topics.values())
    for i, turn in enumerate(turns):
        plt.text(turn, max([retrieved_counts[topic][i] for topic in all_retrieved_topics]), 
                gold_labels[i], color='red', fontsize=12, weight='bold')

    # Customize the plot
    plt.xlabel('Turns')
    plt.ylabel('Number of Retrieved Topics')
    plt.title('Number of Retrieved Topics per Turn with Gold Topics Highlighted')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    conv_num = 1
    top_n = 100
    
    # create_id2topic_file()
    get_retrieved_topics_per_conversations()
    # plot_topics()
    
    
# python analysis/id2topic_topiocqa.py
    
    