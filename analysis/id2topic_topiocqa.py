import csv, json
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


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
    
    result_qrels_file = "component3_retriever/output_results/TopiOCQA/all_history_bm25_results.trec"
    with open(result_qrels_file, 'r') as f:
        result_qrel_data = f.readlines()
        
    id2topic_file = "corpus/TopiOCQA/id2topic.json"
    with open(id2topic_file, 'r') as json_file:
        id2topic = json.load(json_file)
    
    
    # conv_num = 2
    for conv_num in range(3, 206):
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
        retrieved_topics_file = f"processed_datasets/TopiOCQA/topic_analysis/retrieved_topics_{conv_num}.json"
        with open(retrieved_topics_file, 'w') as json_file:
            json.dump(retrieved_topics, json_file, indent=2)
    
def plot_topics():
    conv_num = 6
    gold_topics_file = f"corpus/TopiOCQA/topic_analysis/gold_topics_{conv_num}.json"
    # retrieved_topics_file = f"corpus/TopiOCQA/topic_analysis/retrieved_topics_{conv_num}.json"
    retrieved_topics_file = "corpus/TopiOCQA/topic_analysis/single_retrieved_topics_6.json"

    with open(gold_topics_file, 'r') as json_file:
        gold_topics = json.load(json_file)
        
    with open(retrieved_topics_file, 'r') as json_file:
        retrieved_topics = json.load(json_file)

    topic_counts = Counter()
    for turn_topics in retrieved_topics.values():
        topic_counts.update(turn_topics)
    filtered_topics = {topic for topic, count in topic_counts.items() if count > 1}
    all_retrieved_topics = sorted(filtered_topics)
    # all_retrieved_topics = set()
    # for turn_topics in retrieved_topics.values():
    #     all_retrieved_topics.update(turn_topics)
    # all_retrieved_topics = sorted(all_retrieved_topics)
    
    num_turns = len(retrieved_topics)
    fig, axes = plt.subplots(num_turns, 1, figsize=(25, 0.7 * num_turns), sharex=True)

    # Plot for each turn separately
    for i, (turn, topics) in enumerate(retrieved_topics.items()):
        ax = axes[i] if num_turns > 1 else axes
        # Count the number of repeated retrieved topics for the current turn
        topic_counter = Counter(topics)
        
        # Prepare data for plotting
        x_labels = all_retrieved_topics
        y_values = [topic_counter[topic] for topic in x_labels]
        short_labels = [label[:3] for label in x_labels]  # Only use the first three characters

        # Plot the results
        bars = ax.bar(range(len(x_labels)), y_values, color='blue')
        
        # Highlight the gold topic
        gold_topic = gold_topics.get(turn, "")
        if gold_topic in x_labels:
            gold_index = x_labels.index(gold_topic)
            # Draw a rectangle around the bar corresponding to the gold topic
            # ax.add_patch(Rectangle((gold_index - 0.4, 0), 0.8, y_values[gold_index],
            #                     fill=False, edgecolor='red', linewidth=2))
            # Highlight the x label of the gold topic
            # ax.get_xticklabels()[gold_index].set_color('red')
            # ax.get_xticklabels()[gold_index].set_fontweight('bold')
            ax.axvline(x=gold_index - 0.5, color='red', linestyle=':', linewidth=1)
            ax.axvline(x=gold_index + 0.5, color='red', linestyle=':', linewidth=1)


        # Customize the plot
        ax.set_ylabel(turn)
        # ax.set_title(f'Turn {turn}: Retrieved Topics with Gold Topic Highlighted')
        # ax.grid(True)

    # Customize the common x-axis
    plt.xticks(range(len(short_labels)), short_labels, rotation=90, fontsize=8)
    plt.xlabel('Document Topics')
    
    
    # Highlight the x labels of gold topics after setting all x-ticks
    # for i, (turn, topics) in enumerate(retrieved_topics.items()):
    #     ax = axes[i] if num_turns > 1 else axes
    #     gold_topic = gold_topics.get(turn, "")
    #     if gold_topic in x_labels:
    #         gold_index = x_labels.index(gold_topic)
    #         print(gold_index)
    #         ax.get_xticklabels()[gold_index].set_color('red')
    #         ax.get_xticklabels()[gold_index].set_fontweight('bold')
    
    plt.subplots_adjust(hspace=0.3)  # Adjust this value as needed
    plt.tight_layout()

    # Show the plot
    plt.show()
    
    
    
    
    
    
    # for turn, topics in retrieved_topics.items():
    #     # Count the number of repeated retrieved topics for the current turn
    #     topic_counter = Counter(topics)
        
    #     # Prepare data for plotting
    #     x_labels = all_retrieved_topics
    #     y_values = [topic_counter[topic] for topic in x_labels]
    #     short_labels = [label[:3] for label in x_labels]
        
    #     # Plot the results
    #     plt.figure(figsize=(15, 8))
    #     plt.bar(x_labels, y_values, color='blue')
        
    #     # Highlight the gold topic
    #     # gold_topic = gold_topics.get(turn, "")
    #     # if gold_topic in x_labels:
    #     #     gold_index = x_labels.index(gold_topic)
    #     #     plt.bar(x_labels[gold_index], y_values[gold_index], color='red')
        
    #     gold_topic = gold_topics.get(turn, "")
    #     if gold_topic in x_labels:
    #         gold_index = x_labels.index(gold_topic)
    #         # Draw a rectangle around the bar corresponding to the gold topic
    #         plt.gca().add_patch(Rectangle((gold_index - 0.4, 0), 0.8, y_values[gold_index],
    #                                     fill=False, edgecolor='red', linewidth=2))
    #     # Highlight the x label of the gold topic
    #     plt.gca().get_xticklabels()[gold_index].set_color('red')
    #     plt.gca().get_xticklabels()[gold_index].set_fontweight('bold')

    #     # Customize the plot
    #     plt.xlabel('Document Topics')
    #     plt.ylabel('Number of Retrieved Topics')
    #     plt.title(f'Turn {turn}: Retrieved Topics with Gold Topic Highlighted')
    #     plt.xticks(range(len(short_labels)), short_labels, rotation=90)
    #     # plt.grid(True)
    #     plt.tight_layout()
        
    #     # Show the plot
    #     plt.show()

if __name__ == "__main__":
    conv_num = 1
    top_n = 100
    
    # create_id2topic_file()
    # get_retrieved_topics_per_conversations()
    plot_topics()
    
    
    
    