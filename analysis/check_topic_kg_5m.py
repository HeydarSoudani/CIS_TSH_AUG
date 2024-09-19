import json
from multiprocessing import connection
from tomlkit import item
from tqdm import tqdm


def merge_files(output_file, *input_files):
    with open(output_file, 'w') as outfile:
        for file in input_files:
            with open(file, 'r') as infile:
                for line in tqdm(infile):
                    outfile.write(line)

def get_topic_list(topic_file, output_file):
    
    with open(topic_file, 'r') as json_file:
        data = json.load(json_file)

    with open(output_file, 'w') as jsonl_file:
        for conv_id, conv in data.items():
            
            topic_list = []
            for turn in conv:
                if turn['wd_id'] not in topic_list:
                    topic_list.append(turn['wd_id'])
            
            item = {conv_id: topic_list}
            jsonl_file.write(json.dumps(item) + '\n')
    
def get_triplet_list(topic_list_file, kg_file, output_file):
    
    def find_connections(wikidata_ids, kg_file):
        connections = {item: [] for item in wikidata_ids}

        with open(kg_file, 'r') as file:
            for line in file:
                subject, predicate, obj = line.strip().split('\t')
                for idx, item in enumerate(wikidata_ids):
                    if item != None: 
                        if idx == 0:
                            if (item == subject and obj == wikidata_ids[idx+1]) or (item == obj and wikidata_ids[idx+1] == subject):
                                connections[item].append((subject, predicate, obj))
                            
                        elif idx == len(wikidata_ids) - 1:
                            if (item == subject and obj == wikidata_ids[idx-1]) or (item == obj and wikidata_ids[idx-1] == subject):
                                connections[item].append((subject, predicate, obj))
                        
                        else:
                            if (item == subject and obj == wikidata_ids[idx+1]) or (item == obj and wikidata_ids[idx+1] == subject):
                                connections[item].append((subject, predicate, obj))
                            if (item == subject and obj == wikidata_ids[idx-1]) or (item == obj and wikidata_ids[idx-1] == subject):
                                connections[item].append((subject, predicate, obj))
                            
        return connections
    
    with open(topic_list_file, 'r') as jsonl_file, open(output_file, 'w') as out_file:
        for line in tqdm(jsonl_file):
            conv_data = json.loads(line.strip())
            conv_id, topic_list = next(iter(conv_data.items()))
            
            if len(topic_list) > 1:
                connections = find_connections(topic_list, kg_file)
            else:
                connections = {item: [] for item in topic_list}
            
            item = {conv_id: connections}
            out_file.write(json.dumps(item) + '\n')


if __name__ == "__main__":
    
    # 1) Merge KG
    kg_train = "corpus/wikidata5m_transductive/wikidata5m_transductive_train.txt"
    kg_valid = "corpus/wikidata5m_transductive/wikidata5m_transductive_valid.txt"
    kg_test = "corpus/wikidata5m_transductive/wikidata5m_transductive_test.txt"    
    kg_all = "corpus/wikidata5m_transductive/wikidata5m_transductive.txt"
    # merge_files(kg_all, kg_train, kg_valid, kg_test)
    
    
    subset = "train"
    topic_file = f"analysis/files/topiocqa_{subset}_topics.json"
    output_topic_file = f"analysis/files/topiocqa_{subset}_topics_list.jsonl"
    connection_file = f"analysis/files/topiocqa_{subset}_topics_connections.jsonl"
    # get_topic_list(topic_file, output_topic_file)
    get_triplet_list(output_topic_file, kg_all, connection_file)
    
    # python analysis/check_topic_kg_5m.py
    
    
    