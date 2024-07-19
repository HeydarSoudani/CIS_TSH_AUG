import json
import requests
from urllib.parse import quote
import urllib.request as urllib2
import matplotlib.pyplot as plt
import pandas as pd
import math

def convert_to_url_format(text):
    text = text.replace(" ", "_")
    text = quote(text)
    text = text.replace("/", "%2F")
    return text

def get_wikidata_id(wikipedia_title):
    url = f"https://en.wikipedia.org/w/api.php"
    params = {
        'action': 'query',
        'titles': wikipedia_title,
        'prop': 'pageprops',
        'format': 'json',
    }
    response = requests.get(url, params=params).json()
    pages = response.get('query', {}).get('pages', {})
    for page_id, page_data in pages.items():
        wikidata_id = page_data.get('pageprops', {}).get('wikibase_item')
        if wikidata_id:
            return wikidata_id
    return None

def get_wikidata_triplets(wikidata_id):
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{wikidata_id}.json"
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Error: Unable to fetch data for Wikidata ID {wikidata_id}")
        return []
    
    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError as e:
        print(f"Error: Unable to decode JSON response for Wikidata ID {wikidata_id}")
        return []
    
    entity = data['entities'].get(wikidata_id)
    if not entity:
        print(f"Error: No entity found for Wikidata ID {wikidata_id}")
        return []
    
    triplets = []

    # Extract labels
    labels = entity.get('labels', {})
    label = labels.get('en', {}).get('value', wikidata_id)

    # Extract claims
    claims = entity.get('claims', {})
    for property_id, claim_list in claims.items():
        for claim in claim_list:
            mainsnak = claim.get('mainsnak', {})
            if mainsnak.get('snaktype') == 'value':
                datavalue = mainsnak.get('datavalue', {})
                value = datavalue.get('value', {})
                if isinstance(value, dict) and 'id' in value:
                    object_id = value['id']
                    triplets.append((label, property_id, object_id))
                elif isinstance(value, dict) and 'text' in value:
                    object_text = value['text']
                    triplets.append((label, property_id, object_text))

    return triplets

def get_pageviews(wiki_title):
    TOP_API_URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{lang}.{project}/all-access/all-agents/{topic}/monthly/{date_from}/{date_to}"
    lang = 'en'
    project = 'wikipedia'
    date_from = "2022040100"
    date_to = "2022043000"
    # date_from = "2021010100"
    # date_to = "2021103100"
    all_views = 0
    
    edited_title = convert_to_url_format(wiki_title)
    url = TOP_API_URL.format(lang=lang,
                            project = project,
                            topic = edited_title,
                            date_from = date_from,
                            date_to = date_to)
    try:
        resp = urllib2.urlopen(url)
        resp_bytes = resp.read()
        data = json.loads(resp_bytes)
        all_views = sum([item['views'] for item in data['items']]) 
        # print("Target: {:<15}, Views: {}".format(edited_title, all_views))
    except urllib2.HTTPError as e:
        # print(e.code)
        print("Target: {:<20}, does not have a wikipedia page".format(edited_title))
    except urllib2.URLError as e:
        print(e.args)
    
    return all_views

def topic_extraction_topiocqa():
    input_file = 'data/topiocqa/topiocqa_train.json'
    output_file = 'analysis/topiocqa_train_topics.json'
    # input_file = 'data/topiocqa/topiocqa_dev.json'
    # output_file = 'analysis/topiocqa_dev_topics.json'
    
    with open(input_file, "r") as file:
        conversation_data = json.load(file)
        
    dataset_summary = {}
    for item in conversation_data:
        
        if str(item["Conversation_no"]) not in dataset_summary:
            dataset_summary[str(item["Conversation_no"])] = []
        
        dataset_summary[str(item["Conversation_no"])].append({
            'title': item["Topic"],
            'sub_title': item["Topic_section"]
        })
    
    with open(output_file, "w") as output_file:
        json.dump(dataset_summary, output_file, indent=2)
  
def popularity_plot(input_file):
    with open(input_file, "r") as file:
        conversation_data = json.load(file)
    
    unique_topics = {}
    for conv_id, conv in conversation_data.items():
        for turn in conv:
            title = turn["title"]
            if title not in unique_topics:
                unique_topics[title] = turn["page_views"]

    print(unique_topics)
    print(len(unique_topics))
    
    split_points = [2, 3, 4, 5]
    df = pd.DataFrame(list(unique_topics.items()), columns=['Title', 'Pageview'])
    df['Log_Pageview'] = df['Pageview'].apply(lambda x: math.log(x, 10) if x > 0 else 100000)
    df['Bucket'] = pd.cut(df['Log_Pageview'], bins=[-float('inf')] + split_points + [float('inf')], 
                          labels=['B1', 'B2', 'B3', 'B4', 'B5'])

    # Plot the bar plot with the new log-based buckets
    plt.figure(figsize=(10, 6))
    log_bucket_counts = df['Bucket'].value_counts().sort_index()
    log_bucket_counts.plot(kind='bar', color='skyblue')
    plt.xlabel('Log Popularity Bucket')
    plt.ylabel('Number of Titles')
    plt.title('Distribution of Titles in Log Popularity Buckets')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Display the plot
    plt.show()

    # Display the dataframe to the user
    # tools.display_dataframe_to_user(name="Pageviews Data with Log Buckets", dataframe=df)

def topic_extraction_results():
    input_file = 'baselines/INSCIT/results/fid_dev.json'
    output_file = 'analysis/files/fid_dev_topics.json'

    with open(input_file, "r") as file:
        result_data = json.load(file)
    
    dataset_summary = {}
    for conv_item in result_data:
        conv_id, turn_id = conv_item['conv_id'], conv_item['turn_id']
        evidences = conv_item['output']['evidence']
        item = {
            "passage_id": [ev['passage_id'] for ev in evidences],
        },
        
        if conv_id not in dataset_summary:
            dataset_summary[conv_id] = [item]
        else:
            dataset_summary[conv_id].append(item)
        
    with open(output_file, "w") as output_file:
        json.dump(dataset_summary, output_file, indent=2)

def topic_extraction_inscet():
    print("\n\nExtracting topics from INSCIT dataset ...")
    input_file = 'data/INSCIT/train.json'
    output_file = 'analysis/files/INSCIT_train_topics.json'

    with open(input_file, "r") as file:
        conversation_data = json.load(file)

    dataset_summary = {}
    for conv_id, conv in conversation_data.items():
        print(f"Processing conversation {conv_id} ...")
        
        wiki_titles = []
        turns = conv["turns"]
        last_turn = turns[-1]
        prev_evids = last_turn["prevEvidence"]
        for prev_evid in prev_evids:
            if len(prev_evid) > 0:
                # wiki_title = prev_evid[0]['passage_titles'][0]
                # wiki_title = prev_evid[0]['passage_id'].split(':')[0]
                sub_title = prev_evid[0]["passage_titles"][1:]
                
                wiki_title = {
                    'passage_id': prev_evid[0]['passage_id'], 
                    'title': prev_evid[0]['passage_id'].split(':')[0],
                    'sub_title': sub_title,
                    
                }
                wiki_titles.append(wiki_title)
        
        current_evid = last_turn['labels'][0]['evidence']
        if len(current_evid) > 0:
            sub_title = current_evid[0]["passage_titles"][1:]
            wiki_title = {
                'title': current_evid[0]['passage_id'].split(':')[0],
                'sub_title': sub_title
            }
            wiki_titles.append(wiki_title)
        
        dataset_summary[conv_id] = wiki_titles

    with open(output_file, "w") as output_file:
        json.dump(dataset_summary, output_file, indent=2)
    
def wiki_id_extraction(input_file, output_file):
    print("\n\nExtracting Wikidata IDs from Wikipedia titles ...")
    with open(input_file, "r") as file:
        conversation_sum_data = json.load(file)

    dataset_summary = {}
    for idx, (conv_id, conv) in enumerate(conversation_sum_data.items()):
        print(f"Processing conversation {conv_id} ...")
        
        # if idx == 10:
        #     break
        
        for turn in conv:
            wp_title = turn["title"]
            wp_sub_title = turn["sub_title"]
            wd_id = get_wikidata_id(wp_title)
            pg_view = get_pageviews(wp_title)
            
            if conv_id not in dataset_summary:
                dataset_summary[conv_id] = []
            
            dataset_summary[conv_id].append({
                'title': wp_title,
                'sub_title': wp_sub_title,
                'wd_id': wd_id,
                'page_views': pg_view
            }) 
        
    with open(output_file, "w") as output_file:
        json.dump(dataset_summary, output_file, indent=2)
 
def triplet_extraction(input_file):
    print("\n\nExtracting triplets from Wikidata IDs ...")
    with open(input_file, "r") as file:
        conversation_data = json.load(file) 
    
    new_conversation_data = {}
    for conv_id, conv_turns in conversation_data.items():
        print(f"Processing conversation {conv_id} ...")
        new_conversation_data[conv_id] = []
        
        for turn in conv_turns:
            title, sub_title, wd_id, page_views = turn["title"], turn["sub_title"], turn["wd_id"], turn["page_views"]
            triplets = get_wikidata_triplets(wd_id)
            new_conversation_data[conv_id].append({
                'title': title,
                'sub_title': sub_title,
                'wd_id': wd_id,
                'page_views': page_views,
                'triplets': triplets
            })
    
    with open(input_file, "w") as output_file:
        json.dump(new_conversation_data, output_file, indent=2)

# Function to get categories from Wikidata
def get_wikidata_categories(wikidata_id):
    url = "https://query.wikidata.org/sparql"
    query = """
    SELECT ?category ?categoryLabel WHERE {
      wd:%s wdt:P910 ?category .
      SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
    }
    """ % wikidata_id
    headers = {
        "Accept": "application/sparql-results+json"
    }
    response = requests.get(url, params={'query': query}, headers=headers)
    data = response.json()
    categories = [result['categoryLabel']['value'] for result in data['results']['bindings']]
    return categories

# Function to map categories to high-level categories
def map_to_high_level_category(categories):
    high_level_categories = {
        'Food and drink': ['Cuisine', 'Drink', 'Food'],
        'Hobby': ['Hobby', 'Leisure activities', 'Craft'],
        'Historical events': ['History', 'Historical events'],
        'Geography': ['Geography', 'Places', 'Regions'],
        'Weekly top-25 pages': ['Popular pages', 'Trending', 'Top pages']
    }
    
    mapped_categories = {}
    for category in categories:
        for high_level, keywords in high_level_categories.items():
            if any(keyword.lower() in category.lower() for keyword in keywords):
                if high_level not in mapped_categories:
                    mapped_categories[high_level] = []
                mapped_categories[high_level].append(category)
    return mapped_categories

if __name__ == "__main__":
    
    # input_file = 'analysis/files/INSCIT_dev_topics.json'
    # output_file = 'analysis/files/INSCIT_dev_topics.json'
    input_file = 'analysis/files/INSCIT_train_topics.json'
    output_file = 'analysis/files/INSCIT_train_topics.json'
    # input_file = 'analysis/files/topiocqa_dev_topics.json'
    # output_file = 'analysis/files/topiocqa_dev_topics.json'
    # input_file = 'analysis/files/topiocqa_train_topics.json'
    # output_file = 'analysis/files/topiocqa_train_topics.json'
    
    # === from datasets
    # topic_extraction_inscet()
    # # topic_extraction_topiocqa()
    
    # wiki_id_extraction(input_file, output_file) # Added some wd_id handy
    # # popularity_plot(input_file)
    
    # triplet_extraction(input_file)
    
    
    wikidata_id = 'Q11469'
    # Example usage
    # wikidata_id = 'Q42'  # Replace with your Wikidata ID
    categories = get_wikidata_categories(wikidata_id)
    mapped_categories = map_to_high_level_category(categories)
    for high_level, cat_list in mapped_categories.items():
        print(f"High-Level Category: {high_level}")
        for cat in cat_list:
            print(f"  - {cat}")
    
    
    
    # === From results files
    # topic_extraction_results()
    
    
    
    
    
    
    print("Done!")
    
    
    
    
    
    
    
