import re
import json
import argparse


def extract_json_objects(text):
    """
    Extracts JSON objects from a given string using regex.
    The function handles cases where the JSON object may be incomplete, i.e., missing closing brackets.
    """
    try:
        # Regex pattern to match a JSON object or array structure.
        pattern = r'(\{(?:[^{}]|(?R))*\]?)'
        matches = re.findall(pattern, text)
        
        # Ensure all found matches are valid JSON by checking if they can be parsed
        valid_json_objects = []
        for match in matches:
            try:
                # Attempt to load the match as JSON, and add missing brackets if necessary
                if match[-1] != '}':
                    match += '}'
                if match[-2:] != ']}':
                    match = match.rstrip(']') + ']}'
                json_object = json.loads(match)
                valid_json_objects.append(json_object)
            except json.JSONDecodeError:
                continue

        return valid_json_objects

    except Exception as e:
        print(f"Error occurred: {e}")
        return []

def extract_json_objects_from_string(input_string, query_id):
    """
    This function extracts JSON objects from a string using regular expressions.
    The JSON object is expected to start with '{' and end with '}', but it might be missing the closing '}' or ']'.
    """
    # Regular expression to find JSON objects
    json_object_pattern = r'\{.*?\]'
    
    # Find all matches
    # cleaned_string = input_string.replace('\u201c', '').replace('\u201d', '')
    cleaned_string = input_string.replace('\u201c', '"').replace('\u201d', '"')
    cleaned_string = re.sub(r'\\n', '', cleaned_string)

    cleaned_string = re.sub(r'"\s*"', '","', cleaned_string)

    # print(cleaned_string)
    matches = re.findall(json_object_pattern, cleaned_string, re.DOTALL)
    
    json_objects = []
    for match in matches:
        try:
            # Ensure the JSON string ends with proper closing characters
            match = match.strip()
            if not match.endswith(']'):
                match += ']'
            if not match.endswith('}'):
                match += '}'
            
            # Load the string into a JSON object
            json_obj = json.loads(match)
            json_objects.append(json_obj)
            
        except json.JSONDecodeError as e:
            print(f"Query_id: {query_id}. Error decoding JSON: {e}")
            
            # try to repair
            cleaned_string = re.sub(r'"\s*"', '","', cleaned_string)
            matches = re.findall(json_object_pattern, cleaned_string, re.DOTALL)

    # print(json_objects)
    if len(json_objects) > 0:
        if "rewritten_queries" in json_objects[0]:
            output = json_objects[0]["rewritten_queries"]
        else:
            output = []    
    else:
        output = []
    
    return output

def extract_json_objects_from_string_2(input_string, query_id):
    cleaned_string = input_string.replace('“', '"').replace('”', '"')
    # cleaned_string = cleaned_string.strip()
    cleaned_string = re.sub(r'\s+', ' ', cleaned_string).strip()
    print(cleaned_string)
    
    match = re.search(r'{.*}', cleaned_string)
    if match:
        json_string = match.group(0)
        json_string = json_string.replace('\\', '').replace('shift:', '"shift":').replace('topic:', '"topic":')
        data = json.loads(json_string)

        return data
    else:
        print(f"No match found for {query_id}")
        return {}

def write_json_objects_to_file(json_objects, output_file):
    """
    Writes each JSON object to a new line in a specified output file.
    """
    try:
        with open(output_file, 'w') as file:
            for obj in json_objects:
                json.dump(obj, file)
                file.write('\n')
        print(f"JSON objects successfully written to {output_file}")
    except Exception as e:
        print(f"Error occurred while writing to file: {e}")

def main(args):

    with open(args.input_file_path, 'r') as infile, open(args.output_file_path, 'w') as outfile:
        for idx, line in enumerate(infile):
            
            # if idx == 10:
            #     break
            
            data = json.loads(line)
            sample_text = data["output"]
            # sample_text = sample_text.replace('“', '"').replace('”', '"')
            # json_objects = extract_json_objects(sample_text)
            # rewritten = extract_json_objects_from_string(sample_text, data["query_id"])
            # item = {
            #     "query_id": data["query_id"],
            #     "topic": data["gen_topic"],
            #     "question": data["question"],
            #     "answer": data["answer"],
            #     "rewritten": rewritten
            # }
            
            # == For gen 
            if sample_text != "":
                obj_output = extract_json_objects_from_string_2(sample_text, data["query_id"])
            else:
                obj_output = ""
            item = {
                "query_id": data["query_id"],
                "question": data["question"],
                "answer": data["answer"],
                "topic": data["topic"],
                "output": obj_output
            }
            
            
            outfile.write(json.dumps(item) + '\n')
    print(f"Successfully processed and written to {args.output_file_path}")


def extend_topic(args):
    
    last_topic = ""
    with open(args.input_file_path, 'r') as infile, open(args.output_file_path, 'w') as outfile:
        for idx, line in enumerate(infile):
            
            # if idx == 10:
            #     break
            
            data = json.loads(line)
            output = data["output"]
            print(output)
            
            if output != "":
                last_topic = output['topic']
    
            obj_output = {"topic": last_topic}
            item = {
                "query_id": data["query_id"],
                "question": data["question"],
                "answer": data["answer"],
                "topic": data["topic"],
                "output": obj_output
            }
            outfile.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input_file_path", type=str, default="processed_datasets/TopiOCQA/dev_nuggets.json")
    # parser.add_argument("--output_file_path", type=str, default="processed_datasets/TopiOCQA/dev_nuggets_1.json")
    # parser.add_argument("--input_file_path", type=str, default="processed_datasets/TopiOCQA/topic_aware_query_rewriting.json")
    # parser.add_argument("--output_file_path", type=str, default="processed_datasets/TopiOCQA/topic_aware_query_rewriting_1.json")
    # parser.add_argument("--input_file_path", type=str, default="processed_datasets/TopiOCQA/gen_topic_aware_query_rewriting.json")
    # parser.add_argument("--output_file_path", type=str, default="processed_datasets/TopiOCQA/gen_topic_aware_query_rewriting_1.json")
    # parser.add_argument("--input_file_path", type=str, default="processed_datasets/TopiOCQA/cot_topic_gen.json")
    # parser.add_argument("--output_file_path", type=str, default="processed_datasets/TopiOCQA/cot_topic_gen_1.json")
    parser.add_argument("--input_file_path", type=str, default="processed_datasets/TopiOCQA/topic_gen_100p_shift_detector_no_topic_1.json")
    # parser.add_argument("--output_file_path", type=str, default="processed_datasets/TopiOCQA/topic_gen_100p_shift_detector_no_topic_1.json")
    parser.add_argument("--output_file_path", type=str, default="processed_datasets/TopiOCQA/topic_gen_100p_shift_detector_no_topic_2.json")
    
    args = parser.parse_args()
    
    # main(args)
    extend_topic(args)
    
    # python component2_nugget_generation/2_postprocessing_nugget_extraction.py
    