
import re
import time
import json
import torch
import requests
import itertools
from tqdm import tqdm
import concurrent.futures

from transformers import pipeline
from transformers import AutoTokenizer
# from transformers import AutoTokenizer, AutoModelForCausalLM

def nugget_extraction_prompt_first_turn(current_query, nugget_num=2):
    output_text = f"""
    I will provide a user query. 
    Your task is to extract concise nuggets from the current query.
    Generate {nugget_num} concise and insightful nuggets. Avoid basic or introductory-level information. Keep each nugget to a maximum of 4 words.
    
    Please extract nuggets from the following user query: {current_query}
    Provide the nuggets in the following JSON format: `{{“nuggets”: [“”, ...]}}`
    """.replace('    ', '')
    
    return output_text

def nugget_extraction_prompt(current_query, conv_history, nugget_num=10):
    
    conv_his_context = ""   
    for turn_idx, prev_turn in enumerate(conv_history):
        conv_his_context += f"turn {turn_idx}: Query: {prev_turn['query']}, Answer: {prev_turn['answer']}, Grounded Passage: {prev_turn['passage']}\n"

    output_text = f"""
    You are tasked with extracting key nuggets of information from conversation contexts.
    Your goal is to identify informative nuggets that will help retrieve passages containing answers to the current query.
    
    Conversation Context:
    {conv_his_context}
    Current Query: {current_query}
    
    Generate {nugget_num} concise and insightful nuggets to aid in retrieving relevant passages for the current query.
    Avoid basic or introductory-level information. Each nugget should be a maximum of 6 words.
    Provide the nugget set in the following JSON format: `{{“nuggets”: [“”, “”, ...]}}`
    """.replace('    ', '')
    
    return output_text

def nugget_extraction_prompt_v2(conversation_turn, nugget_num=10):
    
    output_text = f"""
    You are tasked with extracting nuggets of information from a conversation turn and its corresponding passage.
    Your goal is to identify informative nuggets that capture the core topic and focus of the conversation turn.

    Conversation Turn:
    Query: {conversation_turn['query']}, Answer: {conversation_turn['answer']}, Grounded Passage: {conversation_turn['passage']}

    Generate {nugget_num} concise and insightful nuggets that enhance understanding of the general topic of the conversation turn.
    Avoid basic or introductory-level information. Each nugget should be a maximum of 6 words.
    Provide the nugget set in the following JSON format: `{{“nuggets”: [“”, “”, ...]}}`
    """.replace('    ', '')
    
    return output_text

def nugget_extraction_prompt_only_query(conversation_turn, nugget_num=10):
    output_text = f"""
    You are tasked with generating concise, informative nuggets based on the current query and the conversation history.
    Your goal is to create nuggets that facilitate retrieving passages containing the answers to the given query.
    
    Conversation History:
    {conversation_turn['history']}

    Current Query:
    {conversation_turn['query']}
    
    Generate {nugget_num} concise and insightful nuggets, each no longer than 6 words, that help in retrieving relevant answer-containing passages.
    Provide the nugget set in the following JSON format: `{{“nuggets”: [“”, “”, ...]}}`
    """.replace('    ', '')
    
    return output_text
  
def topic_aware_query_rewriting(conversation_turn, rewrite_num=5):
    output_text = f"""
    You are tasked with rewriting the current query using the conversation history and the identified topic for the answer-containing document.
    Your goal is to rewrite the query in a way that improves the retrieval of passages containing the answers relevant to the given query.

    Conversation History: 
    {conversation_turn['history']}

    Current Query: 
    {conversation_turn['query']}

    Topic for Answer-Contained Document:
    {conversation_turn['gen_topic']}

    Generate {rewrite_num} versions of rewritten queries that are optimized for retrieving relevant answer-containing passages.
    Ensure the rewritten queries consider both the conversation history and the topic.
    You MUST mention the topic in the rewritten queries.
    Provide the set of rewritten queries in the following JSON format: `{{rewritten_queries: ["", "", ...]}}`

    Output:""".replace('    ', '')
    
    return output_text  
  

def topic_generation_prompt(conversation_turn):
    output_text = f"""
    You are tasked with generating a topic for the current query using the conversation history.
    Your goal is to generate a topic title for the current query.
    
    Conversation History: 
    {conversation_turn['history']}

    Current Query: 
    {conversation_turn['query']}
    
    Please output ONLY the topic title.
    Topic:""".replace('    ', '')
    
    return output_text  
    
def topic_generation_prompt_cot(conversation_turn):
    output_text = f"""
    You are tasked with generating a topic title for the current query based on the last two turns of conversation history.
    I will provide you with conversation turns and their corresponding topics.
    Your goal is to first detect if there is a topic shift and then generate an appropriate topic title for the current query.
    
    [Examples]
    Example 1:

    Conversation History: 
    Turn 2: what are its properties? Soft, silver-metallic, lustrous and highly crystalline transition metal, Topic: Yttrium
    Turn 3: who discover yttrium?  Carl Axel Arrhenius, Topic: Yttrium

    Current Query: 
    Turn 4: where was he born?

    Output: `{{shift: “YES”, topic: “Carl Axel Arrhenius”}}`


    Example 2:

    Conversation History: 
    Turn 9: who was she? She was an American actress, artistic director, and theatrical producer, Topic: Lucille Lortel
    Turn 10: can you mention any of her works? Ivory Tower, Red Roses for Me, The Chairs etc, Topic: Lucille Lortel

    Current Query: 
    Turn 11: when did she die?

    Output: `{{shift: “NO”, topic: “Lucille Lortel”}}`
    
    [Instructions]
    Determine whether the topic has shifted.
    If the topic has not shifted, use the previous turn’s topic for the current turn. If the topic has shifted, generate a new topic for the current turn.
    DO NOT output any additional text. 
    Provide the output in the following JSON format: `{{shift: “”, topic:””}}`


    Conversation History: 
    {conversation_turn['history']}

    Current Query: 
    {conversation_turn['query']}

    Output: 
    """.replace('    ', '')
    
    return output_text

def topic_generation_100p_shift_detector(conversation_turn):
    output_text = f"""
    You are tasked with generating a topic title for the current query based on the last two turns of conversation history.
    I will provide you with conversation turns and their corresponding topics.
    Your goal is to generate an appropriate topic title for the current query.
    
    [Examples]
    Example 1:

    Conversation History: 
    Turn 2: what are its properties? Soft, silver-metallic, lustrous and highly crystalline transition metal, Topic: Yttrium
    Turn 3: who discover yttrium?  Carl Axel Arrhenius, Topic: Yttrium

    Current Query: 
    Turn 4: where was he born?

    Output: `{{"topic": "Carl Axel Arrhenius"}}`
    
    
    Example 2:

    Conversation History: 
    Turn 6: what awards did it win? Lucille Lortel Awards, Outer Critics Circle Awards, Drama League Awards etc, Topic: Fun Home (musical)
    Turn 7: what is the first one given for? Outstanding Musical, Topic: Fun Home (musical)

    Current Query: 
    Turn 8: is it named after any person?

    Output: `{{"topic": "Lucille Lortel Awards"}}`
    
    [Instructions]
    The topic has shifted. Generate a new topic for the current turn.
    DO NOT output any additional text. 
    Provide the output in the following JSON format: `{{"topic": ""}}`
    
    
    Conversation History: 
    {conversation_turn['history']}

    Current Query: 
    {conversation_turn['query']}
    
    Output: 
    """.replace('    ', '')
    
    return output_text

def topic_generation_100p_shift_detector_2(conversation_turn):
    output_text = f"""
    You are tasked with generating a topic title for the current query based on the last two turns of conversation history.
    I will provide you with conversation turns and their corresponding topics.
    Your goal is to generate an appropriate topic title for the current query.
    
    [Examples]
    Example 1:

    Conversation History: 
    Turn 2: what are its properties? Soft, silver-metallic, lustrous and highly crystalline transition metal
    Turn 3: who discover yttrium?  Carl Axel Arrhenius

    Current Query: 
    Turn 4: where was he born?

    Output: `{{"topic": "Carl Axel Arrhenius"}}`
    
    
    Example 2:

    Conversation History: 
    Turn 6: what awards did it win? Lucille Lortel Awards, Outer Critics Circle Awards, Drama League Awards etc
    Turn 7: what is the first one given for? Outstanding Musical

    Current Query: 
    Turn 8: is it named after any person?

    Output: `{{"topic": "Lucille Lortel Awards"}}`
    
    [Instructions]
    The topic has shifted. Generate a new topic for the current turn.
    DO NOT output any additional text. 
    Provide the output in the following JSON format: `{{"topic": ""}}`
    
    
    Conversation History: 
    {conversation_turn['history']}

    Current Query: 
    {conversation_turn['query']}
    
    Output: 
    """.replace('    ', '')
    
    return output_text


def object_extraction_prompt(input_str):
    output_text = f"""
    extract json from the following string:
    {input_str}
    """.replace('    ', '')
    
    return output_text
    

class PromptFormatter:
    def __init__(self, model_id, sys_prompt=None):
        if sys_prompt is not None:
            self.sys_prompt = sys_prompt
        else:
            self.sys_prompt = "You are a helpful assistant."
        
        self.model_name = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def format_prompt(self, prompt: str) -> list:
        messages = [
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": prompt},
        ]
        
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
     

class Generator_HF:
    # def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
    def __init__(self, model_id, max_tokens=768, temperature=0.75, sys_prompt=None):
        
        self.pipeline = pipeline(
            task="text-generation",
            model=model_id,
            # torch_dtype=torch.bfloat16,
            model_kwargs={
                "torch_dtype": torch.float16,
                "quantization_config": {"load_in_4bit": True},
                "low_cpu_mem_usage": True,
                # "rope_scaling": rope_scaling
            },
            # device_map="auto"
        )
        # Params are obtained from: Generate then Retrieve
        self.max_new_tokens = max_tokens
        self.top_k = 10
        self.top_p = 0.9
        self.temperature = 0.75
        self.formatter = PromptFormatter(model_id, sys_prompt)

    def get_completion_from_prompt(self, prompt: str) -> str:
        max_retry = 5
        for i in range(max_retry):
            try:
                outputs = self.pipeline(
                    prompt,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p
                )
                assistant_response = outputs[0]['generated_text'].split('<|eot_id|><|start_header_id|>assistant<|end_header_id|>')[-1]
                return assistant_response

            except Exception as e:
                print(f"Error: {e}")
                print(f"Retry {i+1}/{max_retry}")
                continue      
        raise Exception("Failed to get completions after multiple retries")

    def pattern_extractor(self, input_text):
        try:
            json_part = re.search(r'"nuggets": \[[\s\S]*?\]', input_text).group()
            json_part = '{' + json_part + '}'
            
            # json_part = re.search(r'{.*}', input_text, re.DOTALL).group()
            # json_part = json_part.replace('“', '"').replace('”', '"')
            # json_part = json_part.replace("'", '"')
            nuggets_dict = json.loads(json_part)
            return nuggets_dict    
        
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            print("The input does not follow the correct JSON format. Please correct it and try again.")
            return None
        
        except AttributeError as e:
            print(f"Regex Search Error: {e}")
            print("The input does not contain the expected 'nuggets' JSON structure. Please correct it and try again.")
            return None

        
        # pattern = r'"nuggets": \[.*?\]}'
        # match = re.search(pattern, input_text)
        # if match:
        #     nuggets_str = match.group(0)
        #     # Fix the string to make it a valid JSON
        #     nuggets_str = '{' + nuggets_str + '}'
            
        #     try:
        #         # Convert the string to a JSON object
        #         nuggets_json = json.loads(nuggets_str)
        #         print(json.dumps(nuggets_json, indent=4))
        #         return nuggets_json
        
        #     except json.JSONDecodeError as e:
        #         print("Failed to decode JSON:", e)
        #         return None
        # else:
        #     print("Pattern not found in the text.")
        #     return None





class Generator_SGL:
    def __init__(self, model_id, max_tokens=768, temperature=0.75, sys_prompt=None):
        self.max_walkers_per_server = 500
        self.urls = [
            "http://localhost:32011/generate",
            "http://localhost:32022/generate",
            "http://localhost:32033/generate",
            "http://localhost:32044/generate",
            "http://localhost:32055/generate",
            "http://localhost:32066/generate",
            "http://localhost:32077/generate",
            "http://localhost:32088/generate",
            "http://localhost:32099/generate",
            "http://localhost:32000/generate",
        ]
        self.headers = {
            "Content-Type": "application/json"
        }
        self.sampling_params = {
            "max_new_tokens": max_tokens,
            "temperature": temperature
        }
        print('TODO: Even though the temperature is set to 0.0, the model may still generate text with some randomness. Need to investigate further.')
        self.formatter = PromptFormatter(sys_prompt)
    
    def set_available_urls(self):
        available_urls = []
        test_data = {
            "text": "test",
            "sampling_params": {
                "max_new_tokens": 1,
                "temperature": 0
            }
        }
        for url in self.urls:
            try:
                response = requests.post(url, headers=self.headers, json=test_data)
                if response.status_code == 200:
                    print(f"URL {url} is available")
                    available_urls.append(url)
            except requests.RequestException:
                print(f"URL {url} is NOT available")
                continue
        print(f"Available URLs: {available_urls}")
        self.available_urls = available_urls

    def _get_completion_from_prompts(self, prompts: list, verbose=False) -> list:
        assert isinstance(prompts, list), "Prompts must be a list of strings"
        self.set_available_urls()
        # prompts = self.apply_prompt_formatting(prompts)
        results = [None] * len(prompts)
        start_time = time.time()
        
        # Distribute requests among available URLs
        url_cycle = itertools.cycle(self.available_urls)
        futures = {}
        max_workers = self.max_walkers_per_server * len(self.available_urls)
        total_prompt_tokens = 0
        total_completion_tokens = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for idx, prompt in enumerate(prompts):
                url = next(url_cycle)
                futures[executor.submit(self.send_request, url, prompt)] = idx
                
            with tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing results") as pbar:
                for future in pbar:
                    result_idx = futures[future]
                    result = future.result()
                    results[result_idx] = result  # store the result in the original order
                    
                    if isinstance(result, dict):
                        total_prompt_tokens += result['meta_info']['prompt_tokens']
                        total_completion_tokens += result['meta_info']['completion_tokens']
                    
                    elapsed_time = time.time() - start_time
                    if elapsed_time > 0:
                        input_tokens_per_second = total_prompt_tokens / elapsed_time
                        output_tokens_per_second = total_completion_tokens / elapsed_time
                        pbar.set_postfix_str(
                            f"est. speed input: {input_tokens_per_second:.2f} toks/s, output: {output_tokens_per_second:.2f} toks/s"
                        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        
        print(f"Total prompt tokens: {total_prompt_tokens}")
        print(f"Total completion tokens: {total_completion_tokens}")

        input_tokens_per_second = total_prompt_tokens / elapsed_time
        output_tokens_per_second = total_completion_tokens / elapsed_time

        if verbose:
            self._print_results(results, input_tokens_per_second, output_tokens_per_second)
        
        # convert results to a list of completion texts
        results = [result['text'] for result in results if isinstance(result, dict)]

        for prompt, response in zip(prompts, results):
            print(f"\033[92m{prompt}\033[00m")
            print(f"\033[94m{response}\033[00m")

        return results

    def get_completion_from_prompts(self, prompts: list, verbose=False) -> list:
        max_retry = 5
        for i in range(max_retry):
            try:
                return self._get_completion_from_prompts(prompts, verbose)
            except Exception as e:
                print(f"Error: {e}")
                print(f"Retry {i+1}/{max_retry}")
                continue
        raise Exception("Failed to get completions after multiple retries")













# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# gpu_memory_utilization = 0.9
# max_model_len = 1024

# class LLMModel_vllm:
#     # ref: https://mohitkr777.medium.com/running-llama-3-llm-with-vllm-library-at-scale-aa9127ac0c27
#     def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
#         self.model = LLM(
#             model_name,
#             gpu_memory_utilization=gpu_memory_utilization,
#             max_model_len=max_model_len
#         )
#         # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.sampling_params = SamplingParams(
#             top_k=10,
#             top_p=0.9,
#             temperature=0.0,
#             # max_new_tokens=50,
#         )

#     def format_prompt(self, input_text):
#         # """<|begin_of_text|><|start_header_id|>system<|end_header_id|><|eot_id|><|start_header_id|>user<|end_header_id|>\n{context}\nQuestion: {question}\n<|eot_id|>"""
#         template = "<|startoftext|>Question: {}\nAnswer:".format(input_text)
#         return template

#     def generate_text(self, prompt, max_length=50):
#         # inputs = self.tokenizer(prompt, return_tensors="pt")
#         # outputs = self.model.generate(**inputs, max_length=max_length)
#         # return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         # outputs = self.model.generate(prompt, self.sampling_params)
        
#         # messages = [{"role": "user", "content": prompt}]
#         # formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         # self.sampling_params
#         outputs = self.model.generate(prompt, self.sampling_params)
#         return outputs