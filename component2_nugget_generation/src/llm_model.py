
import re
import os
import json
import torch
from transformers import pipeline
from transformers import AutoTokenizer
# from vllm import LLM, SamplingParams
# from transformers import AutoTokenizer, AutoModelForCausalLM


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
     
        

class LLMModel_hf:
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
        self.max_new_tokens = 150
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
                    # temperature=self.temperature,
                    # top_k=self.top_k,
                    # top_p=self.top_p
                )
                print(outputs)
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
