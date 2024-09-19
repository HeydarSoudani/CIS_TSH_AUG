import re
import json
import torch

from transformers import pipeline
from transformers import AutoTokenizer


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
    def __init__(self, model_id, max_tokens=768, temperature=0.75, sys_prompt=None):
        
        self.pipeline = pipeline(
            task="text-generation",
            model=model_id,
            model_kwargs={
                "torch_dtype": torch.float16,
                "quantization_config": {"load_in_4bit": True},
                "low_cpu_mem_usage": True,
            },
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

