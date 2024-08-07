import time
import itertools
from tqdm import tqdm
import concurrent.futures

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


class Llama3HFWrapper:
    def __init__(self, model_id, max_tokens=768, temperature=0.75, sys_prompt=None):
        
        # rope_scaling = {
        #     'type': 'linear',  # Example value, adjust based on your model's requirements
        #     'factor': 4.0
        # }

        self.pipeline = pipeline(
            task="text-generation",
            model=model_id,
            model_kwargs={
                "torch_dtype": torch.float16,
                "quantization_config": {"load_in_4bit": True},
                "low_cpu_mem_usage": True,
                # "rope_scaling": rope_scaling
            },
        )
        self.max_new_tokens = max_tokens
        self.temperature = temperature
        self.formatter = PromptFormatter(model_id, sys_prompt)


    def get_completion_from_prompt(self, prompt: str) -> list:
        max_retry = 5
        for i in range(max_retry):
            
            try:
                
                outputs = self.pipeline(
                    prompt,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True
                )
                print(outputs)
                print('======')
                
                assistant_response = outputs[0]['generated_text'].split('<|eot_id|><|start_header_id|>assistant<|end_header_id|>')[-1]
                print(assistant_response)
                print('======')
                
                # assistant_response = outputs[0]["generated_text"][-1]["content"]
                return assistant_response
            
            except Exception as e:
                print(f"Error: {e}")
                print(f"Retry {i+1}/{max_retry}")
                continue
        
        raise Exception("Failed to get completions after multiple retries")
