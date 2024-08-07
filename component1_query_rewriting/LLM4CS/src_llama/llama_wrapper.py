import time
import itertools
from tqdm import tqdm
import concurrent.futures

import torch
from transformers import pipeline
from transformers import AutoTokenizer


class PromptFormatter:
    def __init__(self, sys_prompt=None):
        if sys_prompt is not None:
            self.sys_prompt = sys_prompt
        else:
            self.sys_prompt = "You are a helpful assistant."
        self.model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def format_prompt(self, prompt: str) -> list:
        messages = [
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": prompt},
        ]
        
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


class Llama3HFWrapper:
    def __init__(self, max_tokens=768, temperature=0.75, sys_prompt=None):
        
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.pipeline = pipeline(
            task="text-generation",
            model=model_id,
            model_kwargs={
                "torch_dtype": torch.float16,
                "quantization_config": {"load_in_4bit": True},
                "low_cpu_mem_usage": True,
            },
        )
        self.max_new_tokens = max_tokens
        self.temperature = temperature
        self.formatter = PromptFormatter(sys_prompt)


    def get_completion_from_prompt(self, prompt: str) -> list:
        max_retry = 5
        for i in range(max_retry):
            
            try:
                prompt = self.formatter.format_prompt(prompt)
                outputs = self.pipeline(
                    prompt,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False
                )
                assistant_response = outputs[0]["generated_text"][-1]["content"]
                return assistant_response
            
            except Exception as e:
                print(f"Error: {e}")
                print(f"Retry {i+1}/{max_retry}")
                continue
        
        raise Exception("Failed to get completions after multiple retries")
