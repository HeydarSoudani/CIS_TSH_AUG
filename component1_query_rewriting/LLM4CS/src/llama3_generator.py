import transformers
import torch
import time


class Llama3Generator:
    def __init__(self, n_generation, **kwargs):
        self.model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
        # self.model_name = 'meta-llama/Llama-2-7b-chat-hf'
        self.n_generation = n_generation
        self.kwargs = kwargs
        
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
    
    def parse_result(self, result, parse_fn):
        n_fail = 0
        res = []
        output = parse_fn(result)
        if not output:
            n_fail += 1
        else:
            res.append(output)            
        return n_fail, res
    
    def generate(self, prompt, parse_fn):
        n_generation = self.n_generation
        output = []
        n_try = 0
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "{}".format(prompt)},
        ]
        prompt = self.pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        while True:
            if n_try == 5:
                if len(output) == 0:
                    raise ValueError("Have tried 5 times but still only got 0 successful outputs")
                output += output[:5-len(output)]
                break
        
            while True:
                try:
                    outputs = self.pipeline(
                        prompt,
                        eos_token_id=terminators,
                        do_sample=True,
                        num_return_sequences=n_generation,
                        **self.kwargs
                    )
                    result = outputs[0]["generated_text"][len(prompt):]
                    break
                except Exception as e:
                    print('Error message:', e)
                    time.sleep(20)
                    print("Trigger RateLimitError, wait 20s...")
            
            print(result) 
            n_fail, res = self.parse_result(result, parse_fn)
            output += res
            
            if n_fail == 0:
                return output 
            else:
                n_generation = n_fail
                
            n_try += 1
