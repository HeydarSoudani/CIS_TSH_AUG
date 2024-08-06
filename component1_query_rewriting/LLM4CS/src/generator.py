import os
import time
import openai

client = openai.OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY"),
)


# from https://github.com/texttron/hyde/blob/main/src/hyde/generator.py
class ChatGenerator:
    def __init__(self, 
                #  api_key,
                 n_generation,
                 **kwargs):
        self.model_name = 'gpt-3.5-turbo-16k'
        # self.api_key = api_key
        self.n_generation = n_generation
        self.kwargs = kwargs
    
    def parse_result(self, result, parse_fn):
        choices = result.choices
        n_fail = 0
        res = []
        
        for i in range(len(choices)):
            output = choices[i].message.content
            output = parse_fn(output)
            
            if not output:
                n_fail += 1
            else:
                res.append(output)
                
        return n_fail, res
       
    def generate(self, prompt, parse_fn):
        n_generation = self.n_generation
        output = []
        n_try = 0
        while True:
            if n_try == 5:
                if len(output) == 0:
                    raise ValueError("Have tried 5 times but still only got 0 successful outputs")
                output += output[:5-len(output)]
                break
            
            while True:
                try:
                    result = client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": "{}".format(prompt)},
                        ],
                        n=n_generation,
                        **self.kwargs
                    )
                    break
                except Exception as e:
                    print(f"An error occurred: {e}")
                    print("Trigger RateLimitError, wait 20s...")
                    time.sleep(20)

            n_fail, res = self.parse_result(result, parse_fn)
            output += res
            
            if n_fail == 0:
                return output 
            else:
                n_generation = n_fail    
            n_try += 1
        