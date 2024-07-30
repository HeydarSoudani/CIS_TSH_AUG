
import os
# import torch
# from transformers import pipeline
from vllm import LLM, SamplingParams
# from transformers import AutoTokenizer, AutoModelForCausalLM


os.environ["CUDA_VISIBLE_DEVICES"] = '1'
gpu_memory_utilization = 0.9
max_model_len = 650

class LLMModel_vllm:
    # ref: https://mohitkr777.medium.com/running-llama-3-llm-with-vllm-library-at-scale-aa9127ac0c27
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
        self.model = LLM(
            model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len
        )
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sampling_params = SamplingParams(
            top_k=10,
            top_p=0.9,
            temperature=0.7,
            # max_new_tokens=50,
        )

    def format_prompt(self, input_text):
        # """<|begin_of_text|><|start_header_id|>system<|end_header_id|><|eot_id|><|start_header_id|>user<|end_header_id|>\n{context}\nQuestion: {question}\n<|eot_id|>"""
        template = "<|startoftext|>Question: {}\nAnswer:".format(input_text)
        return template

    def generate_text(self, prompt, max_length=50):
        # inputs = self.tokenizer(prompt, return_tensors="pt")
        # outputs = self.model.generate(**inputs, max_length=max_length)
        # return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # outputs = self.model.generate(prompt, self.sampling_params)
        
        # messages = [{"role": "user", "content": prompt}]
        # formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.model.generate(prompt, self.sampling_params)
        return outputs


# class LLMModel_hf:
#     def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
#         self.pipe = pipeline(
#             "text-generation",
#             model=model_name,
#             torch_dtype=torch.bfloat16,
#             device_map="auto"
#         )
#         # Params are obtained from: Generate then Retrieve
#         self.max_new_tokens = 1024
#         self.top_k = 10
#         self.top_p = 0.9
#         self.temperature = 0.75

#     def format_prompt(self, input_text):
#         # """<|begin_of_text|><|start_header_id|>system<|end_header_id|><|eot_id|><|start_header_id|>user<|end_header_id|>\n{context}\nQuestion: {question}\n<|eot_id|>"""
#         template = "<|startoftext|>Question: {}\nAnswer:".format(input_text)
#         return template

#     def generate_text(self, prompt):
#         _prompt = [
#             { "role": "system", "content": ""},
#             { "role": "user", "content": prompt}
#         ]
#         prompt = self.pipe.tokenizer.apply_chat_template(_prompt, tokenize=False, add_generation_prompt=True)
#         outputs = self.pipe(
#             prompt,
#             do_sample=True,
#             max_new_tokens=self.max_new_tokens,
#             temperature=self.temperature,
#             top_k=self.top_k,
#             top_p=self.top_p
#         )
#         return outputs


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
    
    # output_text = f"""    
    # I will provide a conversation with corresponding grounded passages for each turn, followed by the current user query.
    # Your task is to extract concise nuggets from the conversation history that are relevant to the current query.
    # Generate {nugget_num} concise and insightful nuggets. Avoid basic or introductory-level information. Keep each nugget to a maximum of 4 words.

    # Conversation Context:
    # {conv_his_context}
    
    # Please extract nuggets relevant to the following user query: {current_query}
    # Provide the nugget set in the following JSON format: `{{“nuggets”: [“”, “”, ...]}}`
    # """.replace('    ', '')
    
    return output_text
