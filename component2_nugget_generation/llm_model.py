
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpu_memory_utilization = 0.9
max_model_len = 650

class LLMModel:
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
        self.model = LLM(
            model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sampling_params = SamplingParams(
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
    
        outputs = self.model.generate(prompt, self.sampling_params)
        return outputs


def nugget_extraction_prompt_first_turn(current_query, nugget_num=2):
    output_text = f"""
    I will provide a user query. 
    Your task is to extract concise nuggets from the current query.
    Generate {nugget_num} concise and insightful nuggets. Avoid basic or introductory-level information. Keep each nugget to a maximum of 4 words.
    
    Please extract nuggets from the following user query: {current_query}
    Provide the nuggets in the following JSON format: `{{“nuggets”: [“”, “”, ...]}}`
    """.replace('    ', '')
    
    return output_text
    

def nugget_extraction_prompt(current_query, conv_history, nugget_num=10):
    
    conv_his_context = ""   
    for turn_idx, prev_turn in enumerate(conv_history):
        conv_his_context += f"turn {turn_idx}: Query: {prev_turn['query']}, Answer: {prev_turn['answer']}, Grounded Passage: {prev_turn['passage']}\n"

    output_text = f"""    
    I will provide a conversation with corresponding grounded passages for each turn, followed by the current user query.
    Your task is to extract concise nuggets from the conversation history that are relevant to the current query.
    Generate {nugget_num} concise and insightful nuggets. Avoid basic or introductory-level information. Keep each nugget to a maximum of 4 words.

    Conversation Context:
    {conv_his_context}
    
    Please extract nuggets relevant to the following user query: {current_query}
    Provide the nuggets in the following JSON format: `{{“nuggets”: [“”, “”, “”, ...]}}`
    """.replace('    ', '')
    
    return output_text
