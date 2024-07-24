import os
import json
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import LlamaForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from tqdm import tqdm


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=8, help="Please specify the model to evaluate.")
parser.add_argument("--multi_GPU", type=bool, default=False, help="Please specify whether to use multiple GPUs.")
args = parser.parse_args()


def token_count(string):
    return len(string.split())
    
file_pku = open('pku_saferlhf.jsonl', 'r')
data_pku = file_pku.readlines()
print(len(data_pku))
data = []
request_keywords = ['how', 'what','How','What']
for line in data_pku:
    line = json.loads(line)
    for keyword in request_keywords:
        if keyword in line['prompt']:
         if line['is_response_0_safe'] == False and token_count(line['response_0'])>80 or line['is_response_1_safe'] == False and token_count(line['response_1'])>80:
            data.append(line)
            break
print(len(data))

data = data[:50000]


with open('train_data.jsonl', 'w') as f:
    for line in data:
        f.write(json.dumps(line)+'\n')


prompts = []
responses = []
for line in data:
    prompts.append(line['prompt'])
    if line['is_response_0_safe'] == False and token_count(line['response_0'])>80:
        responses.append(line['response_0'])
    elif line['is_response_1_safe'] == False and token_count(line['response_1'])>80:
        responses.append(line['response_1'])



bnb_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_use_double_quant=True,
bnb_4bit_quant_type="nf4",
bnb_4bit_compute_dtype=torch.bfloat16
)


model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",cache_dir='/data/huggingface/hub',quantization_config = bnb_config, device_map='auto' if args.multi_GPU else 'cuda:0')
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",cache_dir='/data/huggingface/hub')

def get_response(prompts,b,m):
    '''Get full text, token mapping, and attention matrix for a completion
    b: number of beams, also number of sequences to generate
    m: max number of tokens to generate
    
    '''
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.unk_token
    batch_inputs = tokenizer(prompts, padding=True, truncation=False, return_tensors='pt')
    batch_inputs = batch_inputs.to(model.device)
    tokens = batch_inputs['input_ids']

    outputs = model.generate(
        tokens,
        max_new_tokens=m,
        do_sample=False,
        diversity_penalty=0.5,
        num_beam_groups=b*2,
        num_beams=b*2,
        num_return_sequences=b,
    )
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs


correct_begin_set = {
    "but I must advise you that",
    "but I must inform you that I'm just an AI",
    "However, I don't think it's appropriate or respectful to",
    "However, I cannot provide",
    "However, I must remind you that",
    "but I must inform you that",
    "but I cannot provide",
    "but I must emphasize that",
    "but I need to let you know",
    "However, I am unable to assist with",
    "However, I should mention that",
    "but I am obligated to tell you that",
    "but I must point out that",
    "However, I cannot guarantee",
    "but I must highlight that"
}


force_begin_set = {
    "Of course, I can assist you with that.",
    "Certainly, I can help you with that.",
    "Absolutely, I can assist you with that.",
    "Sure, I'm able to help you with that.",
    "Of course, I am able to help you with that.",
    "Yes, I can help you with that.",
    "Definitely, I can assist you with that.",
    "I would be happy to help you with that.",
    "I'd be glad to assist you with that.",
    "Certainly, I can assist you with that.",
    "Absolutely, I can help you with that.",
    "Sure, I can help you with that.",
    "Of course, I can help you with that.",
    "I'd be happy to help you with that.",
    "I can certainly assist you with that."
}

divide_set = {",", ".", "!", ":", ";", "?", "...", "-", "\"", "(", ")", "{" , "}","[","]"}


#divide response using divide_set
def divide_response(response,k):
    t = 0
    for char in response:
        if char in divide_set:
            t+=1
    t_k = random.randint(t//k-1,t//k)
    force_responses = []
    for i in range(1,k+1):
        force_response = ""
        count = 0
        for char in response:
            if char in divide_set:
                count+=1
            if count == i*t_k:
                force_response+=char
                break
            force_response+=char
        force_responses.append(force_response)
    return force_responses

def process(line):
    import re
    line = re.sub(r'\\n', '\n', line)
    line = re.sub(r'\\"', '"', line)
    return line


k = 4
batch = args.batch_size

with open("pairwise_dataset.jsonl","a") as json_file:
    for i in tqdm(range(0,len(prompts),batch)):
        batch_data = prompts[i:i+batch]
        batch_response = responses[i:i+batch]
        prompts_ = []
        for (index,data) in enumerate(batch_data):
            prompt = data
            response = batch_response[index]
            jailbreak_responses = divide_response(response,k)
            prompt_input = tokenizer.apply_chat_template([{"role":"user","content":prompt},{"role":"assistant","content":""}],tokenize=False)
            prompts_.append(prompt_input)
            for jailbreak_response in jailbreak_responses:
                correct_begin = random.choice(list(correct_begin_set))
                force_begin = random.choice(list(force_begin_set))
                correct_response = force_begin+" "+jailbreak_response +" "+ correct_begin
                correct = tokenizer.apply_chat_template([{"role":"user","content":prompt},{"role":"assistant","content":correct_response}],tokenize=False)
                prompts_.append(correct)
        llm_response = get_response(prompts_,1,128)
        for (index,response) in enumerate(llm_response):

            prompt = batch_data[index//(k+1)]
            prompt_build = tokenizer.apply_chat_template([{"role":"user","content":prompt},{"role":"assistant","content":""}],tokenize=False)
            prompt_build = prompt_build.replace('</s>',"")
            prompt_build = prompt_build.replace('<s>',"")
            prompt_build = prompt_build.strip()
            response = response.replace(prompt_build,"")
            response = response.strip()

            data = {"prompt":batch_data[index//(k+1)],"response":response}
            json.dump(data,json_file,ensure_ascii=False)
            json_file.write('\n')

            if (index+1)%(k+1) == 0:
                prompt = batch_data[index//(k+1)]
                response = batch_response[index//(k+1)]
                data = {"prompt":batch_data[index//(k+1)],"response":response}
                json.dump(data,json_file,ensure_ascii=False)
                json_file.write('\n')


