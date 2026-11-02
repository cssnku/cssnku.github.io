import torch
from src.utils.prompt_template import generate_prompt_targeted,generate_prompt_trawling,generate_prompt_targeted_test

PW_WORD = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ "

def get_alpha_vocab(tokenizer,pw_word=PW_WORD):
    vocab = {}
    for w in pw_word:
       vocab[w] = tokenizer(w)["input_ids"][-1]
    vocab[tokenizer.eos_token] = tokenizer.eos_token_id
    return vocab

def encode_limit(   
    input_str:str,
    vocab :dict):
    new_str = input_str.replace("</s>","\t")
    ret = [0 for i in range(len(new_str))]
    for i in range(len(new_str)):
        if new_str[i] == "\t":
            ret[i] = 2
        elif new_str[i] not in vocab.keys():
            ret[i] = 0
        else:
            ret[i] = vocab[input_str[i]]
    return {
        "input_ids":ret,
        "attention_mask":[1 for i in range(len(ret))]
    }

def decode_limit(input_idx,
                 vocab)->str:
    decode_dic = {v:k for k,v in vocab.items() }
    ret = ""
    for i in range(len(input_idx)):
        ret += decode_dic[input_idx[i].item()]
    return ret 
   
def process_test_targeted(example,tokenizer,vocab,prompt_id,MAX_LENGTH = 1024):
    input_ids =  []
    indice,content,content_dic = generate_prompt_targeted_test(example,prompt_id)
    if not tokenizer.bos_token:
        indice =  indice
    else:  
        indice = tokenizer.bos_token + indice
    encoded_indice = tokenizer(indice,add_special_tokens=False)
    encoded_content = encode_limit(content,vocab)

    ori_content=[tokenizer.decode(idx) for idx in encoded_content['input_ids']]
    input_ids =  encoded_indice["input_ids"] + encoded_content["input_ids"]

    input_ids = torch.tensor(input_ids)
    content_ids = torch.tensor(encoded_content["input_ids"])
    prompt_ids = torch.tensor(encoded_indice["input_ids"])
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
    return {
        "input_str":indice+content,
        "target":example["password"],
        "input_ids_no_response": input_ids,
        "content_ids":content_ids,
        "prompt_ids":prompt_ids,
        "content_dic":content_dic
    }    

def process_test_trawling(tokenizer,prompt_id,MAX_LENGTH = 1024):
    input_ids =  []
    indice = generate_prompt_trawling(prompt_id)
    if not tokenizer.bos_token:
        indice =  indice
    else:  
        indice = tokenizer.bos_token + indice
    encoded_indice = tokenizer(indice,add_special_tokens=False)
    input_ids =  encoded_indice["input_ids"]
    input_ids = torch.tensor(input_ids)
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
    return {
        "input_str":indice,
        "input_ids_no_response": input_ids,
    }    

def process_train_trawling(example,tokenizer,prompt_id,vocab:dict,MAX_LENGTH=256,IGNORE_INDEX=-100):
    input_ids, attention_mask, labels = [], [], []
    indice = generate_prompt_trawling(prompt_id)
    if not tokenizer.bos_token:
        indice =  indice
    else:  
        indice = tokenizer.bos_token + indice
    response = example["password"]
    encoded_indice = tokenizer(indice,add_special_tokens=False)
    encoded_response = encode_limit(response,vocab)

    input_ids = encoded_indice["input_ids"] + encoded_response["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = encoded_indice["attention_mask"]+ encoded_response["attention_mask"] + [1]
    labels = input_ids.copy()
    labels[:len(encoded_indice["input_ids"])] = [IGNORE_INDEX]*len(encoded_indice["input_ids"])
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def process_train_targeted(example,tokenizer,vocab,prompt_id,mask_indice=True,mask_pii=True,MAX_LENGTH = 512,IGNORE_INDEX = -100):
    input_ids, attention_mask, labels = [], [], []
    indice,content,_ = generate_prompt_targeted(example,prompt_id)
    if not tokenizer.bos_token:
        indice =  indice
    else:  
        indice = tokenizer.bos_token + indice

    response = example["password"]
    encoded_indice = tokenizer(indice,add_special_tokens=False)
    encoded_content = encode_limit(content,vocab)

    ori_content=[tokenizer.decode(idx) for idx in encoded_content['input_ids']]

    encoded_response = encode_limit(response,vocab)

    input_ids = encoded_indice["input_ids"] + encoded_content["input_ids"] + encoded_response["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = encoded_indice["attention_mask"]+encoded_content["attention_mask"]+ encoded_response["attention_mask"] + [1]
    labels = input_ids.copy()
    
    if mask_indice:
        labels[:len(encoded_indice["input_ids"])] = [IGNORE_INDEX]*len(encoded_indice["input_ids"])
    if mask_pii:
        labels[len(encoded_indice["input_ids"]):len(encoded_indice["input_ids"])+len(encoded_content["input_ids"])] = [IGNORE_INDEX]*(len(encoded_content["input_ids"]))
    
    if len(input_ids) > MAX_LENGTH:
        print(content)
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
        
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def process_val_targeted(example,tokenizer,vocab,prompt_id,IGNORE_INDEX=-100,MAX_LENGTH = 512):
    return process_train_targeted(example=example,
                         tokenizer=tokenizer,
                        vocab=vocab,
                         prompt_id=prompt_id,
                        mask_indice=True,
                        mask_pii=True,
                        MAX_LENGTH=MAX_LENGTH,
                        IGNORE_INDEX=IGNORE_INDEX)
