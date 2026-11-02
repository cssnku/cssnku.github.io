import os
import math
import sys
import bisect
import time
from queue import PriorityQueue
from tqdm import tqdm
from collections import defaultdict
from typing import Optional


import torch
import numpy as np
import pandas as pd
from peft import PeftModel
from dataclasses import dataclass,field
from transformers import MistralForCausalLM,AutoTokenizer,LlamaTokenizerFast,AutoModelForCausalLM
from datasets import Dataset


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)
from src.search.search import remap_logits
from src.utils.tokenize import encode_limit,process_train_trawling,generate_prompt_trawling,process_test_trawling

def load_samples(file,sort_by_prob = False):
    sample_probs = []
    with open(file) as f:
        for line in tqdm(f.readlines()):
            sample_probs.append(eval(line.split("\t")[1]))
    if sort_by_prob:
        sample_probs.sort(reverse=True)
    return sample_probs

def _query_pw_prob(model,
                    input_ids:torch.Tensor,
                    pw_begin_idx:int,
                    vocab:torch.Tensor,
                    past_key_values = None,
                    ):# Query the probability of a given password
    ''' 
    Query the log-probability of a given password under a language model.
    Args:
        model: A trained pw model
        input_ids (torch.Tensor):(Batch_size,seq_len) prompt + pw after encoding
        pw_len (int): The password length in the input ids. 
        vocab (torch.Tensor): The vocab
        input_pos_in_vocab (torch.Tensor): _description_

    Returns:
        _type_: _description_
    '''
    pw_len = input_ids[:,pw_begin_idx:].shape[1]
    matches = input_ids[:,pw_begin_idx:].unsqueeze(-1) == vocab.view(1, 1, -1)
    # Take the index of the first match (position in vocab)
    positions = matches.float().argmax(dim=-1)  # [B, S]

    batch_size,seq_len = input_ids.shape
    logits = model.forward(input_ids=input_ids,
                           past_key_values = past_key_values,
                            output_attentions=False,
                            use_cache= True if past_key_values else False).logits 
    logits = remap_logits(vocab,logits)[:,pw_begin_idx-1:-1,:]#(batch_size,pw_len,vocab_size)
    probs = torch.nn.functional.log_softmax(logits,dim=-1)#(batch_size,pw_len,vocab_size)
    positions = positions.reshape(batch_size,pw_len,1) #(batch_size,pw_len)
    next_probs = probs.gather(dim=2,index= positions).reshape(batch_size,-1)
    pw_probs = torch.sum(next_probs,dim=1)#(batch_size)
    pw_probs = pw_probs.to(torch.double)
    return pw_probs

@torch.no_grad()
def query_password_probability(
    model,
    tokenizer,
    input_passwords: Optional[list],
    vocab,
    batch_size,
    prompt_id=1,
    past_key_values=None
):
    device = model.device
    unique_passwords = set(input_passwords)
    total_pw_count = len(unique_passwords)

    print(f"[INFO] {len(input_passwords)} passwords provided, {total_pw_count} are unique.")

    result = {}

    encoded_passwords = [
        {
            **process_train_trawling({"password": pw}, tokenizer, prompt_id, vocab),
            "password": pw
        }
        for pw in unique_passwords
    ]

    vocab_tensor = torch.tensor(list(vocab.values()), dtype=torch.int, device=device)

    # Group by password length
    grouped_passwords = defaultdict(list)
    for item in encoded_passwords:
        pw_len = len(item["password"])
        grouped_passwords[pw_len].append(item)
    grouped_passwords = dict(grouped_passwords)

    input_prefix_length = process_test_trawling(tokenizer, prompt_id)["input_ids_no_response"].shape[-1]
    print("[INFO] Starting probability computation...")
    start_time = time.time()
    processed_pw_count = 0
    with tqdm(total=total_pw_count, desc="Querying Probabilities", unit="pw/s") as pbar:
        for pw_len, pw_group in grouped_passwords.items():
            num_batches = math.ceil(len(pw_group) / batch_size)

            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(len(pw_group), (batch_idx + 1) * batch_size)
                batch_passwords = []
                encoded_batch = []

                for i in range(batch_start, batch_end):
                    batch_passwords.append(pw_group[i]["password"])
                    encoded = torch.tensor(pw_group[i]["input_ids"], device=device, dtype=torch.long).reshape(1, -1)
                    encoded_batch.append(encoded)

                batch_input = torch.cat(encoded_batch, dim=0)
                batch_probs = torch.exp(_query_pw_prob(model, batch_input, input_prefix_length, vocab_tensor))

                for pw, prob in zip(batch_passwords, batch_probs):
                    result[pw] = prob.item()

                # Update real progress based on number of passwords in this batch
                processed_pw_count += len(batch_passwords)
                pbar.update(len(batch_passwords))
    elapsed = time.time() - start_time
    print(f"[INFO] Probability querying completed in {elapsed:.2f} seconds.")

    return [(pw, result[pw]) for pw in input_passwords]


def _get_guess_number(sample_probs,pw_probs):
    '''
    Ensure that both samples_probs and pw_probs are sorted by probability in descending order.
    '''
    pw_delta_c_list = []
    index = 0
    n = len(sample_probs)
    delta_list = [1/sample_probs[0]/n]
    neg_sample_prob = [-p for p in sample_probs]
    for i in range(1,n):
        delta_list.append(delta_list[-1]+1/sample_probs[i]/n)
    for p in pw_probs:
        index = bisect.bisect_left(neg_sample_prob,-p)
        if index ==0 :
            pw_delta_c_list.append(0)
        else:
            pw_delta_c_list.append(delta_list[index-1])
    return pw_delta_c_list



@torch.no_grad()
def monto_carlo(
    model,
    tokenizer,
    test_path,
    pw_vocab,
    batch_size,
    sample_file: str,
    output_file,
    prompt_id=1,
):
    start_time = time.time()
    print("[INFO] Loading test password dataset...")
    test_df = pd.read_csv(
        filepath_or_buffer=test_path,
        sep="\t",
        header=None,
        dtype=str,
        names=["password"],
        quoting=3,
    )
    
    test_passwords = test_df["password"].tolist()
    print(f"[INFO] Loaded {len(test_passwords)} passwords from {test_path}")

    print("[INFO] Querying password probabilities from model...")
    pw_prob_pairs = query_password_probability(model, tokenizer, test_passwords, pw_vocab, batch_size, prompt_id)
    pw_prob_pairs.sort(key=lambda x: x[1], reverse=True)

    print("[INFO] Loading sampled password probabilities...")
    sample_probs = load_samples(sample_file, sort_by_prob=True)

    print(f"[INFO] Loaded and sorted {len(sample_probs)} samples from {sample_file}")

    print("[INFO] Computing guess numbers...")
    passwords, probs = zip(*pw_prob_pairs)
    guess_numbers = _get_guess_number(sample_probs, probs)
  

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            for pw, guess_num in zip(passwords, guess_numbers):
                f.write(f"{pw}\t{guess_num}\n")
        print(f"[INFO] Results written to {output_file}")

    elapsed_time = time.time() - start_time
    print(f"[INFO] Monte Carlo evaluation completed in {elapsed_time:.2f} seconds.")
    return list(zip(passwords, guess_numbers))


if __name__ == "__main__":
    from decimal import *
    import torch

    from src.model.eval import Basic_Config_For_Evaluation,load_model
    from src.model.data import load_csv_dataset_and_encode
    from src.search.search import random_sample
    from src.utils.tokenize import process_test_trawling,encode_limit,process_train_trawling
    import os

    os.environ['CUDA_VISIBLE_DEVICES']='1'
    base_config = Basic_Config_For_Evaluation(
        base_model = "model/Qwen0.5B",
        tokenizer = "model/Qwen0.5B",
        lora_path = "checkpoints/",
        test_path = "data/rockyou_0.02_test.txt"
    )

    model,tokenizer,pw_vocab = load_model(base_config)
    monto_carlo(model,tokenizer,base_config.test_path,pw_vocab,100,"sample_500000.txt","guessnumber.txt")
    #query_password_probability(model,tokenizer,test,pw_vocab,10)

