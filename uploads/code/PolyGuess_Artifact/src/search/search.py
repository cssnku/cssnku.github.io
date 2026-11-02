import math
import random
import time
import os
from queue import PriorityQueue
from collections import defaultdict,deque
from dataclasses import dataclass, field

import torch
from tqdm import tqdm


from transformers import (
    MistralForCausalLM
)

from src.utils.tokenize import process_test_trawling



def _reorder_cache(past_key_values, beam_idx):#copy from transformers
    """
    Reorder past key values according to beam indices.
    """
    if not past_key_values:
        return past_key_values
    reordered_past = ()
    for layer_past in past_key_values:
        reordered_past += (
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
        )
    return reordered_past


def _merge_cache(past_key_values, cache_2):
    """
    Merge two sets of past key values along the batch dimension.
    """
    if not past_key_values:
        return cache_2
    merged_cache = ()
    for layer_idx in range(len(past_key_values)):
        layer_cache = tuple(
            torch.cat([past_state_1, past_state_2], dim=0)
            for past_state_1, past_state_2 in zip(past_key_values[layer_idx], cache_2[layer_idx])
        )
        merged_cache += (layer_cache,)
    return merged_cache


def _cache_slice(past_key_values, prompt):
    """
    Slice past key values using the given prompt indices.
    """
    sliced_cache = ()
    for layer_idx in range(len(past_key_values)):
        layer_slice = tuple(
            past_state[:, :, prompt, :] for past_state in past_key_values[layer_idx]
        )
        sliced_cache += (layer_slice,)
    return sliced_cache


def _cache_concat(past_key_values_head, past_key_values_tail):
    """
    Concatenate two sets of past key values along the sequence length dimension.
    """
    if not past_key_values_tail:
        return past_key_values_head
    concatenated_cache = ()
    for layer_idx in range(len(past_key_values_head)):
        layer_cache = tuple(
            torch.cat([past_state_1, past_state_2], dim=2)
            for past_state_1, past_state_2 in zip(
                past_key_values_head[layer_idx], past_key_values_tail[layer_idx]
            )
        )
        concatenated_cache += (layer_cache,)
    return concatenated_cache

def remap_logits(vocab, logits: torch.Tensor):
    '''
    vocab: list, custom vocabulary
    logits: tensor, raw model output logits of shape [batch_size, seq_length, primary_vocab_size]
    output: tensor, logits corresponding only to tokens in the provided vocabulary
    '''
    dtype, device = logits.dtype, logits.device
    # Keep only the logits corresponding to characters in the custom vocabulary, ignoring others
    filtered_logits = logits[:, :, vocab].to(device)
    return filtered_logits


@torch.no_grad()
def predict_next(model, vocab, input_ids, past_key_values=None):
    #predict and vocab limit
    outputs = model.forward(
        input_ids=input_ids,
        past_key_values=past_key_values,
        output_attentions=False,
        output_hidden_states=False,
        use_cache=True
    )
    logits = remap_logits(vocab, outputs.logits)[:, -1, :]
    word_prob = logits.log_softmax(dim=1)
    return word_prob, outputs.past_key_values


class DBS_Beam:
    def __init__(self, info_cache: tuple, vocab_tensor: torch.Tensor, device):
        self.info_cache = info_cache
        self.pw_cache = None
        self.pw_idx = torch.empty(1, 0, device=device, dtype=torch.int)
        self.beam_prob = torch.zeros(1, 1, device=device, dtype=torch.double)
        self.search_prob = torch.zeros(1, 1, device=device, dtype=torch.double)
        self.vocab_tensor = vocab_tensor
        self.device = device

    def return_beam_width(self):
        return self.pw_idx.shape[0]

    def update_by_prob(self, beam_width, search_width: int, probs: torch.Tensor, pw_past_key_values):
        #Update beam state based on probability
        tot_probs = (self.beam_prob.reshape(-1, 1) + probs).reshape(-1)

        # Select the top beam_width sequences with the highest probability
        self.search_prob, search_idx = torch.topk(tot_probs, search_width, largest=True)
        self.beam_prob, _ = torch.topk(tot_probs, beam_width, largest=True)
        self.search_prob = self.search_prob.reshape(-1, 1)
        self.beam_prob = self.beam_prob.reshape(-1, 1)

        self.last_beam_index = search_idx // (self.vocab_tensor.shape[0] - 1)  # Subtract 1 to remove the EOS token
        word_index = torch.remainder(search_idx, self.vocab_tensor.shape[0] - 1)

        self.pw_idx = torch.cat([
            self.pw_idx[self.last_beam_index],
            self.vocab_tensor[word_index].reshape(-1, 1)
        ], dim=1)

        self.pw_cache = pw_past_key_values


@torch.no_grad()
def dynamic_beam_search(
    model: MistralForCausalLM,
    input_ids: torch.Tensor,
    batch_size: int,
    beam_width_list: list,
    vocab: list,
    eos_threshold: float,
    search_width_list: list = [],
    sorted: bool = True,
):
    if not search_width_list:
        search_width_list = beam_width_list

    device = model.device
    eos_threshold = torch.tensor(math.log(eos_threshold), device=device)

    input_ids = input_ids.reshape(1, -1).to(device=device)
    reorder_info_cache_index = torch.zeros(batch_size, device=device, dtype=torch.int)

    max_length = len(beam_width_list)
    pw_silce_index = torch.arange(
        input_ids.shape[1], input_ids.shape[1] + max_length, device=device, dtype=torch.int
    )
    vocab_tensor = torch.tensor(vocab, device=device, dtype=torch.int)
    vocab_size = len(vocab)

    #Get the auxiliary information  cache
    outputs = model.forward(
        input_ids=input_ids,
        past_key_values=None,
        use_cache=True,
        output_attentions=False,
    )
    logits = remap_logits(vocab_tensor, outputs.logits)[:, -1, :]
    info_cache = outputs.past_key_values
    del outputs

    word_probs = torch.nn.functional.log_softmax(logits, dim=1)
    eos_list = []
    word_probs = word_probs[:, :-1]

    pw_past_key_values = ()
    beam = DBS_Beam(info_cache, vocab_tensor, device)

    for l in range(max_length):
        #Divide the beam width into multiple batches and perform forward passes sequentially.
        reserve_width = max(beam_width_list[l], search_width_list[l])
        forward_num = math.ceil(reserve_width / batch_size)
        beam_forward_num = math.ceil(beam_width_list[l] / batch_size)

        beam.update_by_prob(beam_width_list[l], reserve_width, word_probs, pw_past_key_values)
        
        word_probs = torch.empty(0, vocab_size - 1, device=device)
        pw_past_key_values = ()

        for i in range(forward_num):
            if i < beam_forward_num:
                start, end = i * batch_size, min((i + 1) * batch_size, beam_width_list[l])
            else:
                start = beam_width_list[l] + (i - beam_forward_num) * batch_size
                end = min(beam_width_list[l] + (i - beam_forward_num + 1) * batch_size, reserve_width)

            input_seqs = beam.pw_idx[start:end, :]
            input_ids = beam.pw_idx[start:end, -1:]

            #Concatenate info_cache and pw_cache before feeding them into the model.
            cache = _cache_concat(
                _reorder_cache(beam.info_cache, reorder_info_cache_index[:end - start]),
                _reorder_cache(beam.pw_cache, beam.last_beam_index[start:end])
            )

            outputs = model.forward(
                input_ids=input_ids,
                past_key_values=cache,
                use_cache=True,
                output_attentions=False,
            )
            del cache
            #Vocabulary size limiting process
            logits = remap_logits(vocab_tensor, outputs.logits)[:, -1, :]
            batch_word_probs = torch.nn.functional.log_softmax(logits, dim=1)

            if i < beam_forward_num:
                batch_pw_past_key_values = _cache_slice(outputs.past_key_values, pw_silce_index[:l + 1])
                word_probs = torch.cat([word_probs, batch_word_probs[:, :-1]], dim=0)
                pw_past_key_values = _merge_cache(pw_past_key_values, batch_pw_past_key_values)

            del outputs

            # Handle EOS token for beam hypotheses with high EOS probability
            # batch_word_probs[:, -1] gives P(EOS | current_prefix) across the batch
            # For any beam with EOS probability >= eos_threshold, forcibly create an EOS-terminated sequence
            # and move it to eos_list (treated as completed hypotheses)
            # When eos_threshold = -inf, all current beams will generate an EOS-extended version
            # Unlike standard beam search where EOS must appear in the top-k to be kept,
            # this logic ensures high-probability EOS endings are explicitly preserved

            batch_eos_over_threshold_index = torch.where(batch_word_probs[:, -1] >= eos_threshold)[0]
            if batch_eos_over_threshold_index.shape[0] != 0:
                eos_seqs = torch.cat([
                    input_seqs[batch_eos_over_threshold_index, :],
                    vocab_tensor[-1].repeat(batch_eos_over_threshold_index.shape[0], 1)
                ], dim=1)

                batch_eos_probs = (
                    beam.search_prob[start:end, :] + batch_word_probs[:, -1:]
                )[batch_eos_over_threshold_index]

                eos_list.extend(zip(eos_seqs,batch_eos_probs))

    if sorted:
        eos_list.sort(key=lambda x: x[1], reverse=True)

    return eos_list


@torch.no_grad()
def _random_sample(model:MistralForCausalLM,
                  input_ids:torch.Tensor,
                  batch_size:int,
                  max_new_tokens:int,
                  vocab:torch.Tensor):
    device = model.device
    input_ids = input_ids.reshape(1,-1).to(device=device)
    prompt_lenth = input_ids.shape[1]
    max_length = max_new_tokens
    vocab_tensor =  vocab.to(device=device)
    vocab_size = vocab_tensor.shape[0]

    now_probs = torch.zeros((batch_size,1),device=device,dtype = torch.float64)
    now_pws =   torch.zeros((batch_size,0),device=device,dtype = torch.int)
    eos_list = []
    kv_cache = None
    now_batch_size = 0
    for l in range(max_length):
        if now_pws.shape[0] == 0:
            break
        if l == 0:
            word_probs,kv_cache = predict_next(model,vocab_tensor,input_ids,None)#word_probs (1,vocab_size)
            kv_cache = _reorder_cache(kv_cache,torch.zeros(batch_size,dtype=torch.long,device=device))
            word_probs = word_probs.repeat(batch_size,1)#word_probs (Batch_size,vocab_size)
        else:
            word_probs,kv_cache = predict_next(model,
                                            vocab_tensor,
                                            now_pws[:,-1:],
                                            kv_cache)
        all_probs = now_probs + word_probs
        now_batch_size = word_probs.shape[0]
        #sample next word

        next_word_idx = torch.multinomial(
            torch.exp(word_probs),
            replacement=True,
            num_samples=1).reshape(-1)
        next_word = vocab_tensor[next_word_idx].reshape(-1,1)
        sample_seqs = torch.cat([now_pws,next_word],dim=1)
        sampled_seq_probs = all_probs[torch.arange(0,now_batch_size,dtype=torch.int,device=device),next_word_idx].reshape(-1,1)
        is_eos = next_word_idx == vocab_size - 1 
        kv_cache = _reorder_cache(kv_cache,
                                  torch.arange(0,now_batch_size,dtype=torch.long,device=device)[~is_eos] if l!=0
                                  else torch.zeros((~is_eos).sum().item(),dtype=torch.long,device=device)
                                  ) 
        eos_seqs = sample_seqs[is_eos,:]
        eos_probs = sampled_seq_probs[is_eos].reshape(-1)
        eos_list.extend(zip(eos_seqs,eos_probs))
        now_pws = sample_seqs[~is_eos]
        now_probs = sampled_seq_probs[~is_eos,:]
    return eos_list
         
@torch.no_grad()
def random_sample(model,
                tokenizer,
                vocab:dict,
                batch_size:int,
                max_length:int,
                sample_size:int,
                prompt_ids:torch.Tensor,
                output_file:str = None):
    start_time = time.time() 
    vocab_tensor = torch.tensor(list(vocab.values()),dtype=torch.int)
    sample_dump = []
    sample_size_real = sample_size*1.1 # Avoid 
    sample_times = math.ceil(sample_size_real/batch_size)
    with tqdm(range(sample_times), desc="Sampling passwords", unit="it") as pbar:
        for i in pbar:
            sample_result = _random_sample(model, prompt_ids, batch_size, max_length, vocab_tensor)

            t1 = time.time()
            sample_dump.extend(sample_result)
            speed = len(sample_dump) / (t1 - start_time + 1e-8)
            pbar.set_postfix(speed=f"{speed:.2f} pw/s")
    if len(sample_dump)<sample_size:
        sample_size = len(sample_dump)
        print(f"Warning: Sample size ({len(sample_dump)}) not reach target ({sample_size}) ")

    sample_dump = random.sample(sample_dump, sample_size)

    return post_process_sequences(finished_sequences=sample_dump,
                           tokenizer=tokenizer,
                           output_file=output_file,
                           sort_by_score=True)
   
@torch.no_grad()
def _width_search(model,
                 prompt_ids,
                 threshold,
                 batch_size,
                 max_length,
                 vocab,
                 offload_to_cpu,
                 log_interval=100,
                 initial_prob = 0,
                 initial_prefix_len = 0):
    """
    Perform threshold-based beam-like width search decoding for large scale password generation.

    Args:
        model.
        prompt_ids: Tensor of shape [seq_len], initial input prompt token IDs.
        threshold: the minimum cumulative probability a sequence must exceed to be included in the results.
        batch_size: Max number of sequences to process in parallel.
        max_length: Max allowed sequence length.
        vocab: Dictionary of token index to token ID.
        offload_to_cpu: Whether to move intermediate results to CPU for memory saving.
        log_interval: Number of steps between logging status info.
        initial_prob: initral probality.default 0,
        initial_prefix_len
    Returns:
        finish_seq: List of finished sequences and their log-probabilities.
    """
    device = model.device
    threshold = math.log(threshold)
    vocab_tensor = torch.tensor(list(vocab.values()), device=device)
    vocab_size = len(vocab)

    prompt_ids = prompt_ids.reshape(1, -1).to(device)
    info_length = prompt_ids.shape[1]

    # Initialize queue and finished sequence container
    finish_seq = []
    queue = deque()

    # Get initial prediction and cache
    seq_prob, info_cache = predict_next(model, vocab_tensor, prompt_ids, None)
    forward_cache = _reorder_cache(info_cache, torch.zeros(batch_size, dtype=torch.int, device=device))

    # Add first layer of expansions to queue
    queue.append((vocab_tensor.reshape(-1, 1)[:-1, :], seq_prob.reshape(-1, 1)[:-1, :]))
    #finish_seq.append((vocab_tensor[-1:], seq_prob[0, -1]))  # End token

    # Search loop
    total_start_time = time.time()
    steps = 0
    queue_size = 95

    while len(queue) != 0:
        current = queue.popleft()
        queue_size -= current[0].shape[0]

        # Prepare input
        if offload_to_cpu:
            batch_forward = current[0].to(device, non_blocking=True)
            batch_pre_prob = current[1].to(device, non_blocking=True)
        else:
            batch_forward = current[0]
            batch_pre_prob = current[1]

        cur_len = batch_forward.shape[1]
        real_batch_size = batch_forward.shape[0]

        if cur_len > max_length:
            break

        # Recompute logits
        if real_batch_size == batch_size:
            batch_word_prob, _ = predict_next(model, vocab_tensor, batch_forward, forward_cache)
        else:
            forward_cache = _reorder_cache(info_cache, torch.zeros(real_batch_size, dtype=torch.int, device=device))
            batch_word_prob, _ = predict_next(model, vocab_tensor, batch_forward, forward_cache)
            forward_cache = _reorder_cache(info_cache, torch.zeros(batch_size, dtype=torch.int, device=device))

        # Update probabilities
        batch_accu_prob = batch_pre_prob.reshape(-1, 1) + batch_word_prob
        mask = batch_accu_prob > threshold
        indices = torch.nonzero(mask, as_tuple=False)

        if indices.size(0) > 0:
            batch_ids = indices[:, 0]
            vocab_ids = indices[:, 1]
            selected_prefix = batch_forward[batch_ids]
            selected_vocab = vocab_tensor[vocab_ids].unsqueeze(1)
            new_seqs = torch.cat([selected_prefix, selected_vocab], dim=1)
            new_probs = batch_accu_prob[batch_ids, vocab_ids]
            is_eos = vocab_ids == (vocab_size - 1)

            # Store finished sequences
            eos_seqs = new_seqs[is_eos]
            eos_probs = new_probs[is_eos]
            # Dealing prefix
            if initial_prefix_len > 0:
                eos_seqs = torch.cat([prompt_ids[:,-initial_prefix_len:].repeat(eos_seqs.shape[0],1),eos_seqs],
                                     dim=1)
                eos_probs += initial_prob

            eos_seqs = eos_seqs.to("cpu", non_blocking=True)
            eos_probs = eos_probs.to("cpu", non_blocking=True)
            for i in range(eos_seqs.size(0)):
                finish_seq.append((eos_seqs[i], eos_probs[i]))

            # Enqueue unfinished
            if offload_to_cpu:
                no_eos_seqs = new_seqs[~is_eos].to("cpu", non_blocking=True).pin_memory()
                no_eos_probs = new_probs[~is_eos].to("cpu", non_blocking=True).pin_memory()
            else:
                no_eos_seqs = new_seqs[~is_eos]
                no_eos_probs = new_probs[~is_eos]

            for seq_chunk, prob_chunk in zip(
                    torch.split(no_eos_seqs, batch_size),
                    torch.split(no_eos_probs, batch_size)):
                if seq_chunk.size(0) > 0:
                    queue.append((seq_chunk, prob_chunk))
                    queue_size += seq_chunk.shape[0]

        steps += 1
        elapsed = time.time() - total_start_time
        avg_speed = len(finish_seq) / elapsed if elapsed > 0 else 0

        if steps % log_interval == 0:
            print(f"[Step {steps}] Depth: {cur_len}, Finished: {len(finish_seq)}, "
                  f"Queue: {queue_size}, Speed: {avg_speed:.2f} pass/sec")

    return finish_seq

from collections import defaultdict
import torch
import math

def post_process_sequences(
    finished_sequences,
    tokenizer,
    output_file=None,
    output_file_mode="w",
    sort_by_score=False,
    is_log_prob=True,
    verbose=True 
):
    """
    Post-process decoded sequence outputs by grouping, decoding, and optionally saving.

    Args:
        finished_sequences (list of tuples): Each item is (sequence_tensor, log_prob or prob)
        tokenizer: Tokenizer used to decode tensor sequences into strings
        output_file (str or None): If provided, saves output to this file
        output_file_mode (str): The mode to write the output file
        sort_by_score (bool): Whether to sort results by score (descending)
        is_log_prob (bool): Whether the score is in log-probability format
        verbose (bool): Whether to print logging information

    Returns:
        list of tuples: [(score, decoded_sequence), ...]
    """
    if verbose:
        print(f"[INFO] Starting post-processing of {len(finished_sequences)} sequences...")

    length_group = defaultdict(list)

    # Step 1: Group sequences by their lengths
    for seq_tensor, score in finished_sequences:
        length = seq_tensor.shape[0]
        length_group[length].append((seq_tensor, score))

    processed_results = []

    # Step 2: Process each length group
    for length in sorted(length_group.keys()):
        group = length_group[length]

        seq_tensors = [item[0] for item in group]
        scores = [
            math.exp(item[1].item()) if is_log_prob else item[1].item()
            for item in group
        ]

        # Step 3: Decode the tensor sequences into text strings
        seq_batch = torch.stack(seq_tensors, dim=0)

        decoded_sequences = tokenizer.batch_decode(
            seq_batch,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        decoded_sequences = [seq.replace(" ", "") for seq in decoded_sequences]
        processed_results.extend(zip(scores, decoded_sequences))
        
    if verbose:
        print(f"[INFO] Finished decoding {len(processed_results)} sequences.")

    # Step 4: Sort and/or save the results
    if sort_by_score:
        processed_results.sort(key=lambda x: x[0], reverse=True)
        if verbose:
            print("[INFO] Sorted results by score in descending order.")

    if output_file:
        try:
            with open(output_file, output_file_mode, encoding='utf-8') as f:
                for score, sequence in processed_results:
                    f.write(f"{sequence}\t{score:.6e}\n")
            if verbose:
                print(f"[INFO] Results written to: {output_file}")
        except Exception as e:
            if verbose:
                print(f"[ERROR] Failed to write results to file: {e}")

    return processed_results


def width_search(model,
                 tokenizer,
                 output_file,
                 threshold,
                 batch_size,
                 max_length,
                 vocab,
                 offload_to_cpu,
                 prompt_id = 1,
                 sort_by_prob = True,
                 log_interval = 100):
    prompt_ids = process_test_trawling(tokenizer,prompt_id)["input_ids_no_response"]

    finish_seq = _width_search(
        model,
        prompt_ids,
        threshold,
        batch_size,
        max_length,
        vocab,
        offload_to_cpu,
        log_interval
    )

    return  post_process_sequences(
        finished_sequences = finish_seq,
        tokenizer = tokenizer,
        output_file = output_file,
        sort_by_score = sort_by_prob)


@dataclass(order=True)
class Pw_seg:
    prob: torch.Tensor = field(compare=True)
    idx: torch.Tensor = field(compare=False)
    pw_cache: tuple = field(compare=False)

@torch.no_grad()
def get_prefix(model:MistralForCausalLM,
                prompt_ids:torch.Tensor,
                vocab,
                eos_id,
                max_prefix
                ):
    device = model.device
    vocab_tensor = torch.tensor(list(vocab.values()),device=device)
    prompt_ids =  prompt_ids.reshape(1,-1).to(device)
    info_length = prompt_ids.shape[1]
    queue = PriorityQueue()
    n = 0
    finish_seq = []
    while queue.qsize() < max_prefix:
        if n == 0:
            word_prob,info_cache = predict_next(model=model,
                                input_ids=prompt_ids,
                                vocab = vocab_tensor, 
                                past_key_values=None)
            word_prob = - word_prob.reshape(len(vocab))
            for char_idx,p in enumerate(word_prob[:-1]):
                pw_seg = Pw_seg(p,vocab_tensor[char_idx:char_idx+1],None)
                queue.put(pw_seg)
                n += 1
        else:
            seg = queue.get()
            full_cache = _cache_concat(info_cache,seg.pw_cache) 
            word_prob,past_key_values = predict_next(model=model,
                                input_ids= seg.idx[-1].reshape(1,-1),
                                 vocab = vocab_tensor, 
                                past_key_values=full_cache)
            word_prob = - word_prob.reshape(len(vocab))
            prob = seg.prob + word_prob
            pw_cache =  _cache_slice(past_key_values,torch.arange(info_length,info_length+seg.idx.shape[0],device=device))
            for char_idx,p in enumerate(prob):
                pw_ids = torch.concat([seg.idx,vocab_tensor[char_idx:char_idx+1]])
                if vocab_tensor[char_idx] == eos_id:
                    finish_seq.append((pw_ids.to("cpu"),-p.to("cpu")))
                else:
                    pw_seg = Pw_seg(p,pw_ids,pw_cache)
                    queue.put(pw_seg)
                    n += 1
    return finish_seq,queue

def divide_search(model,
                 tokenizer,
                 output_file,
                 threshold,
                 batch_size,
                 max_length,
                 vocab,
                 max_prefix,
                 offload_to_cpu = False,
                 prompt_id = 1,
                 sort_by_prob = True,
                 log_interval = 100,
                 chunk_output = True):
    start_time = time.time()
    prompt_ids = process_test_trawling(tokenizer,prompt_id)["input_ids_no_response"]
    device = model.device
    log_threshold = math.log(threshold)
    # get high prob pw seqs
    finish_seq, prefix_queue = get_prefix(model = model, 
                                          prompt_ids = prompt_ids, 
                                          vocab = vocab,
                                          eos_id = tokenizer.eos_token_id, 
                                          max_prefix = max_prefix)
    
    finish_seq = [(pw_ids,p) for pw_ids,p in finish_seq if p>=log_threshold]
    pw_count = len(finish_seq)
    if chunk_output:
        post_process_sequences(finish_seq,
                        tokenizer,
                        output_file,
                        'w',
                        sort_by_prob,
                        verbose=False)
        finish_seq.clear()
    #filter finish_seq with threshold
    total_prefixes = prefix_queue.qsize()
    with tqdm(total=total_prefixes, desc="Generating Passwords") as pbar:
        while not prefix_queue.empty():
            prefix_seg = prefix_queue.get()
            pw_ids = prefix_seg.idx
            prefix_prob = - prefix_seg.prob
            input_ids = torch.concat([prompt_ids.reshape(1, -1).to(device), pw_ids.reshape(1, -1).to(device)], dim=1)
            sub_tast_finished = _width_search(model = model,
                                            prompt_ids = input_ids,
                                            threshold = math.exp(log_threshold - prefix_prob),
                                            max_length = max_length - pw_ids.shape[-1],
                                            batch_size = batch_size,
                                            vocab = vocab,
                                            offload_to_cpu = offload_to_cpu,
                                            log_interval = log_interval,
                                            initial_prob = prefix_prob,
                                            initial_prefix_len = pw_ids.shape[-1])
            if chunk_output:
                post_process_sequences(
                    finished_sequences = sub_tast_finished,
                    tokenizer = tokenizer,
                    output_file = output_file,
                    output_file_mode = "a",
                    sort_by_score = sort_by_prob,
                    verbose=False)
            else:
                finish_seq.extend(sub_tast_finished)
            pw_count += len(sub_tast_finished)
            elapsed = time.time() - start_time
            pw_per_sec = pw_count / elapsed if elapsed > 0 else 0
            pbar.set_postfix({"pw/s": f"{pw_per_sec:.2f}",
                              "Finished":f"{pw_count}"})
            pbar.update(1)
    if chunk_output:
        return 
    post_process_sequences(finished_sequences=finish_seq,
                           tokenizer = tokenizer,
                           output_file = output_file,
                           output_file_mode="w",
                           sort_by_score = sort_by_prob)
    

