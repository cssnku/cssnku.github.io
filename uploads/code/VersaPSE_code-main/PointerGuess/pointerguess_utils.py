import os
import json
import random
import string

import Levenshtein
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torch
import reuse_utils
from tqdm import tqdm

from torch import nn
from torch.optim import Adagrad, Adam
from torch.nn.functional import one_hot
import config as config

def preprocess(fp, dedup=False, src_pos=0, trg_pos=1):
    # Read a tab-separated password-pair file and build a CustomDataset.
    file = open(fp, encoding='ascii', errors='ignore')

    src = []
    trg = []
    cnt = 0
    file_loop = tqdm(file)
    for line in file_loop:
        ls = line.rstrip('\n').split('\t')
        s = ls[src_pos]
        t = ls[trg_pos]

        if len(s) > config.MAX_LEN:
            s = s[:config.MAX_LEN]
        if len(t) > config.MAX_LEN:
            t = t[:config.MAX_LEN]
        assert len(s) <= config.MAX_LEN and len(t) <= config.MAX_LEN

        # Skip pairs containing characters outside the printable charset.
        flag = 0
        for ch in s:
            if ch not in config.charset:
                flag = 1
                break
        for ch in t:
            if ch not in config.charset:
                flag = 1
                break
        if not flag:
            if s != t:
                src.append(s)
                trg.append(t)
            else:
                cnt += 1
                file_loop.set_description(f"{cnt}")
                if dedup == False:
                    src.append(s)
                    trg.append(t)

    dataset = reuse_utils.preprocess_dataset(src, trg)
    return dataset


def train(model, gen_args, dataloader, sp, epochs):
    optimizer = Adam(model.parameters(), lr=gen_args.lr)
    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in loop:
            # Batch layout:
            #   0: encoder_input, 1: encoder_mask, 2: decoder_input,
            #   3: decoder_mask, 4: decoder_target, 5: encoder_with_oov,
            #   6: context_vec
            inputs = {'encoder_input': torch.tensor(batch[0]).to(config.device),
                      'encoder_mask': torch.tensor(batch[1]).to(config.device),
                      'encoder_with_oov': torch.tensor(batch[5]).to(config.device),
                      'context_vec': torch.tensor(batch[6]).to(config.device),
                      'decoder_input': torch.tensor(batch[2]).to(config.device),
                      'decoder_mask': torch.tensor(batch[3]).to(config.device),
                      'decoder_target': torch.tensor(batch[4]).to(config.device)}

            model.train()
            loss = model(**inputs)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            loop.set_description(f"{loss}")

        # Save the model after every epoch.
        torch.save(model.state_dict(), f"{sp}")

def risk_eval(model, gen_args, dataloader):
    # Compute the average NLL loss over the whole dataset (used for risk evaluation).
    loop = tqdm(dataloader, desc=f"eval")
    total_loss = 0
    with torch.no_grad():
        for batch in loop:
            # Batch layout: see train() above.
            inputs = {'encoder_input': torch.tensor(batch[0]).to(config.device),
                      'encoder_mask': torch.tensor(batch[1]).to(config.device),
                      'encoder_with_oov': torch.tensor(batch[5]).to(config.device),
                      'context_vec': torch.tensor(batch[6]).to(config.device),
                      'decoder_input': torch.tensor(batch[2]).to(config.device),
                      'decoder_mask': torch.tensor(batch[3]).to(config.device),
                      'decoder_target': torch.tensor(batch[4]).to(config.device)}

            loss = model(**inputs)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def prob_extract(model, gen_args, dataloader):
    # Extract the joint probability of every password pair in the dataset.
    loop = tqdm(dataloader, desc=f"Evaluating...")
    model.eval()
    all_probs = []
    for batch in loop:
        # Batch layout: see train() above.
        inputs = {'encoder_input': torch.tensor(batch[0]).to(config.device),
                  'encoder_mask': torch.tensor(batch[1]).to(config.device),
                  'encoder_with_oov': torch.tensor(batch[5]).to(config.device),
                  'context_vec': torch.tensor(batch[6]).to(config.device),
                  'decoder_input': torch.tensor(batch[2]).to(config.device),
                  'decoder_mask': torch.tensor(batch[3]).to(config.device),
                  'decoder_target': torch.tensor(batch[4]).to(config.device)}

        model.eval()
        prob = model.forward_eval(**inputs)
        all_probs += prob.cpu().detach().tolist()

    return all_probs
