import argparse
import os
import pickle
import time

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# Internal project modules
import hyper_param as param
import model as mt
import tokenizer
import utils


def prob_extract(model, dataloader):
    """Compute the probability of the ground-truth edit at each atomic step.

    Returns a flat list of per-step probabilities (softmax of the model output
    at the true edit id).
    """
    model.eval()
    start = time.time()
    all_probs = []
    for X1, X2, L, Y in tqdm(dataloader):
        # Move data to the configured device
        X1 = torch.tensor(X1).to(param.device)
        X2 = torch.tensor(X2).to(param.device)
        L = torch.tensor(L).to(param.device)
        Y = torch.tensor(Y).squeeze(1).to(param.device)

        outputs = model(X1, X2, L)
        probs = torch.softmax(outputs, dim=1)
        # Gather the probability of the true edit operation for each sample
        edit_probs = torch.gather(probs, 1, Y.unsqueeze(1)).squeeze(1)
        all_probs += edit_probs.tolist()
    return all_probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--test_file', type=str, default="../dataset/test/pseudo_test_data-targeted.txt")
    parser.add_argument('--load_ckpt', type=str, default="./local.pth")
    parser.add_argument('--load_tokenizer', type=str, default="./edit_tokenizer.pkl", help='Name of the model file to save')
    parser.add_argument('--src_pos', type=int, default=0, help='Position of the source text in each line.')
    parser.add_argument('--trg_pos', type=int, default=1, help='Position of the target text in each line.')
    parser.add_argument('--output_path', type=str, default="./test.txt", help='Path to save the output guesses.')
    args = parser.parse_args()
    with open(args.load_tokenizer, 'rb') as f:
        op_tokenizer = pickle.load(f)
    model = mt.SimpleP2E(type_num=len(op_tokenizer.edit2id) + 5)
    model = model.to(param.device)
    if args.load_ckpt is not None:
        model.load_state_dict(torch.load(args.load_ckpt)['Model'])

    # Preprocess the test pairs and tokenize the edit sequences
    per_step_src, per_step_trg, per_step_edit, per_step_length, paths = utils.preprocess(
        args.test_file, trg_pos=args.trg_pos, src_pos=args.src_pos)
    tokenized_edits = op_tokenizer.Seqs2Tokens(per_step_edit)
    dataset = utils.CustomDataset(per_step_src, per_step_trg, per_step_length, tokenized_edits)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False, collate_fn=utils.my_collate)

    # Collect per-step probabilities, then multiply them along each password pair
    result = prob_extract(model, dataloader)
    probs = []
    idx = 0
    for it in paths:
        prob = 1
        for i in range(len(it)):
            prob *= result[idx]
            idx += 1
        probs.append(prob)

    # Write one overall probability per password pair
    with open(args.output_path, 'w') as f:
        for prob in probs:
            f.write(str(prob) + '\n')
                