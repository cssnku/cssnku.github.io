import argparse
import os

from torch import nn

from model import ResidualLSTMDecoder, ResidualLSTMEncoder, Seq2Seq
import string
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn import CrossEntropyLoss
import torch
from tqdm import tqdm

import sequence_functions as sf
from utils import GetStringTokenizer, CustomDataset, dataset_process, my_collate

device=torch.device("cuda")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a trained model')
    parser.add_argument('--test_file', type=str, default="../dataset/test/pseudo_test_data-targeted.txt")
    parser.add_argument('--load_ckpt', type=str, default="./local.pth")
    parser.add_argument('--src_pos', type=int, default=0, help='Position of the source text in each line.')
    parser.add_argument('--trg_pos', type=int, default=1, help='Position of the target text in each line.')
    parser.add_argument('--output_path', type=str, default="./test.txt", help='Path to save the output probabilities.')
    args = parser.parse_args()

    # Encoder: takes the key-sequence of the source password (98 chars + 3 special tokens).
    encoder = ResidualLSTMEncoder(vocab_size=98 + 3,
                                  num_layers=3,
                                  residual_layers=[0, 1, 2],
                                  embedding_dim=200,
                                  inp_hid_size=128,
                                  device=device,
                                  dropout=0.4)

    # Decoder: generates the transition path (vocab defined by trans_dict, 12000 classes).
    decoder = ResidualLSTMDecoder(vocab_size=12000,
                                  num_layers=3,
                                  residual_layers=[0, 1, 2],
                                  num_attn_layers=0,
                                  dropout=0.4,
                                  inp_size=128,
                                  device=device,
                                  hid_size=128)
    model = Seq2Seq(encoder=encoder, decoder=decoder)
    model.load_state_dict(torch.load(args.load_ckpt))
    model.eval()

    # Load the test dataset and compute the probability of each ground-truth path.
    dataset = dataset_process(args.test_file, mode="test", src_pos=args.src_pos, trg_pos=args.trg_pos)

    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0, collate_fn=my_collate, drop_last=False)

    model.to(device)
    loop = tqdm(dataloader)
    all_probs = []
    for X, Y in loop:
        X = X.to(device)
        Y = Y.to(device)
        result = model.prob_extract(X, Y)
        all_probs += result
    print(all_probs[:10])
    print(len(all_probs))
    with open(args.output_path, 'w') as f:
        for prob in all_probs:
            f.write(str(prob) + '\n')

