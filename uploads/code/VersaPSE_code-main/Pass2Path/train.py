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

device = torch.device("cuda")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--train_file', type=str, default="../dataset/train/pseudo_train_data-sister.txt")
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default="./")
    parser.add_argument('--save_name', type=str, default="local.pth", help='Model save name.')
    parser.add_argument('--src_pos', type=int, default=0, help='Position of the source text in each line.')
    parser.add_argument('--trg_pos', type=int, default=1, help='Position of the target text in each line.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
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
    if args.load_path:
        # Continue from a pretrained checkpoint; freeze decoder layers for transfer learning.
        model.load_state_dict(torch.load(args.load_path))
        model.freeze_layers()
    model.train()

    directory_name = args.save_path
    if not os.path.exists(directory_name):
        os.mkdir(directory_name)

    # Build the training dataset from a tab-separated password-pair file.
    dataset = dataset_process(args.train_file, mode="train", src_pos=args.src_pos, trg_pos=args.trg_pos)

    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0, collate_fn=my_collate, drop_last=False)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    model.to(device)
    for epoch in range(args.epochs):
        loop = tqdm(dataloader)
        for X, Y in loop:
            X = X.to(device)
            Y = Y.to(device)
            result = model(X, Y)
            batch_loss = loss(result.permute(0, 2, 1), Y)
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
            loop.set_description(f"loss={batch_loss.item()}")

        torch.save(model.state_dict(), f'{args.save_path}{args.save_name}')

