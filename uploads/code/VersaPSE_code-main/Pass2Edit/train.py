import argparse
import os
import pickle
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Internal project modules
import hyper_param as param
import model as mt
import tokenizer
import utils


def train(model, dataloader, save_path, epoch):
    """Train the model for the given number of epochs and save the checkpoint.

    Cross-entropy loss is used with an Adam optimizer (lr=0.001). A checkpoint
    is saved to ``save_path`` after every epoch. Loss curves are plotted at the
    end of training (requires an interactive matplotlib backend).
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    total = 0
    total_batch = 0
    total_list_epoch = []
    loss_list_epoch = []
    total_list_batch = []
    loss_list_batch = []
    for i in range(epoch):
        model.train()
        epoch_total_loss = 0
        batchNum = 0
        start = time.time()
        for X1, X2, L, Y in dataloader:
            batchNum += 1
            total_batch += 1
            total_list_batch.append(total_batch)
            # Move data to the configured device
            X1 = torch.tensor(X1).to(param.device)
            X2 = torch.tensor(X2).to(param.device)
            L = torch.tensor(L).to(param.device)
            Y = torch.tensor(Y).squeeze(1).to(param.device)

            # Forward pass and back-propagation
            optimizer.zero_grad()
            outputs = model(X1, X2, L)
            loss = criterion(outputs, Y)
            loss.backward()
            optimizer.step()

            # Log progress every 10 batches
            curr = time.time()
            if batchNum % 10 == 1:
                print(f'Epoch {i + 1}/{epoch}, Batch:{batchNum}, Loss: {loss.item()}, total time:{curr - start}, avg time per batch:{(curr - start) / batchNum}')
            epoch_total_loss += float(loss.item())
            loss_list_batch.append(float(loss.item()))
            # Free memory
            del X1, X2, Y, outputs, loss
            torch.cuda.empty_cache()
        loss_list_epoch.append(epoch_total_loss / batchNum)
        total_list_epoch.append(total)
        total += 1
        torch.save({'Model': model.state_dict()}, os.path.join(save_path))

    # Plot loss curves (epoch-level and batch-level)
    plt.plot(total_list_epoch, loss_list_epoch)
    plt.show()
    plt.plot(total_list_batch, loss_list_batch)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--train_file', type=str, default="../dataset/train/pseudo_train_data-sister.txt")
    parser.add_argument('--save_path', type=str, default="./")
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--load_ckpt', type=str, default=None)
    parser.add_argument('--load_tokenizer', type=str, default=None)
    parser.add_argument('--save_name', type=str, default="local.pth", help='Name of the model file to save')
    parser.add_argument('--src_pos', type=int, default=0, help='Position of the source text in each line.')
    parser.add_argument('--trg_pos', type=int, default=1, help='Position of the target text in each line.')
    args = parser.parse_args()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Preprocess the password pairs into per-step source/target/edit data
    per_step_src, per_step_trg, per_step_edit, per_step_length, paths = utils.preprocess(
        args.train_file, trg_pos=args.trg_pos, src_pos=args.src_pos)

    # Load an existing edit tokenizer, or train a new one and save it
    if args.load_tokenizer is not None:
        with open(args.load_tokenizer, 'rb') as f:
            op_tokenizer = pickle.load(f)
    else:
        op_tokenizer = tokenizer.EditTokenizer()
        op_tokenizer.TrainOnSeqs(paths)
        with open(os.path.join(args.save_path, 'edit_tokenizer.pkl'), 'wb') as f:
            pickle.dump(op_tokenizer, f)
        print(len(op_tokenizer.edit2id))

    # Build the model; the output size depends on the number of edit operations
    model = mt.SimpleP2E(type_num=len(op_tokenizer.edit2id) + 5)
    model = model.to(param.device)
    if args.load_ckpt is not None:
        # Load a pre-trained checkpoint (optionally call model.freeze_layers() before training)
        model.load_state_dict(torch.load(args.load_ckpt)['Model'])

    # Tokenize the edit sequences and build the dataset/dataloader
    tokenized_edits = op_tokenizer.Seqs2Tokens(per_step_edit)
    dataset = utils.CustomDataset(per_step_src, per_step_trg, per_step_length, tokenized_edits)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True, collate_fn=utils.my_collate)
    train(model, dataloader, save_path=args.save_path + args.save_name, epoch=args.epochs)


