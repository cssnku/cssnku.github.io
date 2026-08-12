import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
import pointerguess_utils
import argparse
from reuse_utils import CustomDataset, my_collate

from model import PointerGeneratorNetworks

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('--reuse_tp', type=str, default="../dataset/train/pseudo_train_data-sister.txt",
                        help='Train data path for PointerReuse. Two password separated using tab per line.')
    parser.add_argument('--reuse_sp', type=str, default="./",
                        help='Model save path for PointerReuse.')
    parser.add_argument('--reuse_sp_name', type=str, default="local.pth",
                        help='Model save name for PointerReuse.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs for PointerReuse.')
    parser.add_argument('--load_ckpt_path', type=str, default=None, help='Path to the pre-trained checkpoint.')
    parser.add_argument('--src_pos', type=int, default=0, help='Position of the source text in each line.')
    parser.add_argument('--trg_pos', type=int, default=1, help='Position of the target text in each line.')
    args = parser.parse_args()


    def set_seed(seed=42):
        """Set the random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(42)

    """
    Load and preprocess the training password pairs.
    """
    dataset=pointerguess_utils.preprocess(fp=args.reuse_tp, src_pos=args.src_pos, trg_pos=args.trg_pos)
    dataloader=DataLoader(dataset, batch_size=512, shuffle=True, collate_fn=my_collate)
    if not os.path.exists(args.reuse_sp):
        os.makedirs(args.reuse_sp)

    """
    Build the PointerGuess model.
    """
    class gen_args:
        vocab_num = 98
        embedding_dim = 64
        hidden_dim = 128
        dropout = 0.5
        pointer_gen = True
        use_coverage = False
        min_decode_len = 5
        max_decode_len = 31
        article_max_len = 64
        lr = 1e-3


    pad_idx = 2

    unk_idx = 1

    start_idx = 0

    stop_idx = 1

    model = PointerGeneratorNetworks(
        vob_size=gen_args.vocab_num,
        embed_dim=gen_args.embedding_dim,
        hidden_dim=gen_args.hidden_dim,
        pad_idx=pad_idx,
        dropout=gen_args.dropout,
        pointer_gen=gen_args.pointer_gen,
        use_coverage=gen_args.use_coverage,
        min_decoder_len=gen_args.min_decode_len,
        max_decoder_len=gen_args.max_decode_len,
        unk_token_idx=unk_idx,
        start_token_idx=start_idx,
        stop_token_idx=stop_idx,
    )

    # Optional: initialize from a pre-trained checkpoint and freeze the
    # decoder / reduce layers (continual-learning style reuse).
    if args.load_ckpt_path is not None:
        model.load_state_dict(torch.load(args.load_ckpt_path))
        model.freeze_layers()

    model.train()
    model.to("cuda")

    pointerguess_utils.train(model, gen_args, dataloader, sp=args.reuse_sp+args.reuse_sp_name, epochs=args.epochs)

