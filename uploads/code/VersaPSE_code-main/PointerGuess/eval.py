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
    parser.add_argument('--test_file', type=str, default="../dataset/test/pseudo_test_data-targeted.txt")
    parser.add_argument('--load_ckpt', type=str, default="./local.pth")
    parser.add_argument('--src_pos', type=int, default=0, help='Position of the source text in each line.')
    parser.add_argument('--trg_pos', type=int, default=1, help='Position of the target text in each line.')
    parser.add_argument('--output_path', type=str, default="./test.txt", help='Path to save the output guesses.')
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
    Load and preprocess the test password pairs.
    """
    dataset=pointerguess_utils.preprocess(fp=args.test_file)
    dataloader=DataLoader(dataset, batch_size=512, shuffle=False, collate_fn=my_collate)


    """
    Build the PointerGuess model and load the trained checkpoint.
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

    model.train()
    model.to("cuda")
    model.load_state_dict(torch.load(args.load_ckpt))

    # Compute the log-probability / probability of every test password pair.
    all_probs = pointerguess_utils.prob_extract(model, gen_args, dataloader)

    with open(args.output_path, 'w') as f:
        for prob in all_probs:
            f.write(str(prob)+'\n')