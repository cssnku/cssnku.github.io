from model import PassBERT
import torch.nn.functional as F
import argparse
from utils import read_label, LabelDataset, label_collate
import torch
from tqdm import tqdm

def prob_cal(model, dataloader):
    """Compute the cumulative probability of each ground-truth edit path
    in the dataloader.

    Returns:
        all_probs: list of per-sample cumulative probabilities.
    """
    model.eval()
    all_probs = []
    loop = tqdm(dataloader, desc="Calculating Probabilities")
    for X_seq, _, labels in loop:
        labels = torch.tensor(labels).to('cuda')
        X_seq = torch.tensor(X_seq).to('cuda')
        probs = model.prob_extract(X_seq, labels)
        all_probs += probs.cpu().tolist()
    return all_probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model on test data')
    parser.add_argument('--test_file', type=str, default="./dataset/inplace_edits_test_targeted.txt")
    parser.add_argument('--load_ckpt', type=str, default="./bert_local_to_global_last.pth")
    # parser.add_argument('--src_pos', type=int, default=0, help='Position of the source text in each line.')
    # parser.add_argument('--trg_pos', type=int, default=1, help='Position of the target text in each line.')
    parser.add_argument('--output_path', type=str, default="./test.txt",
                        help='Path to save the output probabilities.')
    args = parser.parse_args()

    model = PassBERT()
    model.load_state_dict(torch.load(args.load_ckpt)['Model'])
    model.to('cuda')
    src_tokens, trg_tokens, padded_labels = read_label(args.test_file)
    dataset = LabelDataset(src_tokens, trg_tokens, padded_labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, collate_fn=label_collate)
    res = prob_cal(model, dataloader)
    with open(args.output_path, 'w') as f:
        for prob in res:
            f.write(str(prob) + '\n')

