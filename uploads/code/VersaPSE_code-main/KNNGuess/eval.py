import torch
import torch.nn as nn
from torch.autograd import Variable
from model import make_model
import logging
from tqdm import tqdm
from PW2SEQ import PW2SEQ
import config
import numpy as np
pw2seq=PW2SEQ()
from knn.datastore.datastore import Datastore
from knn.retriever.retriever import Retriever
from knn.combiner.combiner import Combiner
from lz_delete import beam_decode2
pw2seq_dict=pw2seq.pw2seq_dict
import utils
import config
import logging
import numpy as np
from data_loader import MTDataset
from torch.utils.data import DataLoader
import torch
from PW2SEQ import PW2SEQ
from train import train
from model import make_model, LabelSmoothing
import argparse
from data_loader import subsequent_mask

def eval(data, model, retriever=None, combiner=None):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(data):
            src = batch.src
            src_mask = batch.src_mask
            decoder_input = batch.trg   # already shifted by Batch: trg[:, :-1]
            decoder_target = batch.trg_y  # already shifted by Batch: trg[:, 1:]
            tgt_mask = batch.trg_mask
            memory = model.encode(src, src_mask)
            out = model.decode(memory, src_mask, decoder_input, tgt_mask)  # [batch, seq_len-1, d_model]
            logits = model.generator(out)  # [batch, seq_len-1, vocab], log-prob
            model_prob = torch.exp(logits)

            if retriever is not None and combiner is not None:
                bsz, tlen, dim = out.shape
                flat_out = out.contiguous().view(-1, dim)
                results = retriever.retrieve(flat_out, return_list=["vals", "distances"])
                knn_prob = combiner.get_knn_prob(**results, device=src.device).squeeze(1)
                knn_prob = knn_prob.view(bsz, tlen, -1)
                probs = model_prob * config.lambda_ + knn_prob * (1 - config.lambda_)
            else:
                probs = model_prob

            # gather出每个step的目标token概率
            target_probs = torch.gather(probs, 2, decoder_target.unsqueeze(2)).squeeze(2)  # [batch, seq_len-1]

            # 构造mask: 保留EOS之前与第一个EOS，忽略EOS之后，同时过滤PAD
            eos_idx = config.eos_idx
            pad_idx = config.padding_idx
            is_eos = (decoder_target == eos_idx)  # [batch, seq_len-1]
            eos_cum = is_eos.long().cumsum(dim=1)
            first_eos = is_eos & (eos_cum == 1)
            prefix_mask = (eos_cum == 0) | first_eos
            non_pad_mask = (decoder_target != pad_idx)
            mask = (prefix_mask & non_pad_mask).float()

            # 用mask乘以log(prob)，最后exp(sum)得到累乘概率
            log_probs = torch.log(target_probs + 1e-12) * mask  # 防止log(0)
            seq_log_prob = log_probs.sum(dim=1)
            seq_probs = torch.exp(seq_log_prob)
            all_probs.extend(seq_probs.tolist())
    return all_probs

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--test_file', type=str, default="../dataset/test/pseudo_test_data-targeted.txt")
    parser.add_argument('--load_ckpt', type=str, default="./experiment/local_to_global.pth")
    parser.add_argument('--datastore_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default="./result/knnguess_res.txt", help='Path to save the output guesses.')
    parser.add_argument('--src_pos', type=int, default=0, help='Position of source text in the input file.')
    parser.add_argument('--trg_pos', type=int, default=1, help='Position of target text in the input file.')
    args = parser.parse_args()
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    model.load_state_dict(torch.load(args.load_ckpt))
    model.to(config.device)
    model.eval()

    retriever = None
    combiner = None
    if args.datastore_path is not None:
        datastore = Datastore.load(args.datastore_path, load_list=["vals"])
        datastore.load_faiss_index("keys")
        retriever = Retriever(datastore=datastore, k=config.k)
        combiner = Combiner(
            lambda_=config.lambda_,
            temperature=config.knn_temperature,
            probability_dim=config.src_vocab_size,
        )

    train_dataset = MTDataset(args.test_file, mode='test', sort=False, src_pos=args.src_pos, trg_pos=args.trg_pos)
    logging.info("-------- Dataset Build! --------")
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=config.train_batch_size,
                                collate_fn=train_dataset.collate_fn)
    result=eval(train_dataloader, model, retriever=retriever, combiner=combiner)
    print(result[:10])
    print(len(result))
    with open(args.output_path, "w", encoding="utf-8") as f:
        for prob in result:
            f.write(f"{prob}\n")
