"""
author: Yifei Zhang (FeliceRivarez aka. Znamya)
description:
    Utility function for PointerGuess-based reuse mechanism.
"""

import os
import json
import string
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, TensorDataset
import torch
from tqdm import tqdm
import config

SPECIAL_TOKEN = ['<pad>','<unk>','<start>','<stop>']

word_to_idx = {}
word_to_idx['<start>'] = 0
word_to_idx['<unk>'] = 1
word_to_idx["<stop>"] = 1
word_to_idx['<pad>'] = 2
idx_to_word = {}
idx_to_word[0]='<start>'
idx_to_word[1]="<stop>"
idx_to_word[2]='<pad>'


# for idx, token in enumerate(SPECIAL_TOKEN):
#     word_to_idx[token] = idx
#     idx_to_word.append(token)


for i in range(32, 128):
    word_to_idx[chr(i)] = i - 32 + 3
    idx_to_word[i - 32 + 3]=chr(i)





# idx_to_word.append(token)

pad_idx = 2
pad_token = SPECIAL_TOKEN[pad_idx]

unk_idx = 1
unk_token = SPECIAL_TOKEN[unk_idx]

start_idx = 0
start_token = SPECIAL_TOKEN[start_idx]

stop_idx = 1
stop_token = SPECIAL_TOKEN[stop_idx]

class CustomDataset(Dataset):
    def __init__(self, encoder_input, encoder_mask, decoder_input, decoder_mask, decoder_target, encoder_input_with_oov, context_vec):
        self.encoder_input = encoder_input
        self.encoder_mask = encoder_mask
        self.decoder_input = decoder_input
        self.decoder_mask = decoder_mask
        self.decoder_target = decoder_target
        self.encoder_input_with_oov = encoder_input_with_oov
        self.context_vec = context_vec

    def __len__(self):
        # 返回数据集的大小
        return len(self.encoder_input)

    def __getitem__(self, idx):
        # 根据索引获取一个样本及其标签
        return (self.encoder_input[idx], self.encoder_mask[idx], self.decoder_input[idx], self.decoder_mask[idx], self.decoder_target[idx], self.encoder_input_with_oov[idx], self.context_vec[idx])

def my_collate(batch):
    encoder_input, encoder_mask, decoder_input, decoder_mask, decoder_target, encoder_input_with_oov, context_vec=[],[],[],[],[],[],[]
    for i in batch:
        encoder_input.append(i[0])
        encoder_mask.append(i[1])
        decoder_input.append(i[2])
        decoder_mask.append(i[3])
        decoder_target.append(i[4])
        encoder_input_with_oov.append(i[5])
        context_vec.append(i[6])

    return encoder_input, encoder_mask, decoder_input, decoder_mask, decoder_target, encoder_input_with_oov, context_vec

def preprocess_dataset(base_pw_lst, pw_lst=None):
    # inputs = {'encoder_input': batch[0],
    #           'encoder_mask': batch[1],
    #           'encoder_with_oov': batch[2],
    #           'oovs_zero': batch[3],  # None type
    #           'context_vec': batch[4],  # 不需要unsqueeze
    #           'coverage': batch[5],  # None type
    #           'decoder_input': batch[6],
    #           'decoder_mask': batch[7],
    #           'decoder_target': batch[8]}

    """
    对于encoder来说，只需要两个输入：encoder_input以及encoder_mask：
    torch.onnx.export(
            encoder,  # PyTorch 模型
            input_names=["encoder_input", "encoder_mask"],
            args=(inputs['encoder_input'], inputs["encoder_mask"]),  # 输入数据
            f="encoder.onnx",
            opset_version=12,
        )
    在处理完encoder之后，encoder的输出值包括：
    对于decoder来说，
    """

    if pw_lst == None:
        pw_lst = []
        while len(pw_lst) < len(base_pw_lst):
            pw_lst.append("114514")

    # min_decode_len = 5
    # max_decode_len = 31
    article_max_len = config.MAX_LEN
    abstract_max_len = config.MAX_LEN

    """
    Znamya's remarks:
    decoder_input和decoder_target属于比较特殊的。代码如下：

    target = [vocab.start_token] + target + [vocab.stop_token]

    target = target[: target_max_len + 1]

    target_indexes = [vocab.word_2_idx(word) for word in target]

    decoder_input = target_indexes[: -1]
    decoder_target = target_indexes[1:]

    assert len(decoder_input) == len(decoder_target)

    也就是说，保证decoder_input只有start symbol；decoder_target只有end symbol
    但是从形式上来讲，如果要生成，还是要以start symbol作为输入，然后end symbol作为结束的判定，
    因为这两个的长度相等，训练过程中每个input对应一个target。
    """

    encoder_input = []  # 长度固定为64，原始的没有start/end symbol，直接pad。
    encoder_mask = []  # 其实这里的mask就是一个长度为64的全1张量...
    encoder_input_with_oov = []  # 事实上我们不考虑oov，所以这里的encoder_with_oov和encoder_input完全等价
    context_vec = []  # 初始的时候就是[1,256]的全0张量：init_context_vec = torch.zeros((batch_size, 2 * hidden_dim),dtype=torch.float32)
    decoder_input = []  # 长度固定为31，我们在这加上start（没有end!），然后pad到64
    decoder_mask = []  # 这里按照pad以前的decoder_input确定，也就是对于最后一个字符后面的全部是0...
    decoder_target = []  # 这里没有start symbol，但是有end。同样地，用pad补充到31即可


    for it in tqdm(base_pw_lst):
        temp = []
        for ch in it:
            temp.append(word_to_idx[ch])
        while len(temp) < article_max_len:
            temp.append(pad_idx)
        if len(temp)>article_max_len:
            temp=temp[:article_max_len]
        encoder_input.append(temp)
        encoder_mask.append([1 for i in range(config.MAX_LEN)])
        encoder_input_with_oov.append(temp)

    for it in tqdm(pw_lst):
        temp = [0]
        for ch in it:
            temp.append(word_to_idx[ch])
        temp.append(1)
        temp_input = temp[:-1]
        temp_target = temp[1:]
        assert len(temp_input) == len(temp_target)
        temp_mask = [1 for i in range(len(temp_input))]
        while len(temp_input) < abstract_max_len:
            temp_input.append(pad_idx)
            temp_target.append(pad_idx)
            temp_mask.append(0)
        if len(temp_input)>abstract_max_len:
            temp_input=temp_input[:abstract_max_len]
            temp_mask=temp_mask[:abstract_max_len]
            temp_target = temp_target[:abstract_max_len]
        assert len(temp_input) == len(temp_mask)
        assert len(temp_input) == len(temp_target)
        decoder_mask.append(temp_mask)
        decoder_input.append(temp_input)
        decoder_target.append(temp_target)
        context_vec.append([0 for i in range(256)])

    # inputs = {'encoder_input': batch[0],
    #           'encoder_mask': batch[1],
    #           'encoder_with_oov': batch[2],
    #           'oovs_zero': batch[3],  # None type
    #           'context_vec': batch[4],  # 不需要unsqueeze
    #           'coverage': batch[5],  # None type
    #           'decoder_input': batch[6],
    #           'decoder_mask': batch[7],
    #           'decoder_target': batch[8]}

    # all_encoder_input = torch.tensor(encoder_input, dtype=torch.long)
    # all_encoder_mask = torch.tensor(encoder_mask, dtype=torch.long)

    # all_decoder_input = torch.tensor(decoder_input, dtype=torch.long)
    # all_decoder_mask = torch.tensor(decoder_mask, dtype=torch.int)

    # all_decoder_target = torch.tensor(decoder_target, dtype=torch.long)

    # all_encoder_input_with_oov = torch.tensor(encoder_input_with_oov, dtype=torch.long)

    # all_context = torch.tensor(context_vec, dtype=torch.float32)

    # dataset = TensorDataset(all_encoder_input, all_encoder_mask, all_decoder_input, all_decoder_mask,
    #                         all_decoder_target, all_encoder_input_with_oov, all_context)
    dataset=CustomDataset(encoder_input, encoder_mask, decoder_input, decoder_mask, decoder_target, encoder_input_with_oov, context_vec)
    return dataset
    

def idx_to_token(idx):
    return idx_to_word[idx]
