import string
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from tqdm import tqdm
import word2keypress as kb
import sequence_functions as sf

# Maximum password length (in characters)
class GetStringTokenizer:
    def __init__(self):
        chars = (list(string.ascii_letters) + list(string.punctuation) +
             list(string.digits) + [" ", "\t", "\x03", "\x04"])
        self.char2token={}
        self.token2char={}
        self.pad_idx=0
        for i in range(len(chars)):
            self.char2token[chars[i]]=i+1
            self.token2char[i+1]=chars[i]

    def tokenizeString(self, str):
        tokens=[]
        for char in str:
            tokens.append(self.char2token[char])
        while len(tokens)<30:
            tokens.append(0)
        return tokens

    def fromTokens(self, tokens):
        plaintext=''
        for token in tokens:
            if token==0:
                break
            plaintext+=self.token2char[token]
        return plaintext

class CustomDataset(Dataset):
    def __init__(self, train_X, train_Y):
        self.X = train_X
        self.Y = train_Y

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.X)

    def __getitem__(self, idx):
        # Return a single sample (source sequence, target path) by index
        return self.X[idx], self.Y[idx],


def my_collate(batch):
    X, Y = [], []
    for i in batch:
        X.append(torch.tensor(i[0]))
        Y.append(torch.tensor(i[1]))

    X=pad_sequence(X,batch_first=True)
    Y=pad_sequence(Y,batch_first=True,padding_value=2)
    return X,Y

def _process_one(args):
    # Worker function for the multiprocessing pool.
    # Each process must create its own Keyboard instance.
    src, trg, mode, tokenizer = args
    keyboard = kb.Keyboard()
    src_k = keyboard.word_to_keyseq(src)
    trg_k = keyboard.word_to_keyseq(trg)
    # Convert the password pair into an edit-operation path
    path = sf.pair2path(src_k, trg_k)
    # Prepend <BOS> (0) and append <EOS> (1)
    path = [0] + path + [1]
    if mode == "train":
        if len(path) > 61:
            return None
    # Pad the path to a fixed length with <PAD> (2)
    while len(path) < 61:
        path.append(2)
    src_tok = tokenizer.tokenizeString(src_k)
    temp_max = max(path)
    return (src_tok, path, temp_max)

def dataset_process(file_path, mode="train", src_pos=0, trg_pos=1):
    # Build a Dataset from a tab-separated password-pair file.
    # src_pos/trg_pos give the column indices of source and target passwords.
    import multiprocessing as mp
    train = open(file_path, encoding='ascii', errors='ignore')
    tokenizer = GetStringTokenizer()
    # 1. Read all (src, trg) pairs at once
    src_trg_list = []
    cnt = 0
    for line in train:
        cnt += 1
        ls = line.strip('\n').split("\t")
        src, trg = ls[src_pos], ls[trg_pos]
        src = src.rstrip('\n')
        trg = trg.rstrip("\n")
        if mode == "train":
            if len(src) > 30 or len(trg) > 30:
                continue
        else:
            if len(src) > 30 or len(trg) > 30:
                src = src[:29]
        src_trg_list.append((src, trg, mode, tokenizer))

    # 2. Process every (src, trg) pair in parallel
    with mp.Pool(processes=16) as pool:
        results = list(tqdm(pool.imap(_process_one, src_trg_list), total=len(src_trg_list)))

    train_X = []
    train_Y = []
    max_token = 0
    valid_cnt = 0
    for item in results:
        if item is None:
            continue
        src_tok, path, temp_max = item
        train_X.append(src_tok)
        train_Y.append(path)
        if temp_max > max_token:
            max_token = temp_max
        valid_cnt += 1

    print(max_token)

    dataset = CustomDataset(train_X, train_Y)
    if mode != "train":
        assert len(dataset) == valid_cnt

    return dataset

def dataset_process_single_thread(file_path, mode="train", src_pos=0, trg_pos=1):
    # Single-threaded fallback of dataset_process (kept for debugging / small files).
    train = open(file_path, encoding='ascii', errors='ignore')
    tokenizer = GetStringTokenizer()
    train_X = []
    train_Y = []
    # Special tokens: <BOS>: 0, <EOS>: 1, <PAD>: 2
    cnt = 0
    max_token = 0
    keyboard = kb.Keyboard()
    for line in tqdm(train):
        cnt += 1
        ls = line.strip('\n').split("\t")
        src, trg = ls[src_pos], ls[trg_pos]
        src = src.rstrip('\n')
        trg = trg.rstrip("\n")
        if mode == "train":
            if len(src) > 30 or len(trg) > 30:
                continue
        else:
            if len(src) > 30 or len(trg) > 30:
                src = src[:29]
        src = keyboard.word_to_keyseq(src)
        trg = keyboard.word_to_keyseq(trg)
        path = sf.pair2path(src, trg)
        path = [0] + path + [1]
        if mode == "train":
            if len(path) > 61:
                continue
        while len(path) < 61:
            path.append(2)
        src = tokenizer.tokenizeString(src)
        temp_max = max(path)
        if temp_max > max_token:
            max_token = temp_max
        train_X.append(src)
        train_Y.append(path)

    print(max_token)

    dataset = CustomDataset(train_X, train_Y)
    if mode != "train":
        assert len(dataset) == cnt

    return dataset

