import multiprocessing as mp
import pickle
import string

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from word2keypress import Keyboard

# Internal project modules
import hyper_param as param
import sequence_functions as sf

# Character set used for tokenizing passwords (printable ASCII minus the last 5 chars)
charset = string.printable[:-5]

# Rebuild the charset from keyboard key sequences so that every character can be
# typed on a keyboard (relevant for keypress-level edit modelling)
kb = Keyboard()
kb_charlist = kb.word_to_keyseq(charset)
kb_charset = []
for i in kb_charlist:
    if i not in kb_charset:
        kb_charset.append(i)
charset = ''.join(kb_charset)


class CustomDataset(Dataset):
    """Dataset of per-step password-pair samples.

    Each sample contains the source password, target password, true length and
    the ground-truth edit operation id.
    """

    def __init__(self, X1, X2, L, Y):
        self.X1 = X1
        self.X2 = X2
        self.L = L
        self.Y = Y

    def __len__(self):
        return len(self.X1)

    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.L[idx], self.Y[idx],


def my_collate(batch):
    """Collate a batch of samples into lists (padding is done inside the model)."""
    X1, X2, L, Y = [], [], [], []
    for i in batch:
        X1.append(i[0])
        X2.append(i[1])
        L.append(i[2])
        Y.append(i[3])
    return X1, X2, L, Y


def pw2token(batch_pw):
    """Tokenize password strings into id sequences.

    Token layout: 0 = BOS, 1 = EOS, 2 = padding, 3 = deleted-char placeholder
    ('\xde'), ids >= 4 map to characters in ``charset``. Sequences are padded
    or truncated to ``param.MAX_LEN + 2``.
    """
    tokenized_pw = []
    for pw in batch_pw:
        pw = pw.rstrip('\n')
        tokens = []
        for c in pw:
            if c != '\xde':
                tokens.append(4 + charset.find(c))
            else:
                tokens.append(3)
        tokens += [1]  # EOS
        tokens = [0] + tokens  # BOS
        while len(tokens) < param.MAX_LEN + 2:
            tokens += [2]  # padding
        if len(tokens) > param.MAX_LEN + 2:
            tokens = tokens[:param.MAX_LEN + 2]
        tokenized_pw.append(tokens)
    return tokenized_pw


def _process_one(pair):
    """Convert one (src, trg) password pair into per-step training samples.

    Returns per-step source/target tokens, per-step edit strings, per-step
    lengths and the cleaned edit path. Keyboard() is instantiated inside the
    worker because it cannot be pickled across processes.
    """
    kb = Keyboard()
    src_k = kb.word_to_keyseq(pair[0])
    trg_k = kb.word_to_keyseq(pair[1])
    path = sf.trans2seq(sf.transformation(src_k, trg_k))
    temp = sf.pass2edit(src_k, path)
    per_step_src = []
    per_step_trg = []
    per_step_edit = []
    per_step_length = []
    for it in temp:
        src_token = pw2token([it[0]])[0]
        trg_token = pw2token([it[1]])[0]
        if 1 in src_token:
            src_len = src_token.index(1) + 1
        else:
            src_len = len(src_token)
        if 1 in trg_token:
            trg_len = trg_token.index(1) + 1
        else:
            trg_len = len(trg_token)
        assert src_len == trg_len
        per_step_src.append(src_token)
        per_step_trg.append(trg_token)
        per_step_length.append(src_len)
        per_step_edit.append([sf.trans2seq([it[2]]).rstrip('\xa1')])
    clean_path = path.split('\xa1')[:-1]
    return per_step_src, per_step_trg, per_step_edit, per_step_length, clean_path


def _process_batch(batch):
    """Same as ``_process_one`` but for a list of pairs; used with a process pool."""
    kb = Keyboard()
    batch_src, batch_trg, batch_edit, batch_length, batch_clean_paths = [], [], [], [], []
    for pair in batch:
        src_k = kb.word_to_keyseq(pair[0])
        trg_k = kb.word_to_keyseq(pair[1])
        path = sf.trans2seq(sf.transformation(src_k, trg_k))
        temp = sf.pass2edit(src_k, path)
        per_step_src = []
        per_step_trg = []
        per_step_edit = []
        per_step_length = []
        for it in temp:
            src_token = pw2token([it[0]])[0]
            trg_token = pw2token([it[1]])[0]
            if 1 in src_token:
                src_len = src_token.index(1) + 1
            else:
                src_len = len(src_token)
            if 1 in trg_token:
                trg_len = trg_token.index(1) + 1
            else:
                trg_len = len(trg_token)
            assert src_len == trg_len
            per_step_src.append(src_token)
            per_step_trg.append(trg_token)
            per_step_length.append(src_len)
            per_step_edit.append([sf.trans2seq([it[2]]).rstrip('\xa1')])
        clean_path = path.split('\xa1')[:-1]
        batch_src.extend(per_step_src)
        batch_trg.extend(per_step_trg)
        batch_edit.extend(per_step_edit)
        batch_length.extend(per_step_length)
        batch_clean_paths.append(clean_path)
    return batch_src, batch_trg, batch_edit, batch_length, batch_clean_paths


def preprocess(fp, src_pos=0, trg_pos=1):
    """Load a password-pair file and convert it into per-step training samples.

    The file should contain tab-separated pairs; ``src_pos``/``trg_pos`` select
    the source and target column. Processing is parallelized over 16 workers.
    """
    pairs = []
    count = 0
    with open(fp, 'r', encoding='utf-8') as f:
        for i in f:
            count += 1
            if count % 10000 == 0:
                print(count)
            line = i.rstrip('\n').split('\t')
            pairs.append((line[src_pos], line[trg_pos]))

    batch_size = 1000
    batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]

    with mp.Pool(processes=16) as pool:
        results = list(tqdm(pool.imap(_process_batch, batches), total=len(batches)))

    per_step_src, per_step_trg, per_step_edit, per_step_length, clean_paths = [], [], [], [], []
    for src, trg, edit, length, clean_path in results:
        per_step_src.extend(src)
        per_step_trg.extend(trg)
        per_step_edit.extend(edit)
        per_step_length.extend(length)
        clean_paths.extend(clean_path)
    return per_step_src, per_step_trg, per_step_edit, per_step_length, clean_paths


def preprocess_single_thread(fp, src_pos=0, trg_pos=1):
    """Single-threaded variant of ``preprocess``, kept for debugging."""
    pairs = []
    count = 0
    kb = Keyboard()
    with open(fp, 'r', encoding='utf-8') as f:
        for i in f:
            count += 1
            if count % 10000 == 0:
                print(count)
            line = i.rstrip('\n').split('\t')
            src, trg = kb.word_to_keyseq(line[src_pos]), kb.word_to_keyseq(line[trg_pos])
            pairs.append((src, trg))
    paths = []
    for pair in pairs:
        path = sf.trans2seq(sf.transformation(pair[0], pair[1]))
        paths.append(path)
    per_step_src = []
    per_step_trg = []
    per_step_edit = []
    per_step_length = []
    for pair, path in tqdm(zip(pairs, paths)):
        temp = sf.pass2edit(pair[0], path)
        for it in temp:
            src_token = pw2token([it[0]])[0]
            trg_token = pw2token([it[1]])[0]
            per_step_src.append(src_token)
            per_step_trg.append(trg_token)
            if 1 in src_token:
                src_len = src_token.index(1) + 1
            else:
                src_len = len(src_token)
            if 1 in trg_token:
                trg_len = trg_token.index(1) + 1
            else:
                trg_len = len(trg_token)
            assert src_len == trg_len
            per_step_length.append(src_len)
            per_step_edit.append([sf.trans2seq([it[2]]).rstrip('\xa1')])
    clean_paths = []
    for path in paths:
        clean_paths.append(path.split('\xa1')[:-1])
    return per_step_src, per_step_trg, per_step_edit, per_step_length, clean_paths

