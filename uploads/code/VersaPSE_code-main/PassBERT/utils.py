import string
from torch.utils.data import Dataset
from tqdm import tqdm
import config


# All printable ASCII characters (95 chars) that make up the token vocabulary.
charset = string.printable[:-5]

def preprocess(fp):
    """Read a file of passwords (one per line) and convert each password
    into a padded token sequence of length MAX_LEN + 2.

    Args:
        fp: training data file path. One password per line.
    """
    train_f = open(fp)
    count = 0
    print(charset)
    train_X_seq = []

    for pw in tqdm(train_f):
        count += 1
        pw = pw.rstrip('\n')

        tokens = []
        for c in pw:
            # Token ids start from 3; ids 0/1/2 are reserved for CLS/SEP/PAD.
            tokens.append(3 + charset.find(c))

        # Append SEP token, then prepend CLS token.
        tokens += [1]
        tokens = [0] + tokens
        # Pad with PAD tokens (id=2) to a fixed length.
        while len(tokens) < config.MAX_LEN + 2:
            tokens += [2]

        if len(tokens) > config.MAX_LEN + 2:
            # Skip passwords longer than the maximum sequence length.
            continue
        train_X_seq.append(tokens)
        if count > 10000:
            break

    return train_X_seq


def pw2token(batch_pw):
    """Tokenize a list of passwords into padded token sequences (with CLS/SEP)."""
    tokenized_pw = []

    for pw in batch_pw:
        pw = pw.rstrip('\n')
        tokens = []
        for c in pw:
            tokens.append(3 + charset.find(c))

        tokens += [1]
        tokens = [0] + tokens
        while len(tokens) < config.MAX_LEN + 2:
            tokens += [2]

        if len(tokens) > config.MAX_LEN + 2:
            # Truncate passwords longer than the maximum sequence length.
            tokens = tokens[:config.MAX_LEN + 2]

        tokenized_pw.append(tokens)

    return tokenized_pw

def pw2token_src(batch_pw):
    """Tokenize source passwords with a special suffix (SEP x3 + padding).

    Unlike pw2token, this appends three SEP-like tokens (id=0) plus a final
    SEP (id=1) at the end, used for the edit-path prediction task.
    """
    tokenized_pw = []

    for pw in batch_pw:
        pw = pw.rstrip('\n')
        tokens = []
        for c in pw:
            tokens.append(3 + charset.find(c))

        tokens = [0] + tokens + [0, 0, 0, 1]
        while len(tokens) < config.MAX_LEN + 2:
            tokens += [2]

        if len(tokens) > config.MAX_LEN + 2:
            tokens = tokens[:config.MAX_LEN + 2]

        tokenized_pw.append(tokens)

    return tokenized_pw


class CustomDataset(Dataset):
    """Dataset wrapping a list of tokenized password sequences."""
    def __init__(self, train_X_seq):
        self.X_seq = train_X_seq

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        return self.X_seq[idx]


def my_collate(batch):
    """Collate a batch of token sequences into a list."""
    X_seq = []
    for it_seq in batch:
        X_seq.append(it_seq)

    return X_seq

def read_label(fp):
    """Read (src, trg, label) tuples from a tab-separated file.

    Each line has the form: src_password \t target_password \t label_list
    where label_list is the encoded in-place edit path produced by editpath.py.
    """
    label_f = open(fp)
    srcs = []
    trgs = []
    labels = []
    for line in label_f:
        src, trg, label = line.rstrip('\n').split("\t")
        srcs.append(src)
        trgs.append(trg)
        labels.append(eval(label))
    src_tokens = pw2token_src(srcs)
    trg_tokens = pw2token_src(trgs)
    padded_labels = []
    for label in labels:
        # Align the label sequence with the token sequence: prepend CLS and
        # append SEP, then pad with PAD (id=2).
        padded_label = [0] + label + [0]
        while len(padded_label) < config.MAX_LEN + 2:
            padded_label += [2]
        if len(padded_label) > config.MAX_LEN + 2:
            padded_label = padded_label[:config.MAX_LEN + 2]
        padded_labels.append(padded_label)
    return src_tokens, trg_tokens, padded_labels

class LabelDataset(Dataset):
    """Dataset holding (src_tokens, trg_tokens, labels) tuples."""
    def __init__(self, src_tokens, trg_tokens, labels):
        self.src_tokens = src_tokens
        self.trg_tokens = trg_tokens
        self.labels = labels

    def __len__(self):
        return len(self.src_tokens)

    def __getitem__(self, idx):
        return self.src_tokens[idx], self.trg_tokens[idx], self.labels[idx]

def label_collate(batch):
    """Collate a batch of (src, trg, label) tuples into three lists."""
    src_tokens = []
    trg_tokens = []
    labels = []
    for it in batch:
        src_tokens.append(it[0])
        trg_tokens.append(it[1])
        labels.append(it[2])
    return src_tokens, trg_tokens, labels
