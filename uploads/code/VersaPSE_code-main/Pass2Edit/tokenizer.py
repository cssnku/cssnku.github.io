import pickle

import numpy as np
from tqdm import tqdm


class TwoGramTokenizer:
    """Converts a password into a bag-of-2-grams vector."""

    def __init__(self):
        self.gram2id = {}
        self.id2gram = {}

    def TrainOnPwds(self, pwds):
        """Build the 2-gram vocabulary from a list of passwords."""
        for pw in pwds:
            pw = "\x01" + pw + "\x02"
            for i in range(0, len(pw) - 1):
                if pw[i:i + 2] not in self.gram2id:
                    self.gram2id[pw[i:i + 2]] = len(self.gram2id) + 1
        print(self.gram2id)
        for i in self.gram2id:
            self.id2gram[self.gram2id[i]] = i
        print(self.id2gram)

    def Pwd2Tokens(self, pwd):
        """Return a count vector of the 2-grams contained in one password."""
        tokens = []
        pw = "\x01" + pwd + "\x02"
        for i in range(0, len(pw) - 1):
            tokens.append(self.gram2id[pw[i:i + 2]])
        res = np.zeros(len(self.gram2id) + 1)
        for i in tokens:
            res[i] += 1
        tokens = res.tolist()
        return tokens

    def Pwds2Tokens(self, pwds):
        """Return count vectors for a list of passwords."""
        result = []
        for pwd in pwds:
            result.append(self.Pwd2Tokens(pwd))
        return result


class EditTokenizer:
    """Maps atomic edit operations (e.g. "<1,2,a>") to integer ids."""

    def __init__(self):
        self.edit2id = {}
        # Id 1 is reserved for the EOS operation "<2,0>"
        self.edit2id["<2，0>"] = 1
        self.id2edit = {}

    def TrainOnSeqs(self, seqs):
        """Build the edit-operation vocabulary from a list of edit sequences.

        Ids start from 4: 0 is reserved for unseen operations and 1 for EOS.
        """
        id = 4
        for seq in seqs:
            for edit in seq:
                if edit not in self.edit2id:
                    self.edit2id[edit] = id
                    id += 1

        for i in self.edit2id:
            self.id2edit[self.edit2id[i]] = i

    def Seqs2Tokens(self, seqs, verbose=False):
        """Convert edit sequences (lists of edit strings) to lists of token ids.

        Unseen operations are mapped to id 0.
        """
        result = []
        iterator = tqdm(seqs) if verbose else seqs
        for seq in iterator:
            temp_tokens = []
            for edit in seq:
                if edit not in self.edit2id:
                    temp_tokens.append(0)
                else:
                    temp_tokens.append(self.edit2id[edit])
            result.append(temp_tokens)
        return result

    def Tokens2Seqs(self, tokensArray):
        """Convert token id lists back to edit-operation strings.

        Id 0 and unknown ids fall back to the EOS operation.
        """
        result = []
        for tokens in tokensArray:
            temp_seq = []
            for token in tokens:
                if token == 0 or token not in self.id2edit:
                    temp_seq.append(self.id2edit[1])
                else:
                    temp_seq.append(self.id2edit[token])
            result.append(temp_seq)
        return result