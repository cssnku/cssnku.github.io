"""
ref: https://github.com/Bijeeta/credtweak/blob/master/credTweakAttack/edit_distance_backtrace.py

Author: Tal Daniel
Minimum Edit Distance with Backtrace
-----------------------------------

We wish to find the Minimum Edit Distance (MED) between two strings. That is,
given two strings, align them, and find the minimum operations from {Insert,
Delete, Substitute} needed to get from the first string to the second string.
Then, we want to find the actual operations done in order to reach this MED,
e.g "Insert 'A' at position 3".

We can try and achieve this goal using Dynamic Programming (DP) for optimal
complexity as follows: Define:
* String 1: $X$ of length $n$
* String 2: $Y$ of length $m$
* $D[i,j]$: Edit Distance between substrings $X[1 \rightarrow i]$ and $Y[1 \rightarrow j]$

Using "Bottom Up" approach, the MED between $X$ and $Y$ would be $D[n,m]$.

We assume that the distance between string of length 0 to a string of length k
is k, since we need to insert k characters is order to create string 2.  In
order to actually find the operation, we need to keep track of the operations,
that is, create a "Backtrace".


Complexity:

* Time: O(n*m)
* Space: O(n*m)
* Backtrace: O(n+m)

"""
import random

import numpy as np
import string
import json
import csv
import itertools
import time
from word2keypress import Keyboard
from ast import literal_eval
from functools import lru_cache
from pathlib import Path


# Generate trans_dict_2idx.json / trans_dict_2path.json.
# Indices start from 3 to reserve slots for <PAD>, <BOS> and <EOS>.
def generate_transition_dict():
    '''Generate a dictionary of all possible paths in a JSON format
    Assumptions: words' max length is 30 chars and words are comprised of 98
    available characters
    'd' - ('d', None, 0-30) -> 31 options
    's' - ('s', 0-95, 0-30) -> 98x31 = 3038 options
    'i' - ('i', 0-95, 0-30) -> 98x31 = 3038 options
    Size of table: 31 + 3038 + 3038 = 6107

    # Note, because we are using keyboard sequence we the length of keypress
    # sequence can be twice the size of the password,
    Therefore, 30*2 + 1 is the max_len
    '''
    max_len = 61
    d_list = [('d', None, i) for i in range(max_len)]
    asci = list(string.ascii_letters) # =52
    punc = list(string.punctuation) # =32
    dig = list(string.digits) # =10
    chars = asci + punc + dig + [" ", "\t", "\x03", "\x04"] #32+52+10+4=98
    s_list = [('s', c, i) for c in chars for i in range(max_len)]
    i_list = [('i', c, i) for c in chars for i in range(max_len)]

    transition_table = d_list + s_list + i_list
    transition_dict_2idx = {}
    transition_dict_2path = {}
    for i in range(len(transition_table)):
        transition_dict_2idx[str(transition_table[i])] = i+3
        transition_dict_2path[i+3] = str(transition_table[i])
    with open('./trans_dict_2idx.json', 'w') as outfile:
        json.dump(transition_dict_2idx, outfile)
    with open('./trans_dict_2path.json', 'w') as outfile:
        json.dump(transition_dict_2path, outfile)
    print("Transitions dictionary created as trans_dict_2idx.json & "
          "trans_dict_2path.json")
    '''
    Read:
    if filename:
        with open(filename, 'r') as f:
            transition_dict = json.load(f)
    '''

# generate_transition_dict()
thisfolder = Path(__file__).absolute().parent
TRANS_to_IDX = json.load((thisfolder / 'trans_dict_2idx.json').open())
IDX_to_TRANS = {v: literal_eval(k) for k, v in TRANS_to_IDX.items()}
KB = Keyboard()

CACHE_SIZE = int(1e6)


def find_med_backtrace(str1, str2, cutoff=-1):
    '''
    This function calculates the Minimum Edit Distance between 2 words using
    Dynamic Programming, and asserts the optimal transition path using backtracing.
    Input parameters: original word, target word
    Output: minimum edit distance, path
    Example: ('password', 'Passw0rd') -> 2.0, [('s', 'P', 0), ('s', '0', 5)]
    '''
    # op_arr_str = ["d", "i", "c", "s"]

    # Definitions:
    n = len(str1)
    m = len(str2)
    # D[i][j]: edit distance between the first i chars of str1 and first j chars of str2
    D = np.full((n + 1, m + 1), np.inf)
    trace = np.full((n + 1, m + 1), None)
    trace[1:, 0] = list(zip(range(n), np.zeros(n, dtype=int)))
    trace[0, 1:] = list(zip(np.zeros(m, dtype=int), range(m)))
    # Initialization:
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)

    # Fill the matrices: all operations are applied to string A (str1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # delete: remove a char from A (equiv. inserting into B's tail)
            delete = D[i - 1, j] + 1
            # insert: add a char into A (equiv. deleting from B's tail)
            insert = D[i, j - 1] + 1
            if (str1[i - 1] == str2[j - 1]):
                sub = np.inf
                copy = D[i - 1, j - 1]
            else:
                sub = D[i - 1, j - 1] + 1
                copy = np.inf
            op_arr = [delete, insert, copy, sub]
            D[i, j] = np.min(op_arr)
            op = np.argmin(op_arr)  # record which move was taken
            if (op == 0):
                # delete, go down
                trace[i, j] = (i - 1, j)
            elif (op == 1):
                # insert, go left
                trace[i, j] = (i, j - 1)
            else:
                # copy or subsitute, go diag
                trace[i, j] = (i - 1, j - 1)
    # Find the path of transitions:
    i = n
    j = m
    cursor = trace[i, j]
    path = []
    while (cursor is not None):
        # 3 possible directions:
        if (cursor[0] == i - 1 and cursor[1] == j - 1):
            # diagonal - sub or copy
            if (str1[cursor[0]] != str2[cursor[1]]):
                # substitute
                path.append(("s", str2[cursor[1]], cursor[0]))
            i = i - 1
            j = j - 1
        elif (cursor[0] == i and cursor[1] == j - 1):
            # go left - insert
            path.append(("i", str2[cursor[1]], cursor[0]))
            j = j - 1
        else:
            # go down - delete
            path.append(("d", None, cursor[0]))
            i = i - 1
        cursor = trace[cursor[0], cursor[1]]
    md = D[n, m]
    del D, trace
    return md, list(reversed(path))


# Minimum Edit Distance with Backtrace (KeyPress version).
# \x04: CAPS key, \x03: Shift key
@lru_cache(maxsize=CACHE_SIZE)
def find_med_backtrace_kb(str1, str2):
    '''
    This function calculates the Minimum Edit Distance between 2 words using
    Dynamic Programming, and asserts the optimal transition path using backtracing.
    This version uses KeyPress representation.
    Input parameters: original word, target word
    Output: minimum edit distance, path
    Example:
    ('password', 'PASSword') -> 2.0 , [('i', '\x04', 0), ('i', '\x04', 4)]
    '''
    # Transform to keyboard representation:
    kb_str1 = KB.word_to_keyseq(str1)
    kb_str2 = KB.word_to_keyseq(str2)
    return find_med_backtrace(kb_str1, kb_str2)


# Decoder - given a word and a path of transition, recover the final word.
# Input: original password and transition path; Output: transformed password.
def path2word(word, path):
    '''This function decodes the word in which the given path transitions the input
    word into.  Input parameters: original word, transition path Output: decoded
    word

    '''
    if not path:
        return word
    final_word = []
    word_len = len(word)
    path_len = len(path)
    i = 0
    j = 0
    while (i < word_len or j < path_len):
        # If the current position in the word equals the operation position
        # in the path (i.e. an i/d/s operation occurs here), apply it.
        if (j < path_len and path[j][2] == i):
            if (path[j][0] == "s"):
                # substitute
                final_word.append(path[j][1])
                i += 1
                j += 1
            elif (path[j][0] == "d"):
                # delete
                i += 1
                j += 1
            else:
                # "i", insert
                final_word.append(path[j][1])
                j += 1
        # Otherwise, simply copy the current character.
        else:
            final_word.append(word[i])
            i += 1
    return ''.join(final_word)


# Decoder - given a word and a path of transition, recover the final word:
# KEYPRESS Version
def path2word_kb(word, path):
    '''
    This function decodes the word in which the given path transitions the input word into.
    This is the KeyPress version, which handles the keyboard representations.
    Input parameters: original word, transition path
    Output: decoded word
    '''
    word = KB.word_to_keyseq(word)
    if not path:
        return KB.keyseq_to_word(word)
    final_word = []
    word_len = len(word)
    path_len = len(path)
    i = 0
    j = 0
    while (i < word_len or j < path_len):
        if (j < path_len and path[j][2] == i):
            if (path[j][0] == "s"):
                # substitute
                final_word.append(path[j][1])
                i += 1
                j += 1
            elif (path[j][0] == "d"):
                # delete
                i += 1
                j += 1
            else:
                # "i", insert
                final_word.append(path[j][1])
                j += 1
        else:
            final_word.append(word[i])
            i += 1
    return (KB.keyseq_to_word(''.join(final_word)))

kb = Keyboard()
def path2word_kb_feasible(word, path, print_path=False):
    '''
    This function decodes the word in which the given path transitions the input word into.
    This is the KeyPress version, which handles the keyboard representations.
    If one of the parts components is not feasible (e.g removing a char from out of range index), it skips it
    Input parameters: original word, transition path
    Output: decoded word
    '''

    word = kb.word_to_keyseq(word)
    if not path:
        return kb.keyseq_to_word(word)
    if (print_path):
        print(path)
    final_word = []
    word_len = len(word)
    path_len = len(path)
    i = 0
    j = 0
    while (i < word_len or j < path_len):
        if ((j < path_len and path[j][2] == i) or (i >= word_len and path[j][2] >= i)):
            if (path[j][0] == "s"):
                # substitute
                final_word.append(path[j][1])
                i += 1
                j += 1
            elif (path[j][0] == "d"):
                # delete
                i += 1
                j += 1
            else:
                # "i", insert
                final_word.append(path[j][1])
                j += 1
        else:
            if (i < word_len):
                final_word.append(word[i])
                i += 1
            if (j < path_len and i > path[j][2]):
                j += 1
    final_word=[x for x in final_word if ord(x)>20]
    return (kb.keyseq_to_word(''.join(final_word)))


def pair2path(kw1, kw2, human_readable=False):
    """
    Given a pair of passwords (kw1, kw2) returns one of the possible list of
    transformations (paths) that converts pair[0] into other[1].
    kw1 and kw2 are already in keyboard sequence format.

    Returns a list of transformation indices.  If human_readable is true, then
    returns the list of raw transformations.

    E.g,
    >> pairs2path('\x02password', 'password')
    ['d', 0, None]
    """
    med, path = find_med_backtrace(kw1, kw2)
    if human_readable:
        path_indices = [str(p) for p in path]
    else:
        path_indices = path2idx(path)
    # for testing
    if random.randint(0, 1000) <= 10:
        decoded_word = path2word(kw1, path)
        if (decoded_word != kw2):
            print("Test failed on: {}".format((kw1, kw2)))
            print("Path chosen: {}".format(path))
            print("Decoded Password: {}".format(decoded_word))

    return path_indices


# Convert a human-readable path into dictionary indices.
def path2idx(path):
    '''
    This functions converts human-readable transition path to a
    dictionary-indices path (for future use in RNNs).
    Input parameters: human-readable path, dictionary
    Output: dictionary-indices path
    [('i', '\x04', 0), ('s', '\x03', 1), ('i', '2', 2), ('i', '\x04', 4)] ->
    [6076, 3008, 5737, 6080]
    '''
    idx_path = [TRANS_to_IDX.get(str(p), -1) for p in path]  # -1 if the path does not exist
    return idx_path

# Convert dictionary indices back into a human-readable path.
def idx2path(path):
    '''
    This functions converts dictionary-indices transition path to a
    human-readable path (for future use in RNNs).
    Input parameters: human-readable path, dictionary
    Output: dictionary-indices path
    [6076, 3008, 5737, 6080] ->
    [('i', '\x04', 0), ('s', '\x03', 1), ('i', '2', 2), ('i', '\x04', 4)]
    '''
    new_path=[]
    for i in path:
        if i not in [0, 1, 2]:
            new_path.append(i)
    str_path = [IDX_to_TRANS.get(p, ("<unk>", "<unk>", -1))  # <unk> if the index does not exist
                for p in new_path]
    return str_path


def csv2pws_pairs_gen(csv_fpath, line_s=0, line_e=None):
    '''
    Generator function to parse the csv file, such that every row is a list of
    username and a string of passwords list.  Using itertools, find all the
    combinations of passwords, and generate an appropriate path.  For every
    password and path, build the output password, and compare the result with
    the original pair.

    Input parameter
    @csv_fpath: path to original dataset csv
    line_s: starting line of the file
    line_e: ending line of the file
    '''
    with open(csv_fpath) as csv_file:
        # skip lines csv_file
        csv_reader = csv.reader(
            itertools.islice(csv_file, line_s, line_e), delimiter=','
        )
        for i, row in enumerate(csv_reader):
            if (len(row) != 2):
                print("File format error @ line {}\n{!r}!".format(i, row))
                break
            username, pws_string = row
            try:
                pws_list = json.loads(pws_string)
            except json.decoder.JSONDecodeError as ex:
                print(ex)
                continue
            for p in itertools.permutations(pws_list, 2):
                yield p


def csv2dataset_dict_gen(csv_fpath, human_readable=False):
    '''
    This (generator) function generates the new dataset format from the original
    one.

    The new dataset is in the form: [pass1, pass2, dictionary-indices transition
    path], one line per password pair.

    Input parameter: path to original dataset csv, path to
    the json dictionary file

    '''
    print("Started building dataset...")
    start = time.clock()
    pairs_generator = csv2pws_pairs_gen(csv_fpath)  # iterator over password pairs
    with open('trans_dataset.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(
            csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for i, pair in enumerate(pairs_generator):
            if not i:
                # skip first line
                continue
            if (len(pair[0]) > 30 or len(pair[1]) > 30):
                continue
            path_indices = pair2path(pair, human_readable)
            if (i % 50000 == 0):
                print("Progress: processed {} pairs so far".format(i))
            csv_writer.writerow([
                pair[0], pair[1], json.dumps(path_indices)
            ])
    print(
        "Dataset created in {} seconds on a total of {} passwords pairs".format(
            time.clock() -
            start,
            i))
    print("New Dataset CSV file: trans_dataset.csv")



def run_test(csv_fpath):
    '''
    This function tests the encoder-decoder functions, in order to make sure
    that for evey transition path from pass1 to pass2, the decoded password from pass1
    and the transition path is the same as pass2.
    '''
    start = time.clock()
    pws_pairs_gen = csv2pws_pairs_gen(csv_fpath)
    for i, pair in enumerate(pws_pairs_gen):
        med, path = find_med_backtrace_kb(pair[0], pair[1])
        decoded_word = path2word_kb(pair[0], path)
        if (decoded_word != pair[1]):
            print("Test failed on: {}".format(pair))
            print("Path chosen: {}".format(path))
            print("Decoded Password: {}".format(decoded_word))
        if i % 100 == 0:
            print("Done: {}".format(i))
    print("Testing done in {} seconds on a total of {} passwords pairs"
          .format(time.clock() - start, i))


if __name__ == "__main__":
    # Simple sanity check: convert a pair of key sequences to a path.
    src = "123456"
    src = Keyboard().word_to_keyseq(src)

    trg = "114514"
    trg = Keyboard().word_to_keyseq(trg)

    print(pair2path(src, trg))
