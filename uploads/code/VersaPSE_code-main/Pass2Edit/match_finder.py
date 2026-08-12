import time

import torch
import torch.nn.functional as functional
from keras.preprocessing.text import Tokenizer
from word2keypress import Keyboard

import hyper_param as param

# Tokenizer used to compute cosine similarity between 2-gram vectors
gram_2tokenizer = Tokenizer(num_words=None,
                            filters='',
                            lower=False,
                            split='\t',
                            char_level=False,
                            oov_token=None,
                            document_count=0)


def pass2token(passwd):
    """Split a password string into 2-grams for the tokenizer.

    The password is wrapped with begin/end markers ('\\u59cb'/'\\u7ec8'), and the
    2-grams are joined with '\\t' (tabs are used instead of spaces because some
    passwords contain spaces).
    """
    passwd = '始' + passwd + '终'
    token = str()
    for i in range(len(passwd) - 1):
        token += passwd[i:i + 2]
        token += '\t'
    return token


start = time.time()
base_dict = open(f'./original_dataset/{param.base_dict}_withcount.txt', 'r')
count = 0

orig_basepwd = []    # all raw base passwords
tokenizedDict = []   # 2-gram-tokenized base passwords
orig_baseCount = []  # occurrence counts of the base passwords

# Load the base dictionary (limit to 1000 passwords with count >= 4 for speed)
kb = Keyboard()
for i in base_dict:
    temp, pwd = i.split('\t', 1)
    if len(pwd) > 30 or int(temp) < 4:
        continue
    orig_basepwd.append(pwd)
    orig_baseCount.append(temp)
    pwd = kb.word_to_keyseq(pwd)
    print(pwd)
    tokenizedDict.append(pass2token(pwd))
    count += 1
    if count % 10 == 0:
        print(count)
    if count > 1000:
        break
end = time.time()
print("time" + str(end - start))
print(len(tokenizedDict))

# Fit the 2-gram tokenizer on the base dictionary
start = time.time()
gram_2tokenizer.fit_on_texts(tokenizedDict)
end = time.time()
print(gram_2tokenizer.word_index)
print("time" + str(end - start))

# Convert base passwords to count matrices (on GPU for fast similarity search)
start = time.time()
result = gram_2tokenizer.texts_to_matrix(tokenizedDict, mode='count')
result = torch.from_numpy(result)
end = time.time()
print("time" + str(end - start))

device = torch.device("cuda")
result = result.to(device)

# Match each training password to its closest base passwords using cosine
# similarity, and generate password pairs (base, pwd) with the training count.
training_dict = open(f'./original_dataset/{param.train_dict}_withcount.txt')
pwd_pair = []
count = 0
start = time.time()
for i in training_dict:
    count += 1
    if count % 1000 == 0:
        print(count)
        print(len(pwd_pair))
        end = time.time()
        print(end - start)

    temp, pwd = i.split('\t', 1)
    if len(pwd) > 30:
        continue
    if int(temp) < param.threshold:
        break
    res_curr = pass2token(pwd)
    token_curr = gram_2tokenizer.texts_to_matrix([res_curr])
    token_curr = torch.from_numpy(token_curr)
    token_curr = token_curr.to(device)
    highest_idx = []
    curr = functional.cosine_similarity(token_curr, result, dim=1)
    cs, top = torch.topk(input=curr, k=5)
    for i1 in range(len(top)):
        if cs[i1] > 0.5:
            temp_pair = []
            temp_pair.append(orig_basepwd[top[i1]])
            temp_pair.append(pwd)
            temp_pair.append(temp)
            pwd_pair.append(temp_pair)

# Write the training pairs, replicating each pair by (count / threshold) times
pwdpair = open('./intermediate_data/pwdpair.txt', 'w', encoding='utf-8')
for i in pwd_pair:
    for i2 in range(int(int(i[2]) / param.threshold)):
        pass1 = i[0].rstrip('\n')
        pass2 = i[1].rstrip('\n')
        pwdpair.write(pass1)
        pwdpair.write('\t')
        pwdpair.write(pass2)
        pwdpair.write('\t')
        pwdpair.write(i[2])
        pwdpair.write('\n')

# Match the test set against the base dictionary (keep the single best match)
training_dict = open(f'./{param.strategy}/{param.test_set}_withcount.txt')
pwd_pair = []
count = 0
start = time.time()
for i in training_dict:
    count += 1
    if count % 1000 == 0:
        print(count)
        print(len(pwd_pair))
        end = time.time()
        print(end - start)

    temp, pwd = i.split('\t', 1)
    if int(temp) < 4:
        break
    res_curr = pass2token(pwd)
    token_curr = gram_2tokenizer.texts_to_matrix([res_curr])
    token_curr = torch.from_numpy(token_curr)
    token_curr = token_curr.to(device)
    highest_idx = []
    curr = functional.cosine_similarity(token_curr, result, dim=1)
    cs, top = torch.topk(input=curr, k=1)
    for i1 in range(len(top)):
        if cs[i1] > 0:
            temp_pair = []
            temp_pair.append(orig_basepwd[top[i1]])
            temp_pair.append(pwd)
            temp_pair.append(temp)
            pwd_pair.append(temp_pair)

# Write the test pairs
pwdpair = open('./intermediate_data/test_pwdpair.txt', 'w', encoding='utf-8')
for i in pwd_pair:
    pass1 = i[0].rstrip('\n')
    pass2 = i[1].rstrip('\n')
    pwdpair.write(pass1)
    pwdpair.write('\t')
    pwdpair.write(pass2)
    pwdpair.write('\t')
    pwdpair.write(i[2])
    pwdpair.write('\n')