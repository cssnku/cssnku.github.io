import argparse
import time

import torch
import torch.nn.functional as functional
from word2keypress import Keyboard
import numpy as np

parser = argparse.ArgumentParser(description='Find cosine-similarity password pairs between base and train dicts.')
parser.add_argument('--cs_threshold', type=float, default=0.5,
                    help='cosine similarity threshold, pairs with cs <= this value are filtered out (default: 0.5)')
parser.add_argument('--topk', type=int, default=5,
                    help='number of top-k candidates considered per train password (default: 5)')
parser.add_argument('--base_dict', type=str,
                    default='../../formatted_data/untargeted/Tianya_withcount.txt',
                    help='path to base dict file with count (default: Tianya_withcount.txt)')
parser.add_argument('--train_dict', type=str,
                    default='../../formatted_data/untargeted/Tianya_withcount.txt',
                    help='path to train dict file with count (default: Tianya_withcount.txt)')
parser.add_argument('--threshold', type=int, default=100,
                    help='minimum frequency of train passwords to consider (default: 100)')
parser.add_argument('--output', type=str,
                    default='../intermediate_data/pwdpair.txt',
                    help='path to output pwdpair file (default: ../intermediate_data/pwdpair.txt)')
args = parser.parse_args()

class TwoGramTokenizer:
    def __init__(self):
        self.gram2id={}
        self.id2gram={}
        # gram2id["\x01"]=0 # BOS
        # gram2id["\x02"]=1 # EOS

    def TrainOnPwds(self, pwds):
        for pw in pwds:
            pw="\x01"+pw+"\x02"
            for i in range(0,len(pw)-1):
                if pw[i:i+2] not in self.gram2id:
                    self.gram2id[pw[i:i+2]]=len(self.gram2id)+1
        print(self.gram2id)
        for i in self.gram2id:
            self.id2gram[self.gram2id[i]]=i
        print(self.id2gram)

    def Pwd2Tokens(self, pwd):
        # result=[]
        count=0
        # for pw in pwds:
        tokens=[]
        pw="\x01"+pwd+"\x02"
        for i in range(0,len(pw)-1):
            tokens.append(self.gram2id[pw[i:i+2]])
        res=np.zeros(len(self.gram2id)+1)
        for i in tokens:
            res[i]+=1
        tokens=res.tolist()
        count+=1
        if count%1000==0:
            print(count)
        return tokens

    def Pwds2Tokens(self,pwds):
        result=[]
        for pwd in pwds:
            result.append(self.Pwd2Tokens(pwd))
        return result

names=["126"]

for name in names:

    # 用于计算cosine similarity的tokenizer
    gram_2tokenizer=TwoGramTokenizer()


    # base_dict=open(f'../dataset/untargeted/{param.base_dict}-withcount.txt','r')
    # train_dict=open(f'../dataset/untargeted/{param.train_dict}-withcount.txt','r')
    # test_dict=open(f'../dataset/untargeted/{param.test_set}-withcount.txt','r')

    base_dict = open(args.base_dict, 'r', encoding='ascii', errors='ignore')
    train_dict = open(args.train_dict, 'r', encoding='ascii', errors='ignore')

    all_pws=[]
    original_base=[]
    original_train=[]
    original_test=[]

    base_pws=[]
    base_counts=[]
    kb = Keyboard()
    count=0
    for i in base_dict:
        temp,pwd=i.split('\t',1)
        if len(pwd)>30 or int(temp)<4:
            continue
        original_base.append(pwd)
        pwd = kb.word_to_keyseq(pwd)
        base_pws.append(pwd)
        all_pws.append(pwd)
        base_counts.append(temp)

        print(pwd)
        # tokenizedDict.append(pass2token(pwd))
        count+=1
        if count%10==0:
            print(count)
        if count>1000:
            break

    train_pws=[]
    train_counts=[]
    kb = Keyboard()
    count=0
    for i in train_dict:
        temp,pwd=i.split('\t',1)
        if len(pwd)>30 or int(temp)<100:
            continue
        original_train.append(pwd)
        pwd = kb.word_to_keyseq(pwd)
        train_pws.append(pwd)
        train_counts.append(temp)
        all_pws.append(pwd)
        # tokenizedDict.append(pass2token(pwd))
        count+=1
        if count%10==0:
            print(count)

    # test_pws=[]
    # test_counts=[]
    # kb = Keyboard()
    # count=0
    # for i in test_dict:
    #     pwd,temp=i.split('\t',1)
    #     if len(pwd)>30:
    #         continue
    #     original_test.append(pwd)
    #     pwd = kb.word_to_keyseq(pwd)
    #     test_pws.append(pwd)
    #     test_counts.append(temp)
    #     all_pws.append(pwd)
    #     # tokenizedDict.append(pass2token(pwd))
    #     count+=1
    #     if count%10==0:
    #         print(count)

    gram_2tokenizer.TrainOnPwds(all_pws)
    result=gram_2tokenizer.Pwds2Tokens(base_pws)
    result=torch.tensor(result)

    device = torch.device("cuda")
    result=result.to(device)

    # training_dict=open(f'./original_dataset/{param.train_dict}_withcount.txt')
    pwd_pair=[]
    count=0
    start = time.time()
    for i in range(len(train_pws)):
        count+=1
        if count%1000==0:
            print(count)
            print(len(pwd_pair))
            end = time.time()
            print(end - start)
    
        temp, pwd = train_counts[i], train_pws[i]
        if len(pwd)>30:
            continue
        if int(temp)<args.threshold:
            break
        # res_curr=gram_2tokenizer.Pwd2Tokens(pwd)
        token_curr=gram_2tokenizer.Pwd2Tokens(pwd)
        token_curr=torch.tensor(token_curr)
        token_curr=token_curr.to(device)
        highest_idx = []
        curr = functional.cosine_similarity(token_curr, result, dim=1)
        cs,top=torch.topk(input=curr,k=args.topk)
        for i1 in range(len(top)):
            if cs[i1]>args.cs_threshold:
                temp_pair=[]
                temp_pair.append(original_base[top[i1]])
                temp_pair.append(original_train[i])
                temp_pair.append(temp)
                pwd_pair.append(temp_pair)
    
    pwdpair=open(args.output,'w',encoding='utf-8')
    for i in pwd_pair:
        for i2 in range(int(int(i[2])/args.threshold)):
            pass1=i[0].rstrip('\n')
            pass2 = i[1].rstrip('\n')
            pwdpair.write(pass1)
            pwdpair.write('\t')
            pwdpair.write(pass2)
            pwdpair.write('\t')
            pwdpair.write(i[2])
            pwdpair.write('\n')
