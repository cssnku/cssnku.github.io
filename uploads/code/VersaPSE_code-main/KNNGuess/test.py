
import argparse
# def check(pws):
#     ss, res, cnt=set(), [], 0
#     for pw in pws:
#         pw=kb.keyseq_to_word(pw)
#         if pw not in ss and config.min_len<=len(pw)<config.max_len:
#             res.append(pw)
#             ss.add(pw)
#             cnt+=1
#         if cnt>config.TopK:
#             break
#     return res

# if config.use_datastore:
#     datastore = Datastore.load(config.knn_datastore_path, load_list=["vals"])
#     datastore.load_faiss_index("keys")
#     retriever = Retriever(datastore=datastore, k=config.k)
#     combiner = Combiner(lambda_=config.lambda_,
#                     temperature=config.knn_temperature, probability_dim=config.src_vocab_size)
# else:
#     datastore=0
#     retriever=0
#     combiner=0
# model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
#                        config.d_model, config.d_ff, config.n_heads, config.dropout)
# test_dataset = MTDataset(config.test_data_path)
# test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config.test_batch_size,
#                               collate_fn=test_dataset.collate_fn)
# if config.write_ans:
#     f=open(config.guess_ans_path,"w",encoding='utf-8')
# guess_cnt=0


# pop=set()
# f1=open("./tgaux_cn_toppsw.txt","r",encoding='utf-8')
# for idx,line in enumerate(f1):
#     if idx==300:break
#     lis=line.strip('\n').split('\t')
#     pop.add(lis[0])
# f1.close()

# with torch.no_grad():
#     model.load_state_dict(torch.load(config.model_path))
#     model.eval()
#     for batch in tqdm(test_dataloader):
#         target=[pw2seq.inverse_transform(x[1:].tolist()) for x in batch.trg]
#         source=[pw2seq.inverse_transform(x[1:].tolist()) for x in batch.src]
#         #print(batch.trg)
#         #print(target)
#         #break
#         decode_result = beam_search(model, batch.src, batch.src_mask, config.max_len,
#                                 config.padding_idx, config.bos_idx, config.eos_idx,
#                                 config.beam_size, config.device, datastore, retriever, combiner)
#         for i in range(len(decode_result)):
#             guess=[pw2seq.inverse_transform(x) for x in decode_result[i]]
            
#             guess=check(guess)
#             guess=guess[:700]
#             #guess=guess[:1001]
#             #print(guess)
#             tar=kb.keyseq_to_word(target[i])
#             sour=kb.keyseq_to_word(source[i])
#             if tar in guess or tar in pop:
#                 guess_cnt+=1
#                 if config.write_ans:
#                     f.write(sour+'\t'+tar+'\t'+"YES"+'\t'+str(guess_cnt)+'\n')
#             else:
#                 if config.write_ans:
#                     f.write(sour+'\t'+tar+'\t'+"NO"+'\t'+str(guess_cnt)+'\n')
        

# print(guess_cnt)
# print(guess_cnt/13364)



# popular_pws=get_popular_pws() ########################################################
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--test_data_path",type=str,default="./data/test.txt")
    parser.add_argument("--guess_ans_path",type=str,default="./guess_ans.txt")
    parser.add_argument("--guess_ans_path_nomix",type=str,default="./guess_ans_nomix.txt")
    parser.add_argument("--model_path",type=str,default="./model.pt")
    parser.add_argument("--knn_datastore_path",type=str,default="./knn_datastore")
    parser.add_argument("--gpu_id",type=int,default=0)
    args=parser.parse_args()
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    import torch
    from PW2SEQ import PW2SEQ
    import config
    import numpy as np
    from beam_decoder import beam_search
    from model import make_model
    from collections import defaultdict
    from tqdm import tqdm
    pw2seq=PW2SEQ()
    pw2seq_dict=pw2seq.pw2seq_dict
    from knn.datastore.datastore import Datastore
    from knn.retriever.retriever import Retriever
    from knn.combiner.combiner import Combiner
    from lz_delete import beam_decode2
    from utils import kb,get_popular_pws
    print(args.gpu_id)

    test_pws=[] 
    f=open(args.test_data_path,"r",encoding='utf-8')
    for line in f:
        lis=line.strip('\n').split('\t')
        test_pws.append((lis[0],lis[1]))
    f.close()
    test_cnt=len(test_pws)
    if config.write_ans:
        f=open(args.guess_ans_path,"w",encoding='utf-8') ###########
        f1=open(args.guess_ans_path_nomix,"w",encoding='utf-8')############
    #guess_cnt=0

    if config.use_datastore:
        datastore = Datastore.load(args.knn_datastore_path, load_list=["vals"])
        datastore.load_faiss_index("keys")
        retriever = Retriever(datastore=datastore, k=config.k)
        combiner = Combiner(lambda_=config.lambda_,
                        temperature=config.knn_temperature, probability_dim=config.src_vocab_size)
    else:
        datastore=0
        retriever=0
        combiner=0

    cc=0 #######
    cnt=0
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                    config.d_model, config.d_ff, config.n_heads, config.dropout)
    with torch.no_grad():
        BOS = config.bos_idx  # 2
        EOS = config.eos_idx  # 3
        model.load_state_dict(torch.load(args.model_path))
        model.eval()
        for sour,target in tqdm(test_pws):
            sour_kb=kb.word_to_keyseq(sour)
            src_tokens = [[BOS] + pw2seq.transform(sour_kb) + [EOS]]
            src = torch.LongTensor(np.array(src_tokens)).to(config.device)
            src_mask = (src != 0).unsqueeze(-2)
            decode_result = beam_decode2(model, src, src_mask, config.max_len,
                                        config.padding_idx, config.bos_idx, config.eos_idx,datastore,retriever,combiner)
            #translation = [pw2seq.inverse_transform(_s) for _s in decode_result]
            #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # mixing
            translation = [(_s[0],pw2seq.inverse_transform(_s[1])) for _s in decode_result]
            
            
            decode_ans_list=[x[1] for x in translation]
            #decode_ans_set=set(decode_ans_list)


            for idx,tar_pw in enumerate(decode_ans_list):
                
                #f1.write(kb.keyseq_to_word(tar_pw)+'\n') ##########3
                if kb.keyseq_to_word(tar_pw)==target:
                    f1.write(sour+'\t'+target+'\t'+str(idx)+'\n')
                    print(sour)
                    print(target)
                    print(kb.keyseq_to_word(tar_pw))
                    f1.flush()
                    break
            # f1.write('**************************'+'\n')


            ss=defaultdict(float)
            for prob,tran in translation:
                ss[tran]=prob

            # for popular_pw,log_prob in popular_pws:
            #     if ss[popular_pw]==0:
            #         ss[popular_pw]=log_prob
            #     else:
            #         ss[popular_pw]=max(ss[popular_pw],log_prob)
            ss=sorted(ss.items(),key=lambda x:x[1],reverse=True)
            translation=[x[0] for x in ss[:config.TopK]]
            #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            for idx,tar_pw in enumerate(translation):
                if kb.keyseq_to_word(tar_pw)==target:
                    f.write(sour+'\t'+target+'\t'+str(idx)+'\n')
                    cnt+=1
                    break
    #print(cnt/test_cnt)
    f.close()
    f1.close()
