import os
import sys
import pandas as pd
import math

from datasets import Dataset,load_dataset
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src.utils.tokenize import process_test_trawling,process_train_trawling,process_train_targeted,process_test_targeted

def load_csv_dataset_and_encode(data_path,tokenizer,vocab,sep = "\t",names = ["password"],prompt_id=1):
    # This function is mainly used in trawling scenarios
    data = pd.read_csv(filepath_or_buffer = data_path,sep = sep,names = names, quoting = 3,dtype=str,keep_default_na=False)
    pws = Dataset.from_pandas(data,split=["train"])
    encoded_data = pws.map(process_train_trawling, 
                        fn_kwargs={
                        "tokenizer": tokenizer,
                        "prompt_id": prompt_id,
                        "vocab":vocab
                    })
    return encoded_data

def load_json_dataset_and_encode(data_path,tokenizer,vocab,prompt_id=0):
    # This function is mainly used in targeted scenarios as the auxiliary will be used in this scenarioes
    ds = load_dataset("json", data_files = data_path, split="train")
    encoded__ds = ds.map(
                process_test_targeted,
                fn_kwargs={
                    "tokenizer": tokenizer,
                    "prompt_id": prompt_id,
                    "vocab":  vocab,
                    "mask_indice": True,# Ensure that only pw loss is calculated during training.
                    "mask_pii": True # Ensure that only pw loss is calculated during training.
                }
                )
    return encoded__ds

def read_pw_prob_file(path,sep = "\t",is_log_prob = False):
    ret = []
    with open(path,"r") as f:
        for line in f:
            pw,prob = line.split(sep.strip("\n"))
            prob = float(prob)
            if is_log_prob:
                prob = math.epx(prob)
            ret.append((pw,prob))
    return ret

    