# Standard Library Modules
import os
import math
import time
import json
from dataclasses import dataclass, asdict

# Third-party Libraries
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Local Project Modules
from src.utils.tokenize import process_test_targeted, process_test_trawling, decode_limit,get_alpha_vocab
from src.search.search import dynamic_beam_search,remap_logits
from src.search.monto_carlo import monto_carlo,_get_guess_number
from src.utils.utils import pm_guesser, Log_config

# Utility class to convert dataclass to dict with string values
class dict_pre():
    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}

# Basic configuration for evaluation
@dataclass
class Basic_Config_For_Evaluation(dict_pre):
    base_model: str       # Path to the base model
    tokenizer: str        # Path to the tokenizer
    lora_path: str        # Path to LoRA weights
    test_path: str        # Path to test dataset

# Configuration for targeted search evaluation
@dataclass
class Targeted_Search_Config(dict_pre):
    prompt_template_id: int
    batch_size: int
    beam_width_list: int
    vocab_limit: bool
    eos_threshold: float
    max_guess_number: int

# Configuration for result logging and guesser behavior
@dataclass
class Guesser_Config(dict_pre):
    result_path: str
    log_interval: int
    log_guess_number: list[int]

def load_model(basic_config:Basic_Config_For_Evaluation, merge_lora = True):
    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path = basic_config.base_model,
        device_map=0,
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path = basic_config.tokenizer
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load LoRA model or use base if LoRA path is empty
    if basic_config.lora_path:
        peft_model = PeftModel.from_pretrained(model = base_model,
                                                    is_trainable=False,
                                                    model_id = basic_config.lora_path) 
        peft_model = peft_model.merge_and_unload() if merge_lora else peft_model
    else:
        peft_model = base_model
    pw_vocab = get_alpha_vocab(tokenizer)
    return peft_model,tokenizer,pw_vocab


# Main evaluation class
class GuessLLM_Evaluator():
    def __init__(self,
                 basic_conf: Basic_Config_For_Evaluation,
                 eval_conf: Targeted_Search_Config,
                 guesser_conf: Guesser_Config
                 ) -> None:
        self.basic_config = basic_conf
        self.eval_config = eval_conf
        self.guesser_config = guesser_conf



    # Load the model and tokenizer, with optional LoRA
    def _load_peft_model_from_pretrained(self, PW_WORD="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ "):
        self.base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.basic_config.base_model,
            device_map=0,
            torch_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.basic_config.tokenizer
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load LoRA model or use base if LoRA path is empty
        if self.basic_config.lora_path:
            self.peft_model = PeftModel.from_pretrained(model=self.base_model,
                                                        is_trainable=False,
                                                        model_id=self.basic_config.lora_path)
            self.peft_model = self.peft_model.merge_and_unload()
        else:
            self.peft_model = self.base_model
        #breakpoint()
        # Prepare vocab depending on restriction setting
        if self.eval_config.vocab_limit:
            self.vocab = {}
            for w in PW_WORD:
                self.vocab[w] = self.tokenizer(w)["input_ids"][-1]
            self.vocab[self.tokenizer.eos_token] = self.tokenizer.eos_token_id
        else:
            vocab = self.tokenizer.get_vocab()
            vocab.pop(self.tokenizer.eos_token)
            temp = list(vocab.items())
            temp.append((self.tokenizer.eos_token, self.tokenizer.eos_token_id))
            self.vocab = {k: v for k, v in temp}
        return self.peft_model

    # Initialize the guesser object and logging
    def _init_guesser(self):
        os.makedirs(self.guesser_config.result_path, exist_ok=True)
        log_config = Log_config(log_f=self.guesser_config.result_path + "/log.txt",
                                log_guess_number=self.guesser_config.log_guess_number)
        self.guesser = pm_guesser(f=self.guesser_config.result_path + "/input_output.jsonl",
                                  log_config=log_config,
                                  save_interval=self.guesser_config.log_interval)

    # Prepare dataset by loading and processing with tokenizer
    def _prepare_data_for_evaluation(self):
        self.test_ds = load_dataset("json", data_files=self.basic_config.test_path, split="train")
        self.encoded_test = [process_test_targeted(i, self.tokenizer, self.vocab,self.eval_config.prompt_template_id) for i in self.test_ds]

    # Decode token ids to string passwords
    @staticmethod
    def _decode(search_results, tokenizer, vocab_limit: bool, vocab, return_prob=True):
        guesses = []
        probs = []
        for idx,prob in search_results:
            prob = math.exp(prob.item())
            idx = idx[:-1]  # Remove EOS
            if vocab_limit:
                g_pwd = decode_limit(idx, vocab)
            else:
                g_pwd = tokenizer.decode(idx)
            guesses.append(g_pwd)
            probs.append(prob)
        return list(zip(guesses, probs)) if return_prob else guesses

    # Evaluation loop with guesser output
    @staticmethod
    def _targeted_eval(model: PeftModel, tokenizer, guesser: pm_guesser, encoded_ds: list, eval_conf: Targeted_Search_Config, vocab: dict):
        if len(encoded_ds) == 0:
            return
        
        index=0
        for item in tqdm(encoded_ds):
            # if index>=200:
            #     break
            pwd_list = []
            old_pws = item["content_dic"].get("Old password").split(' ')
            olds = []
            if isinstance(old_pws, str):
                #pwd_list.append((old_pws, 1))
                olds.append(old_pws)
            elif isinstance(old_pws, list):
                for old in old_pws:
                    if isinstance(old, str):
                        #pwd_list.append((old, 1))
                        olds.append(old)
                    elif isinstance(old, list):
                        #pwd_list.extend(zip(old, [1 for _ in range(len(old))]))
                        olds.extend(old)
            if item["target"] not in olds:
                search_results = dynamic_beam_search(
                    model=model,
                    input_ids=item["input_ids_no_response"],
                    batch_size=eval_conf.batch_size,
                    beam_width_list=eval_conf.beam_width_list,
                    vocab=list(vocab.values()),
                    eos_threshold=eval_conf.eos_threshold,
                    sorted=True
                )
                pwds_pros = GuessLLM_Evaluator._decode(search_results[:eval_conf.max_guess_number], tokenizer, eval_conf.vocab_limit, vocab)
                pwd_list.extend(pwds_pros)

                #breakpoint()
                guesser.update(
                    input=item["input_str"],
                    target=item["target"],
                    guess_list=pwd_list[:eval_conf.max_guess_number]
                )
                index+=1
    # Evaluation loop without guesser, return predictions only
    @staticmethod
    def _eval_with_output(model: PeftModel, tokenizer, encoded_ds: list, eval_conf: Targeted_Search_Config, vocab: dict):
        if len(encoded_ds) == 0:
            return
        pwd_list = []
        for item in tqdm(encoded_ds):
            old_pws = item["content_dic"].get("Old password")
            olds = []
            if isinstance(old_pws, str):
                pwd_list.append((old_pws, 1))
                olds.append(old_pws)
            elif isinstance(old_pws, list):
                for old in old_pws:
                    if isinstance(old, str):
                        pwd_list.append((old, 1))
                        olds.append(old)
                    elif isinstance(old, list):
                        pwd_list.extend(zip(old, [1 for _ in range(len(old))]))
                        olds.extend(old)
            if item["target"] not in olds:
                search_results = dynamic_beam_search(
                    model=model,
                    input_ids=item["input_ids_no_response"],
                    batch_size=eval_conf.batch_size,
                    beam_width_list=eval_conf.beam_width_list,
                    vocab=list(vocab.values()),
                    eos_threshold=eval_conf.eos_threshold,
                    sorted=True
                )
                pwds = GuessLLM_Evaluator._decode(search_results[:eval_conf.max_guess_number], tokenizer, eval_conf.vocab_limit, vocab, return_prob=False)
                pwd_list.append(pwds)
        return pwd_list

    # Save evaluation configuration as JSON
    def _save_config(self):
        config = {
            "eval time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
            "basic config": self.basic_config.dict(),
            "eval config": self.eval_config.dict(),
            "guesser config": self.guesser_config.dict(),
        }
        config_path = os.path.join(self.guesser_config.result_path, "eval_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    # Entry for evaluating and returning guesses with filtering
    def output_with_filter(self, filter_func, characters):
        torch.cuda.empty_cache()
        self._load_peft_model_from_pretrained(PW_WORD=characters)
        self._prepare_data_for_evaluation()
        guesses = self._eval_with_output(
            model=self.peft_model,
            tokenizer=self.tokenizer,
            encoded_ds=self.encoded_test,
            eval_conf=self.eval_config,
            vocab=self.vocab
        )
        f_guesses = [filter_func(guess) for guess in guesses]
        return f_guesses
    # Main entry point to run the evaluation and save results
    def eval(self):
        torch.cuda.empty_cache()
        self._load_peft_model_from_pretrained()
        self._init_guesser()
        self._prepare_data_for_evaluation()
        self._save_config()
        self._targeted_eval(model=self.peft_model,
                            tokenizer=self.tokenizer,
                            guesser=self.guesser,
                            encoded_ds=self.encoded_test,
                            eval_conf=self.eval_config,
                            vocab=self.vocab)
