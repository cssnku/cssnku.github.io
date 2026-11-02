import os
import configparser

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments
)

from datasets import load_dataset, Dataset

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)

from src.utils.tokenize import (
    process_train_targeted,
    process_train_trawling
)

from src.search.search import remap_logits


class DLDataCollator(DataCollatorForSeq2Seq):
    def __init__(self,s_tokenizer,t_tokenizer,seo):
         super().__init__(tokenizer=s_tokenizer,padding=True,label_pad_token_id=-100)

         self.s_tokenizer = s_tokenizer
         self.t_tokenizer = t_tokenizer

         self.s_datacollector = DataCollatorForSeq2Seq(tokenizer=s_tokenizer,padding=True,label_pad_token_id=-100)
         self.t_datacollector = DataCollatorForSeq2Seq(tokenizer=t_tokenizer,padding=True,label_pad_token_id=-100)
         self.s_vocab = get_vocab_dict(s_tokenizer)
         self.t_vocab = get_vocab_dict(t_tokenizer)
         self.seo = seo
    def __call__(self, features, return_tensors=None):
        if self.seo == "trawling":
            s_feartures = [process_train_trawling(example,self.s_tokenizer,1,self.s_vocab) for example in features]
            t_feartures = [process_train_trawling(example,self.t_tokenizer,1,self.t_vocab) for example in features]
        elif self.seo == "targeted":
            s_feartures = [process_train_targeted(example,self.s_tokenizer,self.s_vocab,0,True,True) for example in features]
            t_feartures = [process_train_targeted(example,self.t_tokenizer,self.t_vocab,0,True,True) for example in features]
        else:
            raise NotImplementedError
        
        pred_features = self.s_datacollector.__call__(s_feartures, return_tensors)
        expl_features = self.t_datacollector.__call__(t_feartures, return_tensors)
        return {
            'student': pred_features,
            'teacher': expl_features,
        }
def get_vocab_dict(tokenizer):
    vocab = {}
    PW_WORD = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ "
    for w in PW_WORD:
        vocab[w] = tokenizer(w)["input_ids"][-1]
    vocab[tokenizer.eos_token] = tokenizer.eos_token_id
    return vocab  

def compute_fkl(
        logits, 
        teacher_logits, 
        target, 
        padding_id,
        reduction="sum",
        temp = 1.0, 
        
    ):
        logits = logits / temp
        teacher_logits = teacher_logits / temp

        log_probs = torch.log_softmax(logits, -1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
        teacher_log_probs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
        kl = (teacher_probs * (teacher_log_probs - log_probs)) 
        kl = kl.sum(-1)
        if reduction == "sum":
            pad_mask = target.eq(padding_id)
            kl = kl.masked_fill_(pad_mask, 0.0)
            kl = kl.sum()

        return kl

def extract_pw_logits(logits, labels, ignore_id=-100):
    # Create a mask to identify which positions are not equal to ignore_id
    mask = labels != ignore_id  # shape: (batch_size, sequence_length)
    
    # Calculate the number of valid tokens per batch (excluding ignore_id positions)
    valid_token_counts = mask.sum(dim=1)
    
    # Find the maximum number of valid tokens in any batch for padding
    max_valid_tokens = valid_token_counts.max().item()

    # Use the mask to extract the logits corresponding to valid positions
    valid_logit_values = logits[mask.unsqueeze(-1).expand(-1, -1, logits.size(2))]  # shape: (num_valid_tokens, vocab_size)
    
    # Reshape the extracted logits to (batch_size, num_valid_tokens, vocab_size)
    valid_logit_values = valid_logit_values.view(-1, logits.size(2))
    
    # Padding: Create a tensor filled with ignore_id to pad the logits
    padded_logit_values = torch.full(
        (logits.size(0), max_valid_tokens, logits.size(2)), 
        ignore_id, dtype=logits.dtype, device=logits.device
    )
    
    # Create a padded label tensor, also filled with ignore_id
    padded_labels = torch.full(
        (logits.size(0), max_valid_tokens), 
        ignore_id, dtype=labels.dtype, device=labels.device
    )
    
    # Fill the padded logits and labels with valid values
    for i in range(logits.size(0)):
        begin = sum(valid_token_counts[:i])
        padded_logit_values[i, :valid_token_counts[i], :] = valid_logit_values[begin:begin+valid_token_counts[i], :]
        padded_labels[i, :valid_token_counts[i]] = labels[i, mask[i]]

    return padded_logit_values, padded_labels

class KGTrainer(Trainer):
    
    def __init__(
        self,
        model = None,
        teacher_model = None,
        if_use_entropy = False,
        args = None,
        data_collator = None, 
        train_dataset = None,
        eval_dataset = None,
        teacher_tokenizer = None,
        student_tokenizer = None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            #processing_class = student_tokenizer,
        )
        self.teacher_pw_vocab = get_vocab_dict(teacher_tokenizer)
        self.teacher_model = teacher_model

        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.student_pw_vocab = get_vocab_dict(student_tokenizer)
        self.if_use_entropy = if_use_entropy
        
        self.student_vocab = list(self.student_pw_vocab.values())
        self.teacher_vocab = list(self.teacher_pw_vocab.values())
    
    def compute_loss(self, model, inputs, return_outputs=False):
        
        outputs = model(**inputs["student"])
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs["teacher"])
        
        loss = outputs.loss
        logits = remap_logits(self.student_vocab, outputs.logits)
        teacher_logits = remap_logits(self.teacher_vocab,teacher_outputs.logits)
        
        # Since the teacher model and student model use different tokenizers, 
        # first we need to trim the sequences to ensure consistent length
        s_pw_logits, labels = extract_pw_logits(logits, inputs['student']['labels'])
        t_pw_logits, _ = extract_pw_logits(teacher_logits, inputs['teacher']['labels'])


        kl = compute_fkl(s_pw_logits, t_pw_logits, labels, padding_id=-100, temp=2.0)
        
        if self.if_use_entropy:
            loss_total = 0.5 * kl + 0.5 * loss
        else:
            loss_total = kl
        
        return (loss_total, outputs) if return_outputs else loss_total


def run_distillation(
    student_model_path: str,
    teacher_model_path: str,
    teacher_lora_path: str,
    dataset_path: str,
    output_dir: str,
    seo: str ,
    train_args:TrainingArguments,
    use_teacher_entropy: bool = False,
    resume: bool = False,
    cuda_id:str = "0"
):

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id


    student_tokenizer = AutoTokenizer.from_pretrained(student_model_path)
    student_tokenizer.pad_token_id = student_tokenizer.eos_token_id
    student_tokenizer.padding_side = "right"
    student_model = AutoModelForCausalLM.from_pretrained(student_model_path, torch_dtype="bfloat16", device_map=0)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.2,
        task_type=TaskType.CAUSAL_LM
    )
    student_model = get_peft_model(student_model, lora_config)

    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)
    teacher_tokenizer.pad_token_id = teacher_tokenizer.eos_token_id
    teacher_tokenizer.padding_side = "right"
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_path, torch_dtype="bfloat16", device_map=0)
    teacher_model = PeftModel.from_pretrained(teacher_model, teacher_lora_path, torch_dtype="bfloat16")
    teacher_model = teacher_model.merge_and_unload()
    teacher_model.eval()


    if seo == "trawling":
        data = pd.read_csv(dataset_path, sep="\t", names=["password"], quoting=3, dtype=str, keep_default_na=False)
        train_dataset = Dataset.from_pandas(data)
        

    elif seo == "targeted":
        train_dataset = load_dataset("json", data_files=dataset_path, split="train")
    else:
        raise ValueError("Unsupported seo mode.")


    trainer = KGTrainer(
        model=student_model,
        teacher_model=teacher_model,
        if_use_entropy=use_teacher_entropy,
        args=train_args,
        train_dataset=train_dataset,
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        data_collator=DLDataCollator(s_tokenizer=student_tokenizer, t_tokenizer=teacher_tokenizer,seo=seo)
        
    )

    trainer.train(resume_from_checkpoint=resume)
    trainer.save_state()
    student_model.save_pretrained(output_dir)

def run_distill_using_config(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    student_model_path = config["model"]["student_model_path"]
    teacher_model_path = config["model"]["teacher_model_path"]
    teacher_lora_path = config["model"]["teacher_lora_path"]
    dataset_path = config["data"]["dataset_path"]
    seo = config["data"]["seo"]
    output_dir = config["training"]["output_dir"]


    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=int(config["training"]["num_train_epochs"]),
        per_device_train_batch_size=int(config["training"]["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(config["training"]["gradient_accumulation_steps"]),
        logging_steps=int(config["training"]["logging_steps"]),
        report_to='tensorboard',
        save_strategy='epoch',
        save_total_limit=int(config["training"]["save_total_limit"]),
        bf16=True,
        learning_rate=float(config["training"]["learning_rate"]),
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        warmup_ratio=float(config["training"]["warmup_ratio"]),
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        dataloader_num_workers=int(config["training"]["dataloader_num_workers"])
    )


    run_distillation(
        student_model_path=student_model_path,
        teacher_model_path=teacher_model_path,
        teacher_lora_path=teacher_lora_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        seo=seo,
        train_args=args,
        use_teacher_entropy=True,
        resume=False,
    )

        