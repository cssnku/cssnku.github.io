# Standard library modules
import time
import json
from typing import Literal, Optional
from dataclasses import dataclass, asdict

# Third-party libraries
import torch
import pandas as pd
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
    Trainer, TrainingArguments, MistralConfig, MistralForCausalLM
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset

# Local modules
from src.utils.tokenize import (
    process_train_targeted, process_val_targeted,
    process_train_trawling
)

from transformers import Trainer
import torch.nn as nn
import torch.nn.functional as F

class LogFixedTrainer(Trainer):
    """
    Fixed version: Compatible with different versions of transformers,
    receives all additional parameters (*args, **kwargs) and passes them to parent class.
    """
    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        # 1. Fix loss display logic
        if "loss" in logs and self.args.gradient_accumulation_steps > 1:
            # This division ensures the displayed loss is the "average" rather than "accumulated sum"
            logs["loss"] = round(logs["loss"] / self.args.gradient_accumulation_steps, 4)
        
        # 2. Call parent method and pass through all received parameters (including start_time mentioned in error)
        super().log(logs, *args, **kwargs)

class FocalLossTrainer(LogFixedTrainer): # Inherit from LogFixedTrainer to retain log fix functionality
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 1. Get labels and outputs
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # 2. Align Logits and Labels
        # Causal LM characteristic: Logits at position i predict Labels at position i+1
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten
        logits_flat = shift_logits.view(-1, shift_logits.size(-1))
        labels_flat = shift_labels.view(-1)
        
        # 3. Compute Focal Loss
        # Only compute for non-ignore_index (-100) positions
        valid_mask = labels_flat != -100
        valid_logits = logits_flat[valid_mask]
        valid_labels = labels_flat[valid_mask]
        
        if valid_labels.numel() == 0:
            return (torch.tensor(0.0).to(logits.device), outputs) if return_outputs else torch.tensor(0.0).to(logits.device)

        # Compute standard CrossEntropy (reduction='none' to get per-token loss)
        ce_loss = F.cross_entropy(valid_logits, valid_labels, reduction='none')
        
        # Get model's predicted probability for the correct class p_t
        # p_t = exp(-ce_loss)
        pt = torch.exp(-ce_loss)
        
        # Focal Loss core formula: weight = (1 - pt)^gamma
        # gamma is typically 2.0
        gamma = 2.0
        focal_loss = ((1 - pt) ** gamma * ce_loss).mean()

        # Explicitly delete large intermediate variables that are no longer needed
        del logits
        del ce_loss
        del pt

        return (focal_loss, outputs) if return_outputs else focal_loss
class PolyLossTrainer(LogFixedTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Each logit predicts the next token.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        logits_flat = shift_logits.view(-1, shift_logits.size(-1))
        labels_flat = shift_labels.view(-1)
        
        # 1. Filter Mask
        valid_mask = labels_flat != -100
        valid_logits = logits_flat[valid_mask]
        valid_labels = labels_flat[valid_mask]
        
        if valid_labels.numel() == 0:
            zero_loss = valid_logits.sum() * 0.0
            return (zero_loss, outputs) if return_outputs else zero_loss

        # 2. Compute CE Loss
        ce_loss = F.cross_entropy(valid_logits, valid_labels, reduction='none')
        
        # 3. Probability assigned to the target token
        pt = torch.exp(-ce_loss)
        
        # 4. Poly-1 Loss 
        epsilon = 1.0
        poly_loss = ce_loss + epsilon * (1 - pt)
        
        return (poly_loss.mean(), outputs) if return_outputs else poly_loss.mean()

class TopKLossTrainer(LogFixedTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        logits_flat = shift_logits.view(-1, shift_logits.size(-1))
        labels_flat = shift_labels.view(-1)
        
        # 1. Filter
        valid_mask = labels_flat != -100
        valid_logits = logits_flat[valid_mask]
        valid_labels = labels_flat[valid_mask]
        
        if valid_labels.numel() == 0:
            return (torch.tensor(0.0).to(logits.device), outputs) if return_outputs else torch.tensor(0.0).to(logits.device)

        # 2. Compute loss for all tokens (do not take mean)
        pixel_losses = F.cross_entropy(valid_logits, valid_labels, reduction='none')
        
        # 3. 【Core】Only take top k% of losses (e.g., top 30%)
        # This forces the model to focus on the characters it predicts most poorly
        k_fraction = 0.6 # This is a hyperparameter, meaning only train on the hardest 30% of characters
        k = int(valid_labels.numel() * k_fraction)
        
        # Avoid k=0
        k = max(k, 1)
        
        # Sort and take top k
        top_k_losses, _ = torch.topk(pixel_losses, k)
        
        final_loss = top_k_losses.mean()
        
        return (final_loss, outputs) if return_outputs else final_loss


# High-priority characters
class WeightedTrainer(LogFixedTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 1. Initialize weights to 1.0
        # self.model.config.vocab_size is Qwen's vocabulary size (~151936)
        self.loss_weights = torch.ones(self.model.config.vocab_size).to(self.model.device)
        
        # 2. Define characters to prioritize
        # These are the hardest to guess in passwords
        target_chars = "!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~"
        
        tokenizer = self.processing_class
        print("[INFO] Setting up Class Weights for Numbers and Symbols...")
        
        # 3. Find the token IDs for these characters and set weight to 2.0 (or 3.0)
        # Note: Must use encode_limit logic to query IDs to ensure alignment
        count = 0
        for char in target_chars:
            # Use your encode_limit logic here, or query tokenizer directly
            # Note: add_special_tokens=False to avoid getting bos/eos
            ids = tokenizer(char, add_special_tokens=False)["input_ids"]
            if len(ids) == 1: # Ensure it's a single character token
                idx = ids[0]
                self.loss_weights[idx] = 3.0 # 【Key】Amplify number/symbol loss by 3x
                count += 1
        
        print(f"[INFO] Weighted {count} tokens with weight 3.0")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Ensure weights are on the correct GPU
        if self.loss_weights.device != shift_logits.device:
            self.loss_weights = self.loss_weights.to(shift_logits.device)
            
        # Use weighted CrossEntropy
        # This is natively supported by PyTorch, extremely stable, no memory increase
        loss_fct = nn.CrossEntropyLoss(weight=self.loss_weights)
        
        loss = loss_fct(shift_logits.view(-1, self.model.config.vocab_size), shift_labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# Helper base class to convert dataclasses to string-key dictionaries
class dict_pre():
    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}

# Function version for convenience
def dict(obj):
    return {k: str(v) for k, v in asdict(obj).items()}

# Basic training configuration
@dataclass
class Basic_Config_For_Training(dict_pre):
    base_model: str
    tokenizer: str
    train_data_path: str
    validation_path: str



# Model-specific training configuration
@dataclass
class Trainer_Config(dict_pre):
    prompt_template_id: int
    precision: Literal["full", "half"]
    tokenizer_padding_side: str

# Main Trainer class
class TPG_Trainer():
    def __init__(self,
                 basic_config: Basic_Config_For_Training,
                 train_args: TrainingArguments,
                 lora_config: LoraConfig,
                 trainer_config: Trainer_Config) -> None:
        self.basic_config = basic_config
        self.train_args = train_args
        self.lora_config = lora_config
        self.trainer_config = trainer_config

    # Load pretrained base model and tokenizer
    def _load_basic_model(self):
        print("[INFO] Loading base model and tokenizer...")
        if self.trainer_config.precision == "full":
            dtype = torch.float32
        elif self.trainer_config.precision == "half":
            dtype = torch.bfloat16
        else:
            raise ValueError("Precision setting not implemented")

        self.base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.basic_config.base_model,
            device_map="auto",
            torch_dtype=dtype
        )

        print(self.base_model)

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.basic_config.tokenizer
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = self.trainer_config.tokenizer_padding_side

        # Initialize vocabulary mapping
        self.vocab = {}
        PW_WORD = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ "
        for w in PW_WORD:
            self.vocab[w] = self.tokenizer(w)["input_ids"][-1]
        self.vocab[self.tokenizer.eos_token] = self.tokenizer.eos_token_id
    
    # Prepare dataset for training and validation
    def _prepare_data_for_training(self, seo: Literal["trawling", "targeted"]):
        print(f"[INFO] Preparing data for training: scenarios = {seo}")
        if seo == "targeted":
            self.train_ds = load_dataset("json", data_files=self.basic_config.train_data_path, split="train")
           
            # =========== Filtering logic ===========
            # Purpose: If a data sample's Old password (after cleaning) contains no valid passwords different from the target, discard it.
            # This is a simple lambda filter, no need to modify prompt_template, directly filter here for maximum efficiency
            def filter_invalid_samples(example):
                target = example.get("password", "")
                knowledge = example.get("Knowledge", {})
                old_pw_str = knowledge.get("Old password", "")
                
                if not isinstance(old_pw_str, str): return False
                
                # Split
                pws = old_pw_str.split()
                # Check if any password is neither target nor valid
                valid_pws = [p for p in pws if p != target]
                
                # Keep this sample if valid password count > 0
                return len(valid_pws) > 0

            original_len = len(self.train_ds)
            # Decide whether to apply deduplication filtering
            # self.train_ds = self.train_ds.filter(filter_invalid_samples)
            print(f"[INFO] Filtered dataset: {original_len} -> {len(self.train_ds)} samples (Removed self-duplicates)")
            
            # Process dataset according to process_train_targeted
            self.encoded_train_ds = self.train_ds.map(
                process_train_targeted,
                fn_kwargs={
                    "tokenizer": self.tokenizer,
                    "prompt_id": self.trainer_config.prompt_template_id,
                    "vocab": self.vocab,
                },
            )
            # Automatically split if no validation set is added
            if self.basic_config.validation_path:
                self.validation_ds = load_dataset("json", data_files=self.basic_config.validation_path, split="train")
                self.encoded_val_ds = self.validation_ds.map(
                    process_val_targeted,
                    fn_kwargs={
                        "tokenizer": self.tokenizer,
                        "prompt_id": self.trainer_config.prompt_template_id,
                        "vocab": self.vocab
                    },
                )
            else:
                train_and_val = self.encoded_train_ds.train_test_split(test_size=0.05, seed=42)
                self.encoded_train_ds = train_and_val["train"]
                self.encoded_val_ds = train_and_val["test"]

        elif seo == "trawling":
            data = pd.read_csv(self.basic_config.train_data_path, sep="\t", names=["password"],
                               quoting=3, dtype=str, keep_default_na=False)
            self.train_ds = Dataset.from_pandas(data, split=["train"])

            self.encoded_train_ds = self.train_ds.map(
                process_train_trawling,
                fn_kwargs={
                    "tokenizer": self.tokenizer,
                    "prompt_id": self.trainer_config.prompt_template_id,
                    "vocab": self.vocab
                }
            )
            train_and_val = self.encoded_train_ds.train_test_split(test_size=0.05, seed=42)
            self.encoded_train_ds = train_and_val["train"]
            self.encoded_val_ds = train_and_val["test"]
            print(f"[INFO] Number of training samples: {len(self.encoded_train_ds)}")
            print(f"[INFO] Number of validation samples: {len(self.encoded_val_ds)}")
    
    # Train the LoRA-adapted model
    def _train(self):
        print("[INFO] Initializing PEFT model and starting training...")
        self.peft_model = get_peft_model(self.base_model, self.lora_config)
        self.peft_model.print_trainable_parameters()

        self.trainer = PolyLossTrainer( 
            model=self.peft_model,
            args=self.train_args,
            data_collator=DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                padding=True,
                label_pad_token_id=-100
            ),
            train_dataset=self.encoded_train_ds,
            eval_dataset=self.encoded_val_ds,
            processing_class=self.tokenizer
        )

        # Alternative using standard Trainer
        # self.trainer = Trainer(
        #     model=self.peft_model,
        #     args=self.train_args,
        #     data_collator=DataCollatorForSeq2Seq(
        #         tokenizer=self.tokenizer,
        #         padding=True,
        #         label_pad_token_id=-100
        #     ),
        #     train_dataset=self.encoded_train_ds,
        #     eval_dataset=self.encoded_val_ds,
        #     processing_class = self.tokenizer
        # )

        training_begin_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        self.trainer.train()
        training_end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print(f"[INFO] Training finished. Duration: {training_begin_time} ~ {training_end_time}")
        self._final_save(training_begin_time, training_end_time)

    # Save final model and config to disk
    def _final_save(self, bt, et):
        print("[INFO] Saving final model and configuration...")
        config = {
            "training begin time": bt,
            "training end time": et,
            "basic config": self.basic_config.dict(),
            "trainer config": self.trainer_config.dict(),
            "lora config": dict(self.lora_config),
            "training config": self.train_args.to_dict(),
        }

        with open(f"{self.train_args.output_dir}/train_config.json", "w") as f:
            json.dump(config, f, indent=2)
        self.trainer.save_model(f"{self.train_args.output_dir}/final/")

    # Public method to start the entire training process
    def train(self, seo: Literal["trawling", "targeted"]):
        print("[INFO] Starting full training workflow...")
        self._load_basic_model()
        self._prepare_data_for_training(seo=seo)
        self._train()
