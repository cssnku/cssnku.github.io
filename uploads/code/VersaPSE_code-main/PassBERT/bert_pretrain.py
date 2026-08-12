import torch
from tqdm import tqdm
import string
from datasets import Dataset
from transformers import (
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
import config as config

from transformers import PreTrainedTokenizerFast
import json
import os


def build_char_level_tokenizer(charset, max_len, save_dir="./tokenizer"):
    """Build a character-level tokenizer and persist it to disk."""
    os.makedirs(save_dir, exist_ok=True)

    # 1. Build the character vocabulary (special tokens first).
    vocab = {
        "[CLS]": 0,
        "[SEP]": 1,
        "[PAD]": 2,
    }
    for i, char in enumerate(charset):
        vocab[char] = 3 + i

    mask_id = 3 + len(charset)
    unk_id = mask_id + 1
    vocab["[MASK]"] = mask_id
    vocab["[UNK]"] = unk_id

    # 2. Build a valid tokenizer.json configuration.
    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": {
            "length": max_len,
            "strategy": {"Fixed": max_len},
            "direction": "Right",
            "pad_to_multiple_of": None,
            "pad_id": vocab["[PAD]"],
            "pad_type_id": 0,
            "pad_token": "[PAD]"
        },
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": None,
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "WordPiece",
            "vocab": vocab,
            "unk_token": "[UNK]",
            "continuing_subword_prefix": "",
            "max_input_chars_per_word": 1000
        }
    }

    # 3. Persist tokenizer.json to disk.
    tokenizer_file = os.path.join(save_dir, "tokenizer.json")
    with open(tokenizer_file, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_json, f, ensure_ascii=False, indent=2)

    # 4. Load and return the tokenizer.
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_file,
        model_max_length=max_len,
        pad_token="[PAD]",
        mask_token="[MASK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        unk_token="[UNK]",
    )

    # 5. Also save the tokenizer config and special-tokens map
    #    (generates tokenizer_config.json, special_tokens_map.json, etc.).
    tokenizer.save_pretrained(save_dir)

    return tokenizer, len(vocab)

if __name__ == '__main__':
    charset = string.printable[:-5]  # 95 printable ASCII characters
    MAX_SEQ_LEN = config.MAX_LEN + 2

    # Build the real char-level tokenizer.
    tokenizer, vocab_size = build_char_level_tokenizer(charset, MAX_SEQ_LEN)

    # Configure the BERT model for masked language modeling.
    bert_config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=512,
    )
    model = BertForMaskedLM(bert_config)

    # Data: raw password text is fed directly to the tokenizer
    # (no manual pre-tokenization into input ids is needed).
    def read_passwords(fp):
        cnt = 0
        with open(fp, 'r', encoding='utf-8') as f:
            for line in f:
                pw = line.rstrip('\n')
                if pw:
                    yield {"text": pw}
                cnt += 1

    def read_passwords_withcount(fp):
        """Read a 'count \\t password' file, repeating each password by its count."""
        cnt = 0
        with open(fp, 'r', encoding='utf-8') as f:
            for line in f:
                count, pw = line.rstrip('\n').split('\t')
                count = int(count)
                if pw:
                    for i in range(count):
                        yield {"text": pw}
                cnt += count

    from datasets import Dataset
    raw_dataset = Dataset.from_generator(
        read_passwords_withcount,
        gen_kwargs={"fp": "./dataset/Rockyou-withcount.txt"}
    )

    # Tokenize the raw text (auto [CLS]/[SEP]/padding).
    def tokenize_function(examples):
        texts = examples["text"]
        if isinstance(texts, list):
            texts = [t[:MAX_SEQ_LEN - 2] for t in texts]  # truncate
        else:
            texts = texts[:MAX_SEQ_LEN - 2]
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LEN,
            return_special_tokens_mask=True,  # required by the MLM collator
        )

    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    # Use the official MLM data collator.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir="./bert_pretrain_english",
        num_train_epochs=1,
        per_device_train_batch_size=4096,
        save_steps=1000,
        logging_steps=500,
        learning_rate=1e-4,
        weight_decay=0.01,
        fp16=True,
        dataloader_num_workers=0,  # set to 0 on Windows
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()