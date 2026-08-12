# PassBERT: BERT-based Password Reuse Strength Meter

This is a README file for a submodule in the RAID'26 paper *VersaPSE: Versatile
Password Strength Evaluation Using Continual Learning*.

PassBERT models the transformation from a source password to a target
password as a sequence of **in-place edit operations**. For a given source
password, the model predicts, at every character position, an edit operation
(keep / delete / substitute / double-substitute). The probability of the
target password's edit path serves as the password strength.

Note that this is a reproduced version of PassBERT, not the code provided by the original paper. The open-source implementation was built under a very old version of TensorFlow, which isn't widely supported on recent GPU architectures.

## Pipeline Overview

```
raw (src, trg) pairs
      │
      ▼
editpath.py   ──► in-place edit paths (src \t trg \t encoded_labels)
      │
      ▼
bert_pretrain.py ──► MLM pretrained char-level BERT (optional, Chinese/English)
      │
      ▼
bert_local_train.py ──► local model per dataset (from pretrained BERT)
      │
      ▼
bert_global_train.py ──► local-to-global model (fine-tune on rockyou/tianya)
      │
      ▼
eval.py ──► reuse-strength probabilities
```

## File Overview

| File                  | Purpose                                                                                                                       |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `config.py`           | Global configuration: maximum password length (`MAX_LEN = 30`).                                                              |
| `model.py`            | The `PassBERT` model: char-level BERT encoder + per-position linear classification head. Includes `freeze_layers()` and `prob_extract()`. |
| `utils.py`            | Character vocabulary, password -> token-id conversion, dataset classes and collate functions for the edit-path labeling task. |
| `editpath.py`         | Edit-distance (Levenshtein) alignment of password pairs and conversion into in-place edit operation paths. We borrowed this file from the original PassBERT implementation for consistency.                    |
| `bert_pretrain.py`    | Pre-trains a char-level BERT (Masked Language Modeling) on a password corpus (`Rockyou-withcount.txt`).                       |
| `bert_local_train.py` | Local training: initializes BERT from a pretrained checkpoint and trains the whole network on one dataset's edit paths.       |
| `bert_global_train.py`| Local-to-global training: starts from a local checkpoint, freezes most layers, fine-tunes on the global corpus.               |
| `train.py`            | General-purpose training script: builds global base models, global-to-local adaptation, and no-freeze variants.               |
| `eval.py`             | Evaluation: computes the probability of each ground-truth edit path in a test file.                                          |


## Data Format

Raw password-pair files are **tab-separated**, one pair per line:

```
<source_password>\t<target_password>
```

- Raw (src, trg) pairs come from `../dataset/`:
  - Training: `../dataset/train/pseudo_train_data-sister.txt` and
    `../dataset/train/pseudo_train_data-popular.txt`
  - Test: `../dataset/test/pseudo_test_data-targeted.txt` and
    `../dataset/test/pseudo_test_data-untargeted.txt`

`editpath.py` converts these into labeled files with three tab-separated
columns, which are consumed by the training and evaluation scripts:

```
<source_password>\t<target_password>\t<encoded_edit_path>
```

## How to run

All commands are run from the `PassBERT/` directory and require a CUDA GPU.

### Step 0. Pre-train BERT with masked LM

Build the char-level tokenizer and pre-train a BERT encoder on the password
corpus:

```bash
python bert_pretrain.py
```

The pretrained BERT is included in ./bert_pretrain_chinese and ./bert_pretrain_english, which are trained using Tianya and Rockyou datasets respectively.

### Step 1. Prepare data: generate in-place edit paths

Convert the raw (src, trg) pairs of each experiment into labeled edit-path files:

```bash
python editpath.py -c ../dataset/train/pseudo_train_data-popular.txt -o ./dataset/inplace_edits.txt
```

editpath.py is borrowed from the original PassBERT implementation for consistency. 

Transformation into editpaths is needed for BOTH training and testing data. Remember to use --pseudo True for test sets, as PassBERT cannot generate editpaths for some password pairs (a limitation acknowledged by the original paper).

Optionally, you can also modify src/trg_pos, which stands for the positions of source/target passwords in a tab separated line, respectively.

### Step 2. Local reuse training



```bash
python bert_local_train.py \
    --train_file ./dataset/inplace_edits_local.txt \
    --pretrain_bert_path ./bert_pretrain_chinese/checkpoint-7545 \
    --save_path ./bert_local.pth \
    --epochs 5
```

### Step 3. Local-to-global training

Fine-tune each local model (with most layers frozen) on the global corpus:

```bash
python bert_global_train.py \
    --train_file ./dataset/inplace_edits_global.txt \
    --load_ckpt ./bert_local_last.pth \
    --save_path ./bert_local_to_global.pth \
    --epochs 1
```

### Step 4. Evaluation

Compute the reuse-strength probability of every pair in a labeled test file
(one probability per line, written to `--output_path`):

```bash
python eval.py \
    --test_file ./dataset/inplace_edits-targeted.txt \
    --load_ckpt ./bert_local_to_global_last.pth \
    --output_path ./test_result.txt \
```

## Common Arguments

| Argument          | Scripts                  | Description                                              |
| ----------------- | ------------------------ | -------------------------------------------------------- |
| `--train_file`    | train / *_train          | Labeled edit-path training file.                         |
| `--test_file`     | eval                     | Labeled (or raw pair) test file.                         |
| `--load_ckpt`     | train / eval             | Checkpoint to initialize / evaluate from.                |
| `--pretrain_bert_path` | bert_local_train    | Path to the pretrained BERT directory.                   |
| `--save_path`     | train / *_train          | Where the checkpoint is saved (adds `_last` suffix).     |
| `--output_path`   | eval                     | Where results are written.                               |
| `--epochs`        | train / *_train          | Number of training epochs.                               |
| `--src_pos` / `--trg_pos` | eval               | Column index of source / target in each line.            |

## Acknowledgment

The PassBERT paper can be found [here](https://www.usenix.org/conference/usenixsecurity23/presentation/xu-ming), and their original TensorFlow PassBERT implementation can be accessed [here](https://github.com/snow0011/PassBertStrengthMeter).

