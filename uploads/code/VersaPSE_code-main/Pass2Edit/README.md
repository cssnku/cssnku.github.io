# Pass2Edit: Keypress-Level Password Transformation Model

This is a README file for a submodule in the RAID'26 paper *VersaPSE: Versatile
Password Strength Evaluation Using Continual Learning*.

Pass2Edit models how a user edits one password into another at the
**keypress level**: given a source password and a target password (both
converted into keyboard key sequences), the model predicts, for each pair, the
sequence of atomic edit operations (insert / delete / replace) that transforms
the source into the target. In VersaPSE it is used to build **Password
Strength Meters (PSM)** for both targeted (victim-specific) and untargeted
password strength evaluation: the probability of the target password's edit
path serves as the password strength.

Note that this is a PyTorch reproduction of Pass2Edit, not the code provided by the original paper.

## Pipeline Overview

```
raw (src, trg) password pairs
      │
      ▼
utils.preprocess ──► per-step samples (source, target, edit op, length)
      │
      ▼
train.py ──► SimpleP2E checkpoint + edit_tokenizer.pkl
      │
      ▼
eval.py ──► per-pair strength probabilities
```

`utils.preprocess` converts each pair into the *keyboard key sequences* of the
source and target, aligns them with dynamic programming
(`sequence_functions.transformation`) and expands the alignment into a list of
**atomic edit steps** (`sequence_functions.pass2edit`). Each atomic step is a
sample whose label is the edit operation applied at that step. The probability
of a whole password pair is the product of the probabilities of all its atomic
steps.

## File Overview

| File                    | Purpose                                                                                                                |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `hyper_param.py`        | Global hyper-parameters (device, embedding size, GRU layers, tokenizer sizes, `type_num` / `ops_oov_token`, `MAX_LEN`). |
| `model.py`              | `SimpleP2E`, a GRU model that takes a source/target keypress pair and predicts the edit operation per position. Contains `freeze_layers()`. |
| `tokenizer.py`          | `EditTokenizer` maps edit operations (e.g. `<1,2,a>`) to integer ids; `TwoGramTokenizer` builds bag-of-2-grams vectors (used by `match_finder.py`). |
| `sequence_functions.py` | Core edit-path utilities: DP alignment (`transformation`), serialization (`trans2seq` / `seq2trans`), step-by-step application (`pass2edit`), and single-operation helpers. |
| `utils.py`              | Password tokenization (`pw2token`), `CustomDataset` / `my_collate`, and `preprocess` (multi-process conversion of password pairs into per-step samples). |
| `train.py`              | Training entry point. Trains (or fine-tunes) the model and saves the checkpoint and the edit tokenizer.                |
| `eval.py`               | Evaluation entry point. Loads a checkpoint, computes per-pair probabilities, and writes them one per line.             |
| `match_finder.py`       | Finds similar (base, target) password pairs using 2-gram cosine similarity (script-style).                             |
| `export_tokenizers.py`  | Exports the edit tokenizer / keyboard charset / model config as JSON for the browser demo.                             |
| `onnx_convert.py`       | Exports a checkpoint to ONNX for in-browser (WebAssembly) inference.                                                    |

## Data Format

All training / test files are **tab-separated** password pairs, one pair per
line:

```
<source_password>\t<target_password>
```

The column index of the source / target is selected with `--src_pos` /
`--trg_pos` (default `0` / `1`).

- Training pairs:
  - `../dataset/train/pseudo_train_data-sister.txt` — sister password pairs
    (`src \t trg`)
  - `../dataset/train/pseudo_train_data-popular.txt` — popular-password pairs
    (`src \t trg \t count`)
- Test pairs:
  - `../dataset/test/pseudo_test_data-targeted.txt` — targeted pairs
    (`src \t trg \t pop`)
  - `../dataset/test/pseudo_test_data-untargeted.txt` — untargeted pairs
    (`src \t trg \t count`)

## How to train (`train.py`)

Run every command from **this folder** (`open_source/Pass2Edit`).

### 1. Train a model from scratch

```bash
python ./train.py \
    --train_file ../dataset/train/pseudo_train_data-sister.txt \
    --save_path ./ckpts/ --save_name local.pth \
    --src_pos 0 --trg_pos 1 --epochs 3
```

The first run also trains and saves the edit tokenizer
(`./ckpts/edit_tokenizer.pkl`).

### 2. Fine-tune on a general dataset

```bash
python ./train.py \
    --train_file ../dataset/train/pseudo_train_data-popular.txt \
    --save_path ./ckpts/ --save_name global.pth \
    --src_pos 0 --trg_pos 1 --epochs 1 \
    --load_tokenizer ./ckpts/edit_tokenizer.pkl
```

### 3. Train the "global-to-local" model (general → local fine-tuning)

```bash
python ./train.py \
    --train_file ../dataset/train/pseudo_train_data-sister.txt \
    --save_path ./ckpts/ --save_name global_to_local.pth \
    --src_pos 0 --trg_pos 1 --epochs 3 \
    --load_tokenizer ./ckpts/edit_tokenizer.pkl \
    --load_ckpt ./ckpts/global.pth
```

### 4. Train the "nofreeze" model (fine-tuning without freezing)

```bash
python ./train.py \
    --train_file ../dataset/train/pseudo_train_data-popular.txt \
    --save_path ./ckpts/ --save_name nofreeze.pth \
    --src_pos 0 --trg_pos 1 --epochs 1 \
    --load_tokenizer ./ckpts/edit_tokenizer.pkl \
    --load_ckpt ./ckpts/local.pth
```

### Arguments

| Argument           | Default                                     | Description                                                          |
| ------------------ | ------------------------------------------- | -------------------------------------------------------------------- |
| `--train_file`     | `../dataset/train/pseudo_train_data-sister.txt` | Path to the training password-pair file.                         |
| `--save_path`      | `./`                                        | Directory where the checkpoint is saved.                             |
| `--save_name`      | `local.pth`                                 | Checkpoint file name (appended to `--save_path`).                    |
| `--epochs`         | `5`                                         | Number of training epochs.                                           |
| `--load_ckpt`      | `None`                                      | Optional pre-trained checkpoint to continue from (fine-tuning).      |
| `--load_tokenizer` | `None`                                      | Optional pre-trained edit tokenizer to load instead of training one. |
| `--src_pos` / `--trg_pos` | `0` / `1`                           | Column index of the source / target in each line.                    |

> `model.freeze_layers()` (in `model.py`) freezes the classifier head
> (`fc1` / `fc2`) and the first GRU layer. It is provided for partial transfer
> learning but is **not** invoked by `train.py` by default — uncomment it in
> `train.py` if you want the frozen-layers behavior.

## How to evaluate (`eval.py`)

Load a checkpoint, compute the joint probability of the ground-truth edit path
for every pair in a test file, and write one probability per line:

```bash
python ./eval.py \
    --test_file ../dataset/test/pseudo_test_data-targeted.txt \
    --load_ckpt ./ckpts/local.pth \
    --output_path ./result/targeted_res.txt \
    --src_pos 0 --trg_pos 1 \
    --load_tokenizer ./ckpts/edit_tokenizer.pkl
```

### Arguments

| Argument           | Default                                     | Description                                                          |
| ------------------ | ------------------------------------------- | -------------------------------------------------------------------- |
| `--test_file`      | `../dataset/test/pseudo_test_data-targeted.txt` | Path to the test password-pair file.                            |
| `--load_ckpt`      | `./local.pth`                               | Path to the trained checkpoint.                                      |
| `--load_tokenizer` | `./edit_tokenizer.pkl`                      | Path to the saved edit tokenizer.                                    |
| `--output_path`    | `./test.txt`                                | File to save the computed probabilities.                             |
| `--src_pos` / `--trg_pos` | `0` / `1`                           | Column index of the source / target in each line.                    |

## Environment

- PyTorch (GPU recommended; set `device` in `hyper_param.py`)
- `word2keypress` (keyboard key-sequence conversion)
- `matplotlib` (loss plots in `train.py`)
- `tqdm`

## Acknowledgment


