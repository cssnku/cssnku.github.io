# KNNGuess: k-Nearest-Neighbor Targeted Password Guessing

This is a README file for a submodule in the RAID'26 paper *VersaPSE: Versatile
Password Strength Evaluation Using Continual Learning*.

KNNGuess / KNN-TPG is a targeted password guessing model: given an old
(source) password, it predicts the new (target) password using a Transformer
sequence model whose output distribution is interpolated with a
non-parametric **k-nearest-neighbor datastore** built over the hidden states of
the decoder. In VersaPSE it is used to build a **password strength meter**: for
a given source password, the model assigns a probability to the target
password, which serves as the password strength.

This folder is adapted from the official
[KNNGuess/KNNGuess-code](https://github.com/KNNGuess/KNNGuess-code)
repository. The original `README.md` is kept as `README_original.md`. The model
and the `knn/` subpackage are unchanged; most scripts were modified to fit the
VersaPSE workflow (see [Differences from the original](#differences-from-the-original)).

## Pipeline Overview

```
train (main.py) ──► Transformer checkpoint (./experiment/...)
      │
      ▼
gen_datastore.py ──► knn datastore (keys / vals + FAISS index, ./datastore/...)
      │
      ▼
eval.py ──► per-pair strength probabilities (NMT prob × λ + knn prob × (1-λ))
```

## File Overview

| File                | Purpose                                                                                                            |
| ------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `config.py`         | Global configuration: Transformer hyper-parameters, training / knn / testing settings, device. Modified for the VersaPSE data layout. |
| `model.py`          | The Transformer model (6 layers, `d_model=512`, 8 heads) with a `Generator` (log-softmax over the 54-token keypress vocab). Added `freeze_layers()`. |
| `PW2SEQ.py`         | Keypress-sequence tokenizer: maps keyboard tokens to ids (54-token vocab, `<UNK>=0`, `<PAD>=1`, `<BOS>=2`, `<EOS>=3`). |
| `data_loader.py`    | `MTDataset` / `Batch` data loading with source / target masks; filters pairs by keypress cosine similarity during training. |
| `utils.py`          | Helpers: `similar` (2-gram cosine similarity), `get_segment`, `check`, `get_mod_rate`, `get_popular_pws`, `set_logger`. |
| `train.py`          | Training loop (modified to accept `epochs` / `save_path` / `freeze_layers`).                                       |
| `main.py`           | Training entry point (rewritten with argparse + transfer-learning freeze).                                         |
| `gen_datastore.py`  | Builds the knn datastore from a trained model (modified: can mix two training files).                              |
| `lz_delete.py`      | `beam_decode2`: beam search with knn + local-knn probability mixing (guess generation).                            |
| `beam_decoder.py`   | Beam-search decoder (kept from an earlier version of the code; `beam_search` is no longer called by the active scripts). |
| `eval.py`           | **New.** Evaluation entry point: computes the per-pair strength probability (with or without the knn datastore).   |
| `test.py`           | **New.** Guessing-attack evaluation: ranks the target among beam-search guesses and writes the rank.               |
| `guess_one.py`      | Demo: guesses the target for a single source password.                                                             |
| `psm.py`            | Demo: per-character strength of a single (source, target) pair (KNN-PSM).                                         |
| `run.py`            | Batch driver that orchestrates all VersaPSE experiments (training variants, datastore generation, evaluation).     |
| `knn/`              | The knn-mt style `Datastore` / `Retriever` / `Combiner` subpackage (from the original repo).                       |
| `experiment/`, `datastore/`, `guesses/` | Model checkpoints, datastores (to be generated), and sample guessing outputs.                  |

## Data Format

All training / test files are **tab-separated** password pairs, one pair per
line:

```
<source_password>\t<target_password>
```

`config.dataset_have_email` is `False` in this adaptation, so the source is
column 0 and the target is column 1. (The original repo's Tianya/Dodonew files
are `email\tsrc\ttrg`, for which `dataset_have_email=True` selects columns
1 / 2.)

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

## How to train (`main.py`)

Run every command from **this folder** (`open_source/KNNGuess`).

### 1. Train the local (site-specific) model

```bash
python main.py \
    --train_file ../dataset/train/pseudo_train_data-sister.txt \
    --save_path ./experiment/model.pth \
    --epochs 1 --gpu_id 0
```

### 2. Transfer to the global model (local_to_global)

When `--load_ckpt` is given, `model.freeze_layers()` is called before training,
so the decoder, target embedding and generator are frozen and only the encoder
is fine-tuned on the global corpus:

```bash
python main.py \
    --train_file ../dataset/train/pseudo_train_data-popular.txt \
    --save_path ./experiment/local_to_global.pth \
    --epochs 1 --gpu_id 0 \
    --load_ckpt ./experiment/model.pth
```

### Arguments

| Argument        | Default                                       | Description                                                                 |
| --------------- | --------------------------------------------- | --------------------------------------------------------------------------- |
| `--train_file`  | `../dataset/train/pseudo_train_data-sister.txt` | Path to the training password-pair file.                                    |
| `--save_path`   | `./experiment/model.pth`                       | Path where the checkpoint is saved.                                         |
| `--epochs`      | `1`                                           | Number of training epochs.                                                  |
| `--load_ckpt`   | `None`                                        | Optional pre-trained checkpoint; freezes decoder / target embedding / generator. |
| `--gpu_id`      | `0`                                           | CUDA device id (sets `CUDA_VISIBLE_DEVICES`).                               |

## How to build the datastore (`gen_datastore.py`)

`gen_datastore.py` first expands the training pairs with **segment-similar**
sub-pairs (pairs of same-type segments whose keypress 2-gram cosine similarity
is > 0.4) into a `*_knn.txt` file, then runs the trained model over it and
stores the decoder hidden states (keys) and their target tokens (vals),
finally building a FAISS index. `--train_data_path_2` mixes a second file into
the datastore (used for the local_to_global datastore):

```bash
python gen_datastore.py \
    --train_data_path ../dataset/train/pseudo_train_data-sister.txt \
    --train_data_path_2 ../dataset/train/pseudo_train_data-popular.txt \
    --knn_train_data_path ./datastore/sister_knn.txt \
    --model_path ./experiment/local_to_global.pth \
    --knn_datastore_path ./datastore/local_to_global \
    --gpu_id 0
```

## How to evaluate (`eval.py`)

`eval.py` computes the probability of the ground-truth target for every pair in
a test file and writes one probability per line. When `--datastore_path` is
given, the output distribution at each step is the interpolation
`model_prob × λ + knn_prob × (1 - λ)` (`λ = config.lambda_`); otherwise the
pure NMT probability is used:

```bash
python eval.py \
    --test_file ../dataset/test/pseudo_test_data-targeted.txt \
    --load_ckpt ./experiment/local_to_global.pth \
    --output_path ./result/knnguess_res.txt \
    --datastore_path ./datastore/local_to_global
```

### Arguments

| Argument           | Default                                    | Description                                                                 |
| ------------------ | ------------------------------------------ | --------------------------------------------------------------------------- |
| `--test_file`      | `../dataset/test/pseudo_test_data-targeted.txt` | Path to the test password-pair file.                                    |
| `--load_ckpt`      | `./experiment/local_to_global.pth`            | Path to the trained checkpoint.                                             |
| `--datastore_path` | `None`                                     | Optional knn datastore folder; when set, probabilities are knn-mixed.       |
| `--output_path`    | `./result/knnguess_res.txt`                 | File to save the computed probabilities.                                    |

## Guessing attack (`test.py`)

`test.py` runs `beam_decode2` (beam search with knn + local-knn mixing,
keeping only guesses that contain both letters and digits) and reports, for
each test pair, the rank at which the target appears among the top-
`config.TopK` guesses (`guess_ans.txt`), plus the raw no-mixing rank
(`guess_ans_nomix.txt`):

```bash
python test.py \
    --test_data_path ../dataset/test/pseudo_test_data-targeted.txt \
    --guess_ans_path ./guess_ans.txt \
    --guess_ans_path_nomix ./guess_ans_nomix.txt \
    --model_path ./experiment/model.pth \
    --knn_datastore_path ./datastore/local \
    --gpu_id 0
```

## Interactive demos

Guess the target for a single source password:

```bash
python guess_one.py --source_password Orange0301
```

Evaluate the per-character strength of a single (old, new) password pair
(KNN-PSM):

```bash
python psm.py --source_password YOUR_OLD_PASSWORD --target_password YOUR_NEW_PASSWORD
```

The output is the probability of each character of the new password being
predicted by KNNGuess; the greater the probability, the less secure the
character.

## Differences from the original

Compared with the official
[KNNGuess-code](https://github.com/KNNGuess/KNNGuess-code) repository:

| Item               | Original                                                                    | This adaptation                                                                 |
| ------------------ | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `eval.py`          | Not present                                                                 | **Added.** Computes per-pair strength probabilities (PSM evaluation), with optional knn mixing. |
| `test.py`          | Not shipped (only referenced by `run.py`)                                   | **Added.** Working guessing-attack evaluation that writes guess ranks.           |
| `main.py`          | Reads `config.py` directly, no CLI, no freezing                             | Rewritten with argparse (`--train_file`, `--save_path`, `--epochs`, `--load_ckpt`, `--gpu_id`); fine-tunes with frozen decoder / target embedding / generator when `--load_ckpt` is given. |
| `train.py`         | Fixed `config.epoch_num` epochs, saves to `config.model_path`               | Accepts `epochs` / `save_path` / `freeze_layers` parameters.                    |
| `model.py`         | —                                                                           | Added `Transformer.freeze_layers()` (freezes decoder, target embedding, generator). |
| `gen_datastore.py` | Uses `config` directly, one train file                                      | Added argparse and a second training file (`--train_data_path_2`) to mix local + global data into the datastore. |
| `config.py`        | `dataset_have_email=True`, `train_batch_size=256`, `train_data_path='./tianya_dodonew_train.txt'`, `model_path='./experiment/model.pth'` | `dataset_have_email=False`, `train_batch_size=1024`, VersaPSE data paths under `../dataset/`, model path under `./experiment/`, plus `write_ans` / guess-path settings. |
| `data_loader.py`   | `MTDataset(data_path, mode='train')` uses `config.sort_dataset`             | Accepts a `sort` override.                                                      |
| `lz_delete.py`     | The `check(y)` filter is commented out (no letter+digit filter on guesses)  | `check(y)` is enabled, so guesses must contain both letters and digits.         |
| `run.py`           | Simple 3-step script (`main` → `gen_datastore` → `test`)                    | Expanded into a batch driver for all VersaPSE experiments (local / global / global_to_local / nofreeze training, datastore generation, evaluation). |
| `exp_ans/`         | Included (sample experiment outputs)                                        | Not included.                                                                    |
| `datastore/`       | Contains pre-built datastores                                               | Empty; datastores must be generated with `gen_datastore.py`.                    |

`README_original.md` in this folder is the original README from the KNNGuess
repository.

## Environment

- Python 3.7
- torch (CUDA recommended), torchvision
- `word2keypress`
- `faiss-gpu` (used by the knn retriever)

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c pytorch -c nvidia faiss-gpu=1.7.3
```

## Acknowledgment

This repository is adapted from
[KNNGuess/KNNGuess-code](https://github.com/KNNGuess/KNNGuess-code), the
official implementation of *Targeted Password Guessing Using k-Nearest
Neighbors*.
