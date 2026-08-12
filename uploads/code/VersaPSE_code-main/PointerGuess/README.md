# Using PointerGuess to Construct Password Strength Meters

This is a README file for a submodule in the RAID'26 paper *VersaPSE: Versatile Password Strength Evaluation Using Continual Learning*.

PointerGuess is a sequence-to-sequence pointer-generator network that models
the transformation from one password (source) to another (target). In this
work it is used to build a **password strength meter**: for a given source
password, the model assigns a probability to the target password, which serves
as the password strength.

Note that the original authors of PointerGuess did not open-source their code, hence the PointerGuess I provide here is only for reference, and does not guarantee to reproduce the password guessing results for the original PointerGuess paper.

## File Overview

| File                   | Purpose                                                                                                            |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `config.py`            | Global configuration: compute device (`cuda`) and maximum password length (`MAX_LEN = 30`).                       |
| `model.py`             | The PointerGeneratorNetworks model (bidirectional-LSTM encoder + pointer-generator decoder with attention).         |
| `reuse_utils.py`       | Vocabulary construction (ASCII charset) and conversion of raw password pairs into model-ready tensors / dataset.    |
| `pointerguess_utils.py`| Data loading, preprocessing, training loop, risk evaluation, and probability extraction.                           |
| `reuse_train.py`       | Training script. Trains a model on password pairs, optionally continuing from a pre-trained checkpoint.            |
| `eval.py`              | Evaluation script. Loads a trained checkpoint and outputs the probability of each password pair in a test file.    |

## Data Format

All training / test files are **tab-separated** password pairs, one pair per
line, e.g.:

```
<source_password>\t<target_password>
```


## How to train (`reuse_train.py`)

Train a model on password pairs (source-domain "local" model):

```bash
python reuse_train.py \
    --reuse_tp ../dataset/train/pseudo_train_data-sister.txt \
    --reuse_sp ./ckpts/ \
    --reuse_sp_name local.pth \
    --epochs 5 \
    --src_pos 0 \
    --trg_pos 1
```

Continue training from a pre-trained checkpoint (e.g., adapt a global model to
a local domain). With `--load_ckpt_path`, the decoder and reduce layers are
**frozen** so only the encoder is updated:

```bash
python reuse_train.py \
    --reuse_tp ../dataset/train/pseudo_train_data-sister.txt \
    --reuse_sp ./ckpts/ \
    --reuse_sp_name global_to_local.pth \
    --epochs 5 \
    --src_pos 0 \
    --trg_pos 1 \
    --load_ckpt_path ./ckpts/global.pth
```

### Arguments

| Argument         | Default                         | Description                                            |
| ---------------- | ------------------------------- | ------------------------------------------------------ |
| `--reuse_tp`     | `../dataset/train/pseudo_train_data-sister.txt` | Path to the training password-pair file.  |
| `--reuse_sp`     | `./`                          | Directory where the checkpoint is saved.              |
| `--reuse_sp_name`| `local.pth`                     | Checkpoint file name.                                 |
| `--epochs`       | `5`                             | Number of training epochs.                            |
| `--load_ckpt_path`| `None`                          | Optional pre-trained checkpoint to initialize from (freezes decoder/reduce). |
| `--src_pos`      | `0`                             | Column index of the source password in each line.     |
| `--trg_pos`      | `1`                             | Column index of the target password in each line.     |

## How to evaluate (`eval.py`)

Load a trained checkpoint and compute the probability of every password pair
in a test file. Results are written one probability per line:

```bash
python eval.py \
    --test_file ../dataset/test/pseudo_test_data-targeted.txt \
    --load_ckpt ./ckpts/local.pth \
    --output_path ./result/pointerguess_res.txt \
    --src_pos 0 \
    --trg_pos 1
```

### Arguments

| Argument         | Default                        | Description                                            |
| ---------------- | ------------------------------ | ------------------------------------------------------ |
| `--test_file`    | `../dataset/test/pseudo_test_data-targeted.txt` | Path to the test password-pair file. |
| `--load_ckpt`    | `./local.pth`                 | Path to the trained checkpoint.                       |
| `--output_path`  | `./test.txt`                  | File to save the computed probabilities. |
| `--src_pos`      | `0`                            | Column index of the source password in each line.     |
| `--trg_pos`      | `1`                            | Column index of the target password in each line.     |

## Acknowledgment

The PointerGuess model is based on [this](https://github.com/hquzhuguofeng/New-Pointer-Generator-Networks-for-Summarization-Chinese) repository.


The original PointerGuess paper (USENIX Security 2024) can be found [here](cssnku.github.io/uploads/publications/usenix24-pointerguess-full-v17.pdf).

