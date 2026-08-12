# Pass2Path: Transition-Path Password Strength Meter

This is a README file for a submodule in the RAID'26 paper *VersaPSE: Versatile
Password Strength Evaluation Using Continual Learning*.

**Pass2Path** models the transformation from a source password to a target
password as a **transition path**: a sequence of edit operations over the
*keyboard sequence* of the source. It learns a Seq2Seq model with stacked
residual LSTMs to predict such paths. In VersaPSE the joint probability of the
ground-truth transition path of a password pair serves as the password
strength.

Note that this is a reproduced version of Pass2Path, not the code provided by
the original paper. The open-source implementation was built under a very old
version of TensorFlow, which isn't widely supported on recent GPU
architectures.

## Pipeline Overview

```
raw (src, trg) password pairs
      │
      ▼
utils.dataset_process ──► (keyboard-seq source, transition-path indices)
      │
      ▼
train.py ──► Seq2Seq (residual LSTM) checkpoint
      │
      ▼
eval.py ──► per-pair transition-path probabilities
```

`utils.dataset_process` converts each pair into the *keyboard key sequences*
of the source and target, aligns them with dynamic programming
(`sequence_functions.find_med_backtrace`) and serializes the alignment into a
transition-path index sequence (`sequence_functions.pair2path`). Each path is
prepended with `<BOS>` (0), appended with `<EOS>` (1), and padded to a fixed
length of 61 with `<PAD>` (2).

## File Overview

| File                      | Purpose                                                                                                                              |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `train.py`                | Training entry point. Trains a local (site-specific) model, or fine-tunes a pretrained checkpoint into a "local_to_global" transfer model (freezes the decoder when `--load_path` is given). |
| `eval.py`                 | Evaluation entry point. Loads a checkpoint, computes the joint probability of the ground-truth transition path for every test password pair, and writes one probability per line. |
| `model.py`                | Model definition: `ResidualLSTMEncoder`, `ResidualLSTMDecoder`, `Seq2Seq`. Contains `prob_extract` (probability of a given path) and `freeze_layers` (for transfer learning). |
| `sequence_functions.py`   | Edit-distance utilities over keyboard sequences: `pair2path` (password pair -> transition path), `path2word` / `path2word_kb` (path -> transformed password), transition-dictionary builders, and the global `trans_dict_2idx.json` / `trans_dict_2path.json` loading. |
| `utils.py`                | Dataset construction: character/keyboard tokenizer (`GetStringTokenizer`), `CustomDataset`, `my_collate`, and the multiprocessing-based `dataset_process` (plus a single-threaded fallback). |
| `trans_dict_2idx.json`    | Maps a human-readable transition `('op', char, pos)` to an index (indices start at 3; 0/1/2 are `<PAD>`/`<BOS>`/`<EOS>`).             |
| `trans_dict_2path.json`   | Inverse mapping from index back to a human-readable transition.                                                                     |

## Data Format

All dataset files are **tab-separated** password pairs, one pair per line:

```
<source_password>\t<target_password>
```

Column positions are given by `--src_pos` / `--trg_pos` (default `0` / `1`).

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

Run every command from **this folder** (`open_source/Pass2Path`).

### 1. Train the local (site-specific) model

```bash
python ./train.py \
    --train_file ../dataset/train/pseudo_train_data-sister.txt \
    --save_path ./ckpts/ --save_name local.pth \
    --src_pos 0 --trg_pos 1 --epochs 5
```

### 2. Transfer to the global model (local_to_global)

The local model is fine-tuned on a large public dataset. When `--load_path` is
given, the model is initialized from the checkpoint and `freeze_layers()` is
called, so **the decoder is frozen** and only the encoder is updated:

```bash
python ./train.py \
    --train_file ../dataset/train/pseudo_train_data-popular.txt \
    --save_path ./ckpts/ --save_name local_to_global.pth \
    --src_pos 0 --trg_pos 1 --load_path ./ckpts/local.pth --epochs 3
```

> Pretrained `local_to_global.pth` checkpoints can be placed under `./ckpts/`
> so that Stages 1–2 can be skipped for evaluation-only reproduction.

### Arguments

| Argument                | Default                                | Description                                                              |
| ----------------------- | -------------------------------------- | ------------------------------------------------------------------------ |
| `--train_file`          | `../dataset/train/pseudo_train_data-sister.txt` | Path to the training password-pair file.                    |
| `--save_path`           | `./`                                   | Directory where the checkpoint is saved.                                 |
| `--save_name`           | `local.pth`                            | Checkpoint file name (appended to `--save_path`).                        |
| `--load_path`           | `None`                                 | Optional pre-trained checkpoint to initialize from (freezes the decoder).|
| `--epochs`              | `50`                                   | Number of training epochs.                                               |
| `--src_pos` / `--trg_pos` | `0` / `1`                           | Column index of the source / target in each line.                        |

## How to evaluate (`eval.py`)

Each `eval.py` call computes the joint probability of the ground-truth
transition path for every pair and writes one probability per line:

```bash
python ./eval.py \
    --test_file ../dataset/test/pseudo_test_data-targeted.txt \
    --load_ckpt ./ckpts/local_to_global.pth \
    --output_path ./result/targeted_res.txt \
    --src_pos 0 --trg_pos 1
```

For **untargeted** evaluation, use
`../dataset/test/pseudo_test_data-untargeted.txt`.

### Arguments

| Argument                | Default                                | Description                                                              |
| ----------------------- | -------------------------------------- | ------------------------------------------------------------------------ |
| `--test_file`           | `../dataset/test/pseudo_test_data-targeted.txt` | Path to the test password-pair file.                            |
| `--load_ckpt`           | `./local.pth`                          | Path to the trained checkpoint.                                         |
| `--output_path`         | `./test.txt`                           | File to save the computed probabilities.                                 |
| `--src_pos` / `--trg_pos` | `0` / `1`                           | Column index of the source / target in each line.                        |

## Notes

- Special tokens used in the path sequence: `<BOS>` = 0, `<EOS>` = 1,
  `<PAD>` = 2; transition-dictionary indices start at 3.
- The encoder vocabulary covers 98 characters + 3 special tokens; the decoder
  output space is the 12000-class transition dictionary.
- The maximum path length is fixed to 61.

## Environment

- PyTorch (CUDA recommended)
- `numpy`, `tqdm`
- `word2keypress` (keyboard key-sequence conversion)

## Acknowledgment

The stacked-residual-LSTM Seq2Seq architecture is based on
[matejklemen/stacked-residual-lstm](https://github.com/matejklemen/stacked-residual-lstm),
and the edit-distance backtrace is adapted from
[credtweak](https://github.com/Bijeeta/credtweak) (see the header comments in
`model.py` / `sequence_functions.py`).
