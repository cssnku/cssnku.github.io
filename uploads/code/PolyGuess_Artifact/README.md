# POLYGUESS — NDSS 2027 Artifact

**Paper:** *Targeted Guessing Using Multiple Leaked Passwords* (NDSS 2027, paper #2575)

This artifact received the **Artifact Functional** badge. It contains:

- POLYGUESS training code (Poly-1 loss, LoRA fine-tuning of Qwen2.5-0.5B);
- a dynamic beam-search evaluation pipeline (top-100 guesses per target);
- the fine-tuned LoRA checkpoint for Scenario #5 (Naz, k ∈ [1,4]) and a small de-identified subset of that scenario's train/test data;
- the nine-metric multi-password behavior-analysis and plotting code used for the paper's Section III characterization.

## Repository layout

```text
PolyGuess_AE/
├── main.py                    # entry point (--mode eval / train / distill / wsgen / dsgen)
├── src/                       # model training, evaluation, search, tokenization, prompt building
│   ├── model/                 # train.py, eval.py, data.py, knowledge_distillation.py
│   ├── search/                # search.py, generation.py, monto_carlo.py
│   └── utils/                 # tokenize.py, prompt_template.py, utils.py
├── configs/mini/              # minimal configurations (evaluation + optional training)
├── configs/full/             # full-scale configurations (original paper parameters)
├── data/mini/                 # de-identified mini subsets (Scenario #5, Naz k=1..4)
├── data/full/                # (place full-scale data here; not redistributed)
├── checkpoints/naz_1_4_final/ # fine-tuned LoRA adapter + Qwen tokenizer files (~21 MB total)
├── scripts/                   # download_models.sh, run_mini_eval.sh, run_mini_train.sh, run_metrics.py
├── results/                   # evaluation outputs and metric figures (generated)
└── docs/                      # artifact appendix (2 pages)
```

## Hardware and software requirements

- Linux (x86-64), 8+ cores, 16+ GB RAM.
- **One NVIDIA GPU with ≥16 GB VRAM and bf16 support** (e.g., RTX 3090 / 4090 / A10G) for model inference and optional training. On a single RTX 4090, the bundled mini training and evaluation together complete within 10 minutes. The full paper experiments require substantially longer runtimes.
- Python 3.10+; CUDA-capable PyTorch; transformers, peft, datasets (see requirements.txt).
- Network access is needed once to download the exact base model Qwen/Qwen2.5-0.5B (~1 GB).
- Full-scale reproduction additionally needs the public Naz corpus (not redistributed, see below).

## Setup

Set up an environment on your own GPU machine using the steps below.

```bash
# (optional but recommended) create a dedicated environment with Python 3.10
conda create -n ae_ndss python=3.10 -y
conda activate ae_ndss
```

**Purpose and expected result.** These commands create and activate an isolated Python 3.10
environment. Run the remaining commands in this activated environment from the repository root.

```bash
# 1) CUDA build of PyTorch (adjust cu124 to the CUDA version on the machine)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 2) Python dependencies
pip install -r requirements.txt

# 3) Base model (once)
bash scripts/download_models.sh
sha256sum model/Qwen0.5B/model.safetensors
# expect: 88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342

# 4) Verify GPU
python -c "import torch; print(torch.cuda.is_available())"   # expect True
```

**What these commands do.** The first command installs a CUDA-enabled PyTorch build; the
second installs the project dependencies; the third downloads the exact Qwen base model required
by the provided LoRA adapter; the checksum command verifies the downloaded model file; and the
last command confirms that PyTorch can use the GPU.

**Success criteria.** `sha256sum` must print
`88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`, and the GPU check must
print `True`.

**Setup troubleshooting.** If the model hash differs, or a later evaluation produces empty
`outputPasswords`, remove `model/Qwen0.5B/` and run `bash scripts/download_models.sh` again.
The required base model is Qwen/Qwen2.5-0.5B at revision
`060db6499f32faf8b98477b0a26969ef7d8b9987`, not the Instruct variant. If the GPU check prints
`False`, install a CUDA-enabled PyTorch build and check the NVIDIA driver. A `bfloat16` error
indicates that the GPU does not support bf16; use an Ampere-or-newer GPU. If downloading is slow,
set `HF_ENDPOINT` to an appropriate Hugging Face mirror.

## Quick start — evaluation with the provided checkpoint

```bash
bash scripts/run_mini_eval.sh
```

**Purpose and behavior.** This command runs the supplied Scenario #5 mini test set through the
provided Qwen base model and LoRA checkpoint. Internally, it invokes `main.py` in evaluation mode
with `configs/mini/eval_naz_mini.ini` and generates up to 100 guesses for each of 201 test targets.

Expected outputs in `results/mini_eval/`:

- `input_output.jsonl` — one record per test target: the target password and the top-100 guesses;
- `log.txt` — per-sample success and cumulative success at guess numbers [1, 10, 50, 100];
- `eval_config.json` — the exact configuration used.

Success criteria: the script exits 0, `log.txt` shows cumulative success counts, and `input_output.jsonl` contains one record per test sample.

Example of the first and last lines of `log.txt` (201 lines, one per test sample; the
exact numbers differ depending on the data and GPU):

```text
id:0 max_guess_size:100  guess number:[1, 10, 50, 100]  cracked number:[0, 0, 0, 0] cracked rate:[0.0, 0.0, 0.0, 0.0]
...
id:200 max_guess_size:100  guess number:[1, 10, 50, 100]  cracked number:[6, 12, 15, 16] cracked rate:[0.0299, 0.0597, 0.0746, 0.0796]
```

Example of the first record in `input_output.jsonl` (structure only; the `outputPasswords`
field contains the top-100 guesses with their probabilities):

```json
{"index": 0, "min_cracked_guess_number": 0, "model_input": "As a targeted password guessing model, ...", "real password": "plans2007", "outputPasswords": [["password", 0.007738971859168843], ["Password", 0.0015371896757155699], ["Password11", 0.0012151268097595072], ["...", "..."], ["Love", 1.2605566317706272e-05]]}
```

The key checks are that `input_output.jsonl` contains 201 records (one per test target, each
with a non-empty `outputPasswords` list) and `log.txt` has 201 lines ending with a
`cracked number:[...]` line that reports four cumulative success counts.

**Evaluation troubleshooting.** Run this command from the environment configured above. If
`outputPasswords` is empty, re-download and verify the base model as described in Setup. If a
tokenizer error says that the text input must be a string, ensure that you are using the current
repository version and that `prompt_template_id = 0` is present in
`configs/mini/eval_naz_mini.ini`. For GPU out-of-memory errors, reduce `batch_size` or
`beam_width_list` in that configuration file.

## Optional — train on the mini subset

```bash
bash scripts/run_mini_train.sh
```

**Purpose and behavior.** This command fine-tunes the base model with POLYGUESS's Poly-1 loss on
`data/mini/naz_train_mini.json`, using `configs/mini/train_naz_mini.ini`.

**Expected output and success criterion.** It writes a LoRA checkpoint to
`checkpoints/mini_train_output/` and exits with status 0. On a single RTX 4090, the bundled mini
training and evaluation together complete within 10 minutes.

## Full-scale configurations (original paper parameters)

The artifact ships the original full-scale configurations **for reference only**. The full-scale
data are **not provided** for ethical and legal reasons (consistent with the paper's ethics
statement), so these configurations cannot be executed out of the box; they document the exact
parameters used for the paper's experiments:

- `configs/full/train_naz_full.ini` — original training parameters: 3 epochs, batch size 5, gradient accumulation 64, bf16, LoRA r=16/alpha=32/dropout=0.2, learning rate 5e-4, warmup 0.1, seed 42.
- `configs/full/eval_naz_full.ini` — original evaluation parameters: beam width list `[95,1000]+[1000]*14`, batch size 1000, max 1000 guesses, success logged at [1,10,50,100,500,1000].

If you wish to run them yourself, you must download the public corpora, construct the Scenario #5
train/test data in the same JSON format as `data/mini/`, and place them at:

- `data/full/naz_train_full.json`
- `data/full/naz_test_full.json`

Then run (takes ~1.5 h training + ~9.5 h evaluation per scenario on the recommended GPU):

```bash
python main.py --mode train --config configs/full/train_naz_full.ini
python main.py --mode eval --config configs/full/eval_naz_full.ini
```

The first command trains a LoRA adapter using the supplied full-scale configuration and writes
it to `checkpoints/naz_full/final/`. By default, the second command evaluates the provided
`checkpoints/naz_1_4_final` adapter. To evaluate the newly trained adapter, first change
`lora_path` in `configs/full/eval_naz_full.ini` to `checkpoints/naz_full/final/`.
These configurations require independently prepared data and are provided as a reference;
the repository does not provide a complete raw-corpus-to-experiment data preparation pipeline.

## Behavior metrics and plots (Section III)

```bash
python scripts/run_metrics.py
```

**Purpose and behavior.** This CPU-only command reads `data/mini/naz_mini_users.txt` (one user
per line and password set) and calculates the nine behavior metrics used in Section III.

**Expected output and success criterion.** It exits with status 0 and writes
`results/metrics/Similarity_*.pdf` for MeanSim, MaxSim, VarSim, PivotSim, TemplateCons, LenCV,
MaxLCSCov, MS_PosCopyRate, and MutRatio, grouped by password count.

## Data and privacy notes

- `data/mini/` contains three small demonstration files derived from Naz: 2,000 training
  examples, 201 test examples, and 600 password sets for behavior analysis. They contain password
  strings without email addresses or account identifiers. Passwords remain sensitive; use and
  redistribution must comply with applicable rights, conditions, and ethical obligations.
- JSON format: each record is `{"Knowledge": {"Old password": "pw1 pw2 ..."}, "password": "target"}`.
  The code expects `Old password` as a space-joined string (this is what the original pipeline used);
  the supplied mini samples do not have a target equal to an input password. The prompt builder
  does not filter or deduplicate input passwords; validate independently prepared test data.
- The full COMB / Cit0day / Naz corpora are public but are **not** redistributed, consistent with the
  paper's ethics statement. Full-scale reproduction requires downloading them yourself.
- The artifact received the **Functional** badge. Numbers obtained on the mini subset are illustrative and are
  not claimed to reproduce the paper's full-scale numbers.

## Artifact appendix

See `docs/NDSS27_AE_Appendix_PolyGuess.pdf` for the claim-to-artifact mapping.

