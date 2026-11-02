# POLYGUESS LoRA adapter — Scenario #5 (Naz, k ∈ [1,4])

- Base model: Qwen/Qwen2.5-0.5B, revision `060db6499f32faf8b98477b0a26969ef7d8b9987` (loaded from `model/Qwen0.5B`, see repo README).
- PEFT LoRA: r=16, alpha=32, dropout=0.2, targets [q_proj, v_proj, k_proj].
- Size: ~21 MB. This checkpoint is used by `scripts/run_mini_eval.sh`.
