# Experiment Report — mmBERT Tibetan Boundary Detection

Brief summary of the prepared corpus in `data/processed/` and the training run logged in `output/training_log.json` (run started 2026-08-06).

## Goal

Fine-tune [jhu-clsp/mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base) (ModernBERT, token classification) to detect outline / segment-boundary tokens (`B`) vs non-boundary (`O`) in Tibetan Buddhist volume text.

## Dataset (`data/processed/`)

| Item | Value |
|------|------:|
| Source volumes | 9,657 annotated JSONs |
| Split policy | Work-level 80/10/10 (seed 42); no work ID leakage |
| Train / val / test works | 3,447 / 430 / 430 |
| Train / val / test docs | 7,978 / 430 / 430 |
| Window size / stride | 8,192 / 256 tokens |
| Boundary radius | ±3 characters |
| Train windows (after downsample) | 228,353 |
| Val / test windows | 16,209 / 15,968 |
| Train neg-sample ratio | 0.5 (50% of O-only windows kept) |
| Train B:O ratio (labeled tokens) | ~1 : 1,724 |

Val/test keep all windows (no negative downsample). The task remains extremely imbalanced (~0.03–0.06% `B` tokens).

## Training setup

| Setting | Value |
|---------|------:|
| Hardware | 1× NVIDIA A100-SXM4-40GB |
| Precision | bf16 AMP |
| Epochs planned | 5 |
| Batch / grad accum / effective | 4 / 4 / 16 |
| LR / warmup | 2e-5 / 7,136 steps |
| Steps per epoch | 14,272 |
| Loss | Focal (γ=2.5, α_B=0.90, α_O=0.10) |
| Selection metric | Validation F2 |
| Early stopping | Patience 10 evals (eval every 750 steps) |
| Tracking | W&B `Bo-boundary-mmbert` / `sandy-gorge-1` |

## Results

Training stopped early at **step 19,500** (epoch 2, ~37% through epoch 2) after **47h 32m**. No F2 gain for 10 consecutive evals.

| Checkpoint | Step | P | R | F1 | F2 |
|------------|-----:|--:|--:|---:|---:|
| **Best F2** (saved) | 12,000 | 0.492 | 0.885 | 0.632 | **0.763** |
| Best F1 | 12,750 | 0.653 | 0.783 | **0.712** | 0.753 |
| Last eval | 19,500 | 0.466 | 0.898 | 0.613 | 0.757 |

- Train loss fell from ~0.0038 → ~5e-5; val loss stayed ~3–7e-5 after mid–epoch 1.
- Best model: `output/checkpoints/best` (step 12,000). Final weights also saved at stop: `output/checkpoints/final`.
- Epoch 2 did not beat the epoch-1 F2 peak (max epoch-2 F2 ≈ 0.762).

## Takeaways

1. **Recall-heavy operating point.** Best F2 favors high recall (0.885) with moderate precision (0.492); FP counts remain large relative to TP.
2. **Precision–recall tradeoff is unstable across evals.** Several steps show very high recall (≥0.92) with precision collapsing below 0.35 (e.g. steps 6750, 8250, 17250).
3. **Most gains in epoch 1.** F2 rose from 0.20 → 0.76 by step 12,000; further training mainly traded precision/recall without a new F2 high.
4. **Imbalance still dominates.** Even with negative-window downsampling and focal loss, validation FP >> TP at the best F2 checkpoint (~34k TP vs ~34k FP — near 1:1 at that point, but many earlier/later evals are much worse).

## Artifacts

| Path | Role |
|------|------|
| `data/processed/dataset/` | Train/val/test windows |
| `data/processed/data_config.json` | Prepare-run counts and params |
| `data/processed/split_info.json` | Work/doc IDs per split |
| `output/training_log.json` | Per-eval metrics |
| `output/checkpoints/best` | Best validation F2 model |
