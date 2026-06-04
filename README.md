# mmBERT Tibetan Text Boundary Detector

Fine-tunes [jhu-clsp/mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base) (ModernBERT architecture, 307M params, 8192-token context) for token classification to detect text boundaries in Tibetan Buddhist texts.

## Data

| Item | Count |
|------|-------|
| Documents | 2,117 |
| Unique BDRC works | 560 |
| Annotated boundaries | 82,560 |

Annotations live in `data/Text_boundary_annotation.json`. Each entry has character-offset `breakpoints`, `titles`, and `segments`. Document text files live in `data/documents/{doc_id}.txt`.

## Setup

```bash
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

## Step 1 — Prepare data

Groups documents by BDRC work ID (e.g. `W8LS73547` from `W8LS73547_I8LS73558_...`), deduplicates, and splits at the work level so no volumes of the same collected work leak across train/val/test. Training gets all volumes; val and test get one representative doc per work.

Tokenizes each document into sliding windows (8192 tokens, 512-token stride) and assigns B/O labels based on boundary character offsets.

```bash
python prepare_data.py
```

Quick test with fewer works:

```bash
python prepare_data.py --max-works 20 --max-length 512 --stride 128
```

**Outputs** (in `data/processed/`):

| File | Contents |
|------|----------|
| `dataset/` | HuggingFace DatasetDict with train/validation/test splits |
| `split_info.json` | Work IDs and doc IDs per split |
| `dedupe_report.json` | Work → volume counts, deduplication stats |
| `data_config.json` | Tokenization parameters, label counts |

## Step 2 — Sample benchmark (optional)

Creates a fixed benchmark subset from the test split, stratified by boundary density, for quick and consistent comparisons across experiments.

```bash
python sample_benchmark.py
```

To cap benchmark size:

```bash
python sample_benchmark.py --num-docs 30 --max-windows 500
```

**Output:** `data/benchmark/`

## Step 3 — Train

Fine-tunes mmBERT with weighted cross-entropy loss to handle the extreme class imbalance (~1:700 B:O ratio). Supports MPS (Apple Silicon), CUDA, and CPU.

```bash
python train.py
```

Key flags:

```bash
python train.py \
  --epochs 5 \
  --batch-size 2 \
  --grad-accumulation 8 \
  --lr 2e-5 \
  --pos-weight 10.0 \
  --eval-steps 500 \
  --save-steps 500
```

Resume from a checkpoint:

```bash
python train.py --resume output/checkpoints/checkpoint-1000
```

**Outputs** (in `output/`):

| Path | Contents |
|------|----------|
| `checkpoints/best/` | Best model by validation F1 |
| `checkpoints/final/` | Model after last epoch |
| `checkpoints/checkpoint-{step}/` | Periodic saves |
| `training_log.json` | Per-step metrics |

### Tuning tips

- **Low recall (missing boundaries):** increase `--pos-weight` to 15–20
- **Low precision (too many false positives):** decrease `--pos-weight` to 5
- **Out of memory:** reduce `--batch-size` to 1, increase `--grad-accumulation` proportionally

## Step 4 — Evaluate

Converts token-level predictions back to character positions, then matches against ground truth boundaries with configurable character tolerance.

```bash
python evaluate.py --model output/checkpoints/best
```

Use the benchmark subset:

```bash
python evaluate.py --model output/checkpoints/best --on-benchmark
```

Adjust tolerance:

```bash
python evaluate.py --model output/checkpoints/best --tolerance 20
```

Reports micro/macro precision, recall, F1, plus per-document breakdown. Full report saved to `output/eval_report.json`.

## Step 5 — Predict on new text

Run inference on raw Tibetan text files:

```bash
# Single file
python predict.py input.txt --model output/checkpoints/best

# Save results as JSON
python predict.py input.txt --model output/checkpoints/best --output boundaries.json

# Save annotated text with <b> markers
python predict.py input.txt --model output/checkpoints/best --annotated output_annotated.txt

# Batch process a directory
python predict.py docs_folder/ --model output/checkpoints/best --output results/
```

Adjust confidence threshold (default 0.5):

```bash
python predict.py input.txt --model output/checkpoints/best --threshold 0.3
```

## Project structure

```
├── config.py              # Shared hyperparameters and paths
├── prepare_data.py        # Step 1: data prep with work-level dedup
├── sample_benchmark.py    # Step 2: benchmark sampling
├── train.py               # Step 3: fine-tuning
├── evaluate.py            # Step 4: evaluation
├── predict.py             # Step 5: inference
├── requirements.txt
├── data/
│   ├── Text_boundary_annotation.json
│   ├── documents/         # Raw text files ({doc_id}.txt)
│   ├── processed/         # Tokenized datasets (generated)
│   └── benchmark/         # Benchmark subset (generated)
└── output/
    ├── checkpoints/       # Saved models (generated)
    └── training_log.json  # Training metrics (generated)
```

## Configuration

All defaults are in `config.py`. Override via command-line flags or edit the file directly.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | `jhu-clsp/mmBERT-base` | Base model |
| `MAX_SEQ_LENGTH` | 8192 | Sliding window size in tokens |
| `STRIDE` | 512 | Window overlap |
| `LEARNING_RATE` | 2e-5 | Peak learning rate |
| `NUM_EPOCHS` | 5 | Training epochs |
| `POS_WEIGHT` | 10.0 | Loss weight for B class |
| `TOLERANCE_CHARS` | 15 | Char tolerance for eval matching |
| `SEED` | 42 | Random seed for reproducibility |
