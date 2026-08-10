"""Shared configuration for the mmBERT boundary detection pipeline.

All values are module-level constants used as defaults throughout the
package. Override individual settings via command-line flags on each
subcommand — this file is the single source of truth for defaults.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
# PROJECT_ROOT resolves to the repository root regardless of how the package
# is installed (editable or otherwise).
PROJECT_ROOT = Path(__file__).parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
ANNOTATED_DATA_DIR = DATA_DIR / "Annotated_jsons"
PROCESSED_DIR = DATA_DIR / "processed"
BENCHMARK_DIR = DATA_DIR / "benchmark"
OUTPUT_DIR = PROJECT_ROOT / "output"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"

# ── Parallelism ─────────────────────────────────────────────────────────────
NUM_WORKERS = 0  # bounded to avoid OOM with large tokenizer subprocesses

# ── Model ───────────────────────────────────────────────────────────────────
MODEL_NAME = "jhu-clsp/mmBERT-base"
MAX_SEQ_LENGTH = 8192
STRIDE = 256  # overlap between sliding windows

# ── Labels ──────────────────────────────────────────────────────────────────
LABEL_O = 0
LABEL_B = 1
LABEL_NAMES = ["O", "B"]
NUM_LABELS = len(LABEL_NAMES)

# ── Boundary labeling ────────────────────────────────────────────────────────
BOUNDARY_RADIUS = 3  # chars around the boundary char to label as B (0 = exact token only)

# ── Training defaults ────────────────────────────────────────────────────────
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 2
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
GRAD_ACCUMULATION_STEPS = 16
MAX_GRAD_NORM = 1.0
POS_WEIGHT = None  # computed dynamically from label counts; set a float to override
NEG_SAMPLE_RATIO = 0.20  # keep 20 % of windows that have zero B labels
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 1.5
FOCAL_ALPHA = 0.90

# ── Data splits ───────────────────────────────────────────────────────────────
TEST_RATIO = 0.10
VAL_RATIO = 0.10

# ── Evaluation ────────────────────────────────────────────────────────────────
TOLERANCE_CHARS = 25  # char-level tolerance for matching predicted vs true boundaries

# ── Seed ──────────────────────────────────────────────────────────────────────
SEED = 42

# ── Weights & Biases ──────────────────────────────────────────────────────────
WANDB_PROJECT = "Bo-boundary-mmbert"
