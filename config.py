"""Shared configuration for the mmBERT boundary detection pipeline."""

from pathlib import Path

# ── Paths ──
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
ANNOTATIONS_FILE = DATA_DIR / "Text_boundary_annotation.json"
PROCESSED_DIR = DATA_DIR / "processed"
BENCHMARK_DIR = DATA_DIR / "benchmark"
OUTPUT_DIR = PROJECT_ROOT / "output"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"

# ── Model ──
MODEL_NAME = "jhu-clsp/mmBERT-base"
MAX_SEQ_LENGTH = 8192
STRIDE = 512  # overlap between sliding windows

# ── Labels ──
LABEL_O = 0
LABEL_B = 1
LABEL_NAMES = ["O", "B"]
NUM_LABELS = len(LABEL_NAMES)

# ── Boundary labeling ──
BOUNDARY_RADIUS = 0  # how many tokens around the boundary char to label as B (0 = exact token only)

# ── Training defaults ──
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 2
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
GRAD_ACCUMULATION_STEPS = 16
MAX_GRAD_NORM = 1.0
POS_WEIGHT = None  # computed dynamically from label counts; set a float to override
NEG_SAMPLE_RATIO = 0.05  # keep 5% of windows that have zero B labels
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = None  # computed dynamically; set a float to override

# ── Data splits ──
TEST_RATIO = 0.10
VAL_RATIO = 0.10

# ── Evaluation ──
TOLERANCE_CHARS = 15  # char-level tolerance for matching predicted vs true boundaries

# ── Seed ──
SEED = 42
