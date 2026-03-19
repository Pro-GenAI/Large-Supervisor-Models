from pathlib import Path
import torch

# Paths
ROOT = Path(__file__).parent.parent.parent
CHECKPOINT_DIR = ROOT / "checkpoints" / "lsm-transformer"
TRAINED_MODEL_DIR = ROOT / "trained-model" / "lsm-transformer"

# Model
MODEL_NAME = "distilbert-base-uncased"

# Runtime
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = torch.cuda.is_available()

# Data
MAX_LEN = 128

# Training
BATCH_SIZE = 400
GRAD_ACCUM = 1
EPOCHS = 3

# Eval
EVAL_BATCH_SIZE = 600
