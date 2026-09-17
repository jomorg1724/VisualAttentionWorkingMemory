"""Pinned local frozen-core diagnostic protocol."""

TASK = "motion_duration_cued"
DELAYS = (0, 4, 12, 24)
SPLIT_SEEDS = {"train": 731091, "val": 731092, "test": 731093}
MODEL_SEED = 731101
BOOTSTRAP_SEED = 731102
CHECKPOINT = (
    "WorkingMemory/SpatialTaskBattery/SingleTaskMotion/runs/"
    "motion_20260913_214605/training/checkpoint_012200.pt"
)
CHECKPOINT_SHA256 = (
    "7cec4c48c3da81b4d4935c65098b58974a38a8ff3a825dfe12e4d8d16e9a0e26"
)
CHECKPOINT_STEP = 12200
CHECKPOINT_VERSION = "single_task_motion_continuation_v1"

# Pinned after the smoke profile. Production values may only be reduced before
# production starts if the common 1,800-second absolute deadline requires it.
PRODUCTION_PER_DELAY = {"train": 512, "val": 128, "test": 256}
SMOKE_PER_DELAY = {"train": 8, "val": 4, "test": 4}
LOOKS = (5, 10, 20, 40)
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
CLIP = 1.0
BOOTSTRAP_REPLICATES = 2000
CPU_THREADS = 2

