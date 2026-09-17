"""Fixed scratch protocol for the canonical cued motion-duration task."""
import copy

import numpy as np

from WorkingMemory.AttentionContextComparator.protocol import (
    SPATIAL_TEST_SEED,
    SPATIAL_VAL_SEED,
    recipe as suite_recipe,
)

TASK = "motion_duration_cued"
DELAYS = (0, 4, 12, 24)
UPDATES = 4000
BATCH_SIZE = 8
TOTAL_EPISODES = UPDATES * BATCH_SIZE
VALIDATION_TARGETS = (800, 1600, 2400, 3200, 4000)
VALIDATION_N = 64
TEST_N = 256
RUNTIME_SECONDS = 14400


def recipe():
    """Return the unmodified five-task scratch config used to build the model."""
    cfg = copy.deepcopy(suite_recipe())
    if (
        cfg["updates"] != UPDATES
        or cfg["batch_size"] != BATCH_SIZE
        or cfg["task_classes"][TASK] != 4
        or tuple(cfg["train_names"][TASK])
        != tuple(f"{TASK}_D{delay}" for delay in DELAYS)
    ):
        raise RuntimeError("Canonical attention-context recipe changed")
    return cfg


def training_name(cfg, step):
    """Match the suite's task-local condition scheduler at a given update."""
    names = cfg["train_names"][TASK]
    task_index = list(cfg["task_classes"]).index(TASK)
    generator = np.random.default_rng(
        cfg["scheduler_seed"] + 100003 * task_index + step // len(names)
    )
    return names[int(generator.permutation(len(names))[step % len(names)])]

