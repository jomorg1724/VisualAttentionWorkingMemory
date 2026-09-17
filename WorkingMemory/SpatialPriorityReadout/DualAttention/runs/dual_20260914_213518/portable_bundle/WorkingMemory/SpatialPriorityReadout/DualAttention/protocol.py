"""Exact historical five-task recipe and exposure, with no task intervention."""
import copy

from WorkingMemory.UnbiasedAttention.protocol import (
    SPATIAL_TEST_SEED,
    SPATIAL_VAL_SEED,
    recipe as inherited_recipe,
    summarize_spatial,
)

ARM = "dual_attention_spatial_priority_scratch"
START_STEP = 0
UPDATES = 4000
BATCH_SIZE = 8
TASKS_PER_UPDATE = 5
EPISODES_PER_UPDATE = 40
TOTAL_EPISODES = 160000
VALIDATION_TARGETS = (800, 1600, 2400, 3200, 4000)
VALIDATION_N = 64
TEST_N = 256


def recipe():
    cfg = copy.deepcopy(inherited_recipe("spatial_unbiased"))
    cfg["updates"] = UPDATES
    if (
        cfg["batch_size"] != BATCH_SIZE
        or len(cfg["task_classes"]) != TASKS_PER_UPDATE
        or cfg["battery"] != "spatial"
    ):
        raise RuntimeError("Historical BiasedTraining recipe changed")
    return cfg


def assert_fixed(cfg):
    reference = inherited_recipe("spatial_unbiased")
    reference["updates"] = UPDATES
    if cfg != reference:
        raise ValueError("Dual attention experiment diverged from control recipe")
    return True
