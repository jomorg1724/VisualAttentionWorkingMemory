"""Pinned five-task protocol; this module intentionally introduces no recipe changes."""
import copy

from WorkingMemory.UnbiasedAttention.protocol import (
    SPATIAL_TEST_SEED,
    SPATIAL_VAL_SEED,
    recipe as inherited_recipe,
    summarize_spatial,
)

ARM = "prospective_query"
START_STEP = 8400
UPDATES = 4000
BATCH_SIZE = 8
TASKS_PER_UPDATE = 5
EPISODES_PER_UPDATE = BATCH_SIZE * TASKS_PER_UPDATE
TOTAL_EPISODES = UPDATES * EPISODES_PER_UPDATE
VALIDATION_TARGETS = (9200, 10000, 10800, 11600, 12400)
MATCHED_CONTROL_VALIDATION_TARGETS = (9200, 10000, 10800, 11600)
VALIDATION_N = 64
TEST_N = 256


def comparison_status(step=None, heldout=False):
    if not heldout and step in MATCHED_CONTROL_VALIDATION_TARGETS:
        return "matched_historical_control_validation"
    return "unmatched_exploratory"


def recipe():
    cfg = copy.deepcopy(inherited_recipe("spatial_unbiased"))
    cfg["updates"] = UPDATES
    if cfg["batch_size"] != BATCH_SIZE or len(cfg["task_classes"]) != TASKS_PER_UPDATE:
        raise RuntimeError("Inherited five-task exposure changed")
    return cfg


def assert_same_as_biased_training(cfg):
    reference = inherited_recipe("spatial_unbiased")
    reference["updates"] = UPDATES
    if cfg != reference:
        raise ValueError("Prospective-query recipe diverged from BiasedTraining")
    return True
