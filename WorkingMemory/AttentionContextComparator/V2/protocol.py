"""AV-context v2 protocol: unchanged five-task recipe, 12,000 updates, pre-registered gate."""
import copy
import math

from WorkingMemory.AttentionContextComparator.protocol import (
    SPATIAL_TEST_SEED,
    SPATIAL_VAL_SEED,
    recipe as v1_recipe,
)
from WorkingMemory.AttentionContextComparator.V2.model import ALL_CHANGES, ARM, CHANGE_KEYS

START_STEP = 0
UPDATES = 12000
BATCH_SIZE = 8
TASKS_PER_UPDATE = 5
EPISODES_PER_UPDATE = 40
TOTAL_EPISODES = UPDATES * EPISODES_PER_UPDATE
VALIDATION_TARGETS = tuple(range(800, UPDATES + 1, 800))
VALIDATION_N = 64
TEST_N = 256
DIAGNOSTIC_EVERY = 32
CHECKPOINT_EVERY = 256

# Pre-registered stopping rule, evaluated on the validation-2400 evaluation.
GATE_STEP = 2400
GATE_EPISODES = GATE_STEP * EPISODES_PER_UPDATE
CUED_TASKS = ("orientation_cued", "motion_duration_cued", "krauzlis_cued_motion")
GATE_PRIORITY_ENTROPY_FRACTION = 0.98  # of ln(169), on at least one cued task
GATE_HEAD1_MASS_BEYOND_3 = 0.01  # head-1 attention mass beyond 3 cells
UNIFORM_PRIORITY_ENTROPY = math.log(169)
RUNTIME_SECONDS = 40 * 3600


def recipe(changes=None):
    cfg = copy.deepcopy(v1_recipe())
    cfg["updates"] = UPDATES
    enabled = dict(ALL_CHANGES)
    if changes:
        enabled.update(changes)
    cfg["v2_changes"] = {key: bool(enabled[key]) for key in CHANGE_KEYS}
    if cfg["v2_changes"]["E"]:
        cfg["parent_lr"] = cfg["new_lr"]
    if (
        cfg["batch_size"] != BATCH_SIZE
        or len(cfg["task_classes"]) != TASKS_PER_UPDATE
        or cfg["battery"] != "spatial"
    ):
        raise RuntimeError("Historical five-task recipe changed")
    return cfg


def assert_fixed(cfg):
    """Everything except updates, the change flags and (under E) parent_lr is the v1 recipe."""
    reference = copy.deepcopy(v1_recipe())
    probe = copy.deepcopy(cfg)
    if probe.pop("updates") != UPDATES:
        raise ValueError("v2 exposure must be 12,000 updates")
    changes = probe.pop("v2_changes")
    if set(changes) != set(CHANGE_KEYS):
        raise ValueError("Malformed v2 change flags")
    if changes["E"]:
        if probe["parent_lr"] != probe["new_lr"] or probe["new_lr"] != reference["new_lr"]:
            raise ValueError("Change E must unify parent_lr with the historical new_lr")
        probe["parent_lr"] = reference["parent_lr"]
    reference.pop("updates")
    if probe != reference:
        raise ValueError("v2 recipe diverged from the pinned five-task recipe")
    return True


def gate(validation):
    """Apply the pre-registered rule to a validation summary with v2 diagnostics."""
    entropy = validation["priority_entropy_fraction_by_task"]
    cued = {task: entropy[task] for task in CUED_TASKS}
    head1 = validation["attention_mass_beyond_3_by_head"][1]
    entropy_ok = min(cued.values()) < GATE_PRIORITY_ENTROPY_FRACTION
    mass_ok = head1 > GATE_HEAD1_MASS_BEYOND_3
    return dict(
        step=validation["step"],
        cued_priority_entropy_fraction=cued,
        head1_attention_mass_beyond_3=head1,
        priority_entropy_criterion=entropy_ok,
        head1_mass_criterion=mass_ok,
        passed=bool(entropy_ok and mass_ok),
        rule=(
            f"At step {GATE_STEP} require both: priority entropy below "
            f"{GATE_PRIORITY_ENTROPY_FRACTION:.0%} of ln(169) on at least one of "
            f"{CUED_TASKS}, and head-1 attention mass beyond 3 cells above "
            f"{GATE_HEAD1_MASS_BEYOND_3:.0%}. Failing either stops the arm."
        ),
    )
