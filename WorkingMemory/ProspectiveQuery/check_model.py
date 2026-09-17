"""Focused CPU equivalence, migration, routing, and lineage checks."""
import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from PreAttentiveVision.train import sha
from WorkingMemory.PreUpdateAttention.model import AttentionMemory
from WorkingMemory.ProspectiveQuery.model import (
    ARM,
    PARENT_SHA256,
    VERSION,
    ProspectiveJointAttention,
    migrate,
)
from WorkingMemory.ProspectiveQuery.protocol import (
    EPISODES_PER_UPDATE,
    MATCHED_CONTROL_VALIDATION_TARGETS,
    TOTAL_EPISODES,
    UPDATES,
    assert_same_as_biased_training,
    comparison_status,
    recipe,
)
from WorkingMemory.ProspectiveQuery.sweep import newest_indexed_checkpoint
from WorkingMemory.SpatialTaskBattery.BiasedTraining.model import (
    BiasedMemory,
    InstrumentedJointAttention,
)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.manual_seed(80217)
started = time.time()

parent_path = (
    ROOT
    / "WorkingMemory/PreUpdateAttention/runs/attention_20260913_143459/"
    "retrieved/remote_results/preupdate_attention/checkpoint_008400.pt"
)
if sha(parent_path) != PARENT_SHA256:
    raise RuntimeError("Original attention8400 artifact hash changed")
parent = torch.load(parent_path, map_location="cpu")
cfg = recipe()
assert_same_as_biased_training(cfg)
model, optimizer, lineage = migrate(parent, ARM, cfg, sha(parent_path))

# All parent tensors and all compatible Adam entries remain exact.
for name, value in parent["model"].items():
    if not torch.equal(model.state_dict()[name], value):
        raise AssertionError("Changed inherited tensor " + name)
old = AttentionMemory(activation_checkpoint=False)
old.load_state_dict(parent["model"], strict=True)
from WorkingMemory.PreUpdateAttention.model import groups as old_groups

old_optimizer = torch.optim.Adam(
    old_groups(old, parent["config"]), eps=parent["config"]["adam_eps"]
)
old_optimizer.load_state_dict(parent["optimizer"])
oldnames = dict(old.named_parameters())
newnames = dict(model.named_parameters())
for name in lineage["retained_adam_names"]:
    for key, expected in old_optimizer.state[oldnames[name]].items():
        actual = optimizer.state[newnames[name]][key]
        if torch.is_tensor(expected):
            if not torch.equal(expected, actual):
                raise AssertionError("Changed inherited Adam tensor " + name + "/" + key)
        elif expected != actual:
            raise AssertionError("Changed inherited Adam value " + name + "/" + key)

# At gamma=0, query tokens, weights and attention outputs reproduce the original.
original_attention = InstrumentedJointAttention()
original_attention.load_state_dict(
    {name.removeprefix("attention."): value for name, value in parent["model"].items()
     if name.startswith("attention.")},
    strict=True,
)
prospective_attention = ProspectiveJointAttention()
prospective_attention.load_state_dict(
    dict(original_attention.state_dict(), gamma=torch.zeros(())), strict=True
)
field = torch.randn(2, 64, 13, 13)
memory = torch.randn(2, 64, 13, 13)
with torch.no_grad():
    original_query = original_attention.query(
        original_attention.query_norm(memory.flatten(2).transpose(1, 2))
        + original_attention.position
        + original_attention.source[1]
    )
    new_query = prospective_attention.query_tokens(field, memory)
    original_output, _ = original_attention(field, memory)
    new_output, _ = prospective_attention(field, memory)
query_zero_error = float((new_query - original_query).abs().max())
attention_zero_error = float((new_output - original_output).abs().max())
if query_zero_error > 1e-7 or attention_zero_error > 1e-7:
    raise AssertionError("gamma=0 changed original attention")

# Gamma receives a nonzero gradient at zero; sensory conditioning changes Q/routing.
prospective_attention.zero_grad(set_to_none=True)
weighted = prospective_attention(field, memory)[0] * torch.linspace(
    0.1, 1.0, field.numel() // field.shape[0]
).reshape(1, 64, 13, 13)
weighted.sum().backward()
gamma_gradient = float(prospective_attention.gamma.grad)
if not torch.isfinite(prospective_attention.gamma.grad) or abs(gamma_gradient) <= 1e-9:
    raise AssertionError("Zero-initialized gamma has no usable gradient")

changed = field.clone()
changed[:, :, 4:8, 7:10] += torch.linspace(-1.25, 1.25, 64).reshape(1, 64, 1, 1)
with torch.no_grad():
    prospective_attention.gamma.fill_(0.25)
    query_before = prospective_attention.query_tokens(field, memory)
    query_after = prospective_attention.query_tokens(changed, memory)
    route_before = prospective_attention.attention_weights(field, memory)
    route_after = prospective_attention.attention_weights(changed, memory)
query_change = float((query_after - query_before).abs().max())
routing_change = float((route_after - route_before).abs().max())
if query_change <= 1e-6 or routing_change <= 1e-8:
    raise AssertionError("Changed sensory field did not change query/routing")

# A full inherited old-task output is also identical at gamma=0.
model.eval()
old.eval()
model.attention.gamma.data.zero_()
images = torch.rand(1, 2, 3, 100, 100)
with torch.no_grad():
    inherited_logits = old(images, "motion_direction")
    prospective_logits = model(images, "motion_direction")
full_output_error = float((prospective_logits - inherited_logits).abs().max())
if full_output_error > 2e-6:
    raise AssertionError("Unrelated inherited output changed at gamma=0")

new_tensors = set(model.state_dict()) - set(parent["model"])
expected_heads = {
    f"readout.heads.{task}.{suffix}"
    for task in cfg["task_classes"]
    for suffix in ("weight", "bias")
}
if new_tensors != expected_heads | {"attention.gamma"}:
    raise AssertionError("Unexpected new parameter/buffer lineage")
biased_reference = BiasedMemory(cfg)
for name in expected_heads:
    if not torch.equal(model.state_dict()[name], biased_reference.state_dict()[name]):
        raise AssertionError("Fresh semantic head differs from BiasedTraining " + name)
if not torch.equal(model.attention.source_bias, old.attention.source_bias):
    raise AssertionError("Source bias changed")
if not torch.equal(model.attention.raw_locality, old.attention.raw_locality):
    raise AssertionError("Locality changed")
if (UPDATES, EPISODES_PER_UPDATE, TOTAL_EPISODES) != (4000, 40, 160000):
    raise AssertionError("Exposure changed")
if MATCHED_CONTROL_VALIDATION_TARGETS != (9200, 10000, 10800, 11600):
    raise AssertionError("Historical matched-control scope changed")
if comparison_status(12400) != "unmatched_exploratory":
    raise AssertionError("Unmatched terminal validation is mislabeled")
if comparison_status(10000, heldout=True) != "unmatched_exploratory":
    raise AssertionError("Unmatched held-out result is mislabeled")

historical_training = (
    ROOT
    / "WorkingMemory/SpatialTaskBattery/BiasedTraining/runs/"
    "biased_20260913_194506/retrieved/biased_results/spatial_biased/training"
)
resume_checkpoint = newest_indexed_checkpoint(historical_training, 10000)
if Path(resume_checkpoint).name != "checkpoint_010000.pt":
    raise AssertionError("Newest indexed checkpoint selection failed")

try:
    migrate({"version": VERSION}, ARM, cfg, PARENT_SHA256)
except ValueError as error:
    remigration_refused = "already-migrated" in str(error)
else:
    remigration_refused = False
if not remigration_refused:
    raise AssertionError("Already-migrated checkpoint was accepted")
try:
    migrate(parent, ARM, cfg, "0" * 64)
except ValueError as error:
    wrong_hash_refused = "SHA256" in str(error)
else:
    wrong_hash_refused = False
if not wrong_hash_refused:
    raise AssertionError("Wrong parent hash was accepted")

result = dict(
    status="passed",
    seconds=time.time() - started,
    parent_sha256=PARENT_SHA256,
    parent_version=parent["version"],
    parent_step=parent["step"],
    retained_tensors_exact=len(parent["model"]),
    retained_adam_states_exact=len(lineage["retained_adam_names"]),
    new_tensors=sorted(new_tensors),
    total_parameters=sum(parameter.numel() for parameter in model.parameters()),
    gamma_initial=float(model.attention.gamma),
    gamma_gradient_at_zero=gamma_gradient,
    gamma_zero_query_max_abs_error=query_zero_error,
    gamma_zero_attention_output_max_abs_error=attention_zero_error,
    gamma_zero_full_model_output_max_abs_error=full_output_error,
    changed_sensory_query_max_abs_change=query_change,
    changed_sensory_routing_max_abs_change=routing_change,
    remigration_refused=remigration_refused,
    wrong_parent_hash_refused=wrong_hash_refused,
    fixed_exposure=dict(updates=UPDATES, episodes_per_update=EPISODES_PER_UPDATE,
                        episodes=TOTAL_EPISODES),
    recipe_exact_biased_training=True,
    fresh_heads_exact_biased_training=True,
    source_bias_exact=True,
    locality_exact=True,
    newest_indexed_checkpoint_at_10000=Path(resume_checkpoint).name,
    matched_historical_control_validation_steps=list(
        MATCHED_CONTROL_VALIDATION_TARGETS
    ),
    terminal_and_heldout_label="unmatched_exploratory",
)
output = Path(__file__).with_name("construction_checks.json")
output.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
