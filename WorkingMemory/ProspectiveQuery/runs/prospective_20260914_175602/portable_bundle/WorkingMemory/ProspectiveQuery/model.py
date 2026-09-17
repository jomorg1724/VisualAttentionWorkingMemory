"""Five-task continuation with one prospective sensory query residual."""
import copy
import math

import torch
from torch.nn import functional as F

from WorkingMemory.PreUpdateAttention.model import AttentionMemory, groups as parent_groups
from WorkingMemory.SpatialTaskBattery.BiasedTraining.model import (
    BiasedMemory,
    InstrumentedJointAttention,
)
from WorkingMemory.UnbiasedAttention.model import groups

VERSION = "spatial_five_task_prospective_query_v1"
ARM = "prospective_query"
PARENT_VERSION = "spatial_preupdate_joint_attention_v1"
PARENT_STEP = 8400
PARENT_SHA256 = "e37602aa20ccfc400ea8fe9d98c11f29c508069388897c55803b97f2ccdf1bc9"


class ProspectiveJointAttention(InstrumentedJointAttention):
    """Condition old-memory queries on the co-located current sensory field."""

    def __init__(self):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.zeros(()))

    def query_tokens(self, field, old):
        visual = field.flatten(2).transpose(1, 2)
        memory = old.flatten(2).transpose(1, 2)
        query_input = (
            self.query_norm(memory)
            + self.position
            + self.source[1]
            + self.gamma * self.query_norm(visual)
        )
        return self.query(query_input)

    def attention_weights(self, field, old):
        b = field.shape[0]
        visual = field.flatten(2).transpose(1, 2)
        memory = old.flatten(2).transpose(1, 2)
        q = self.query_tokens(field, old)
        raw = torch.cat((visual, memory), 1)
        identities = torch.cat(
            (self.position + self.source[0], self.position + self.source[1]), 0
        )
        k = self.key(self.key_norm(raw) + identities)
        q = q.reshape(b, 169, 2, 32).transpose(1, 2)
        k = k.reshape(b, 338, 2, 32).transpose(1, 2)
        logits = q @ k.transpose(-1, -2) / math.sqrt(32)
        bias = (
            self.source_bias.repeat_interleave(169, dim=1)[:, None, :]
            - F.softplus(self.raw_locality)[:, None, None] * self.distance_squared
        )
        return (logits + bias[None]).softmax(-1)

    def forward(self, field, old, diagnostic=False):
        b = field.shape[0]
        visual = field.flatten(2).transpose(1, 2)
        memory = old.flatten(2).transpose(1, 2)
        raw = torch.cat((visual, memory), 1)
        weights = self.attention_weights(field, old)
        v = self.value(raw).reshape(b, 338, 2, 32).transpose(1, 2)
        u = self.output((weights @ v).transpose(1, 2).reshape(b, 169, 64))
        collect = diagnostic or getattr(self, "capture", False)
        stats = (
            {}
            if not collect
            else dict(
                memory_attention_mass=float(
                    weights[:, :, :, 169:].detach().sum(-1).mean()
                ),
                memory_attention_mass_by_head=weights[:, :, :, 169:]
                .detach()
                .sum(-1)
                .mean((0, 2))
                .tolist(),
                attention_entropy_by_head=(
                    -(weights.clamp_min(1e-30).log() * weights).sum(-1)
                )
                .detach()
                .mean((0, 2))
                .tolist(),
                locality=F.softplus(self.raw_locality).detach().tolist(),
                gamma=float(self.gamma.detach()),
            )
        )
        if getattr(self, "capture", False):
            self.captured.append(stats)
        return u.transpose(1, 2).reshape(b, 64, 13, 13), stats


class ProspectiveMemory(BiasedMemory):
    def __init__(self, cfg):
        super().__init__(cfg)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(cfg["model_seed"])
            self.attention = ProspectiveJointAttention()


def migrate(parent, arm, cfg, parent_sha256):
    """Migrate only the intact attention8400 parent; never remigrate a child."""
    if parent.get("version") == VERSION:
        raise ValueError("Refusing already-migrated prospective-query checkpoint")
    if parent_sha256 != PARENT_SHA256:
        raise ValueError("Expected exact original attention8400 SHA256")
    if (
        parent.get("version") != PARENT_VERSION
        or parent.get("step") != PARENT_STEP
        or arm != ARM
        or cfg.get("battery") != "spatial"
    ):
        raise ValueError("Expected original attention8400 and five-task spatial recipe")

    old = AttentionMemory(activation_checkpoint=False)
    old.load_state_dict(parent["model"], strict=True)
    oldopt = torch.optim.Adam(
        parent_groups(old, parent["config"]), eps=parent["config"]["adam_eps"]
    )
    oldopt.load_state_dict(parent["optimizer"])

    model = ProspectiveMemory(cfg)
    state = model.state_dict()
    for name, value in parent["model"].items():
        if name not in state or state[name].shape != value.shape:
            raise RuntimeError("Unexpected parent tensor mismatch " + name)
        state[name] = value

    missing = set(state) - set(parent["model"])
    head_prefixes = tuple(
        "readout.heads." + task + "." for task in cfg["task_classes"]
    )
    expected_new = {"attention.gamma"} | {
        name for name in missing if name.startswith(head_prefixes)
    }
    if missing != expected_new or len(missing - {"attention.gamma"}) != 10:
        raise RuntimeError("Migration added tensors beyond gamma and five task heads")
    model.load_state_dict(state, strict=True)

    opt = torch.optim.Adam(groups(model, cfg), eps=cfg["adam_eps"])
    oldnames = dict(old.named_parameters())
    inherited = []
    for name, parameter in model.named_parameters():
        if name in oldnames and oldnames[name] in oldopt.state:
            opt.state[parameter] = copy.deepcopy(oldopt.state[oldnames[name]])
            inherited.append(name)

    return model, opt, dict(
        version=VERSION,
        parent_version=PARENT_VERSION,
        parent_step=PARENT_STEP,
        parent_sha256=PARENT_SHA256,
        retained_tensors=list(parent["model"]),
        retained_adam_names=inherited,
        new_tensors=sorted(missing),
        gamma_initial=0.0,
        gamma_shape=[],
        architecture_change=(
            "Q=Wq(LN(old_memory)+position+memory_source"
            "+gamma*LN(current_sensory)); all K/V, biases, recurrence and readout unchanged"
        ),
        initialization=(
            "Original intact attention8400 tensors and compatible Adam state; "
            "same deterministic five semantic heads as BiasedTraining"
        ),
    )
