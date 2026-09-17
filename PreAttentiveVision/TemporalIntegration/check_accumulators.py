"""Focused CPU-only checks of the newly introduced causal computation."""
import copy
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from PreAttentiveVision.hybrid_models import SEResidualEncoder
from PreAttentiveVision.neuroscience_stimuli import TASK_CLASSES
from PreAttentiveVision.TemporalIntegration.accumulators import (
    MODEL_NAMES, StreamingPAVClassifier, SpatialKDA, SpatialConvGRU,
    OpponentAccumulator, kda_update,
)


def digest_common(model):
    result = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name.startswith(("projections.", "readout.")):
            result.update(name.encode())
            result.update(value.detach().numpy().tobytes())
    return result.hexdigest()


def check():
    torch.set_num_threads(2)
    torch.manual_seed(40991)
    started = time.monotonic()
    parent = Path(__file__).parents[1] / "runs/allocation_20260912_160414/contour_focus_seed20271/checkpoint_002268.pt"
    checkpoint = torch.load(parent, map_location="cpu")
    encoder = SEResidualEncoder()
    encoder.load_state_dict({key[len("encoder."):]: value for key,value in checkpoint["model"].items()
                             if key.startswith("encoder.")}, strict=True)
    results = {"device":"cpu", "threads":2, "parent":str(parent),
               "parent_step":checkpoint["step"], "models":{}}
    common = []
    pair = torch.rand(1,2,3,100,100)
    for name in MODEL_NAMES:
        model = StreamingPAVClassifier(copy.deepcopy(encoder), TASK_CLASSES, name).eval()
        common.append(digest_common(model))
        encoder_calls = []
        hook = model.encoder.register_forward_pre_hook(lambda module,args: encoder_calls.append(list(args[0].shape)))
        with torch.no_grad():
            batch_logits = model(pair, "motion_direction")
        hook.remove()
        assert encoder_calls == [[1,3,100,100],[1,3,100,100]], encoder_calls
        with torch.no_grad():
            first_features, state = model.step(pair[:,0])
            second_features, _ = model.step(pair[:,1], state)
            stream_logits = model.classify(second_features, "motion_direction")
            torch.testing.assert_close(batch_logits, stream_logits, rtol=0, atol=0)
            # A separate sequence cannot mutate hidden internal state.
            model(torch.rand_like(pair), "motion_direction")
            again, _ = model.step(pair[:,0], None)
            torch.testing.assert_close(first_features, again, rtol=0, atol=0)
            reset_logits = model(pair,"motion_direction",reset_before_second=True)
            current_only, _ = model.step(pair[:,1], None)
            torch.testing.assert_close(reset_logits, model.classify(current_only,"motion_direction"),rtol=0,atol=0)
        initial_encoder = {key:value.clone() for key,value in model.encoder.state_dict().items()}
        model.train()
        assert not model.encoder.training and all(not p.requires_grad for p in model.encoder.parameters())
        optimizer = torch.optim.Adam(model.trainable_parameters(),lr=.001,weight_decay=.0001)
        loss = F.cross_entropy(model(pair,"motion_direction"),torch.tensor([1]))
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is None for p in model.encoder.parameters())
        for module in (model.projections,model.accumulators,model.readout):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            assert sum(float(g.abs().sum()) for g in grads)>0
        torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()),5.)
        optimizer.step()
        assert all(torch.equal(value,initial_encoder[key]) for key,value in model.encoder.state_dict().items())
        results["models"][name] = dict(stream_equals_unroll=True,reset_isolation=True,
              reset_before_second_equals_current_only=True,encoder_two_separate_calls=True,
              encoder_frozen_after_optimizer=True,finite_nonzero_new_gradients=True,
              trainable_parameters=sum(p.numel() for p in model.trainable_parameters()),
              core_parameters=sum(p.numel() for p in model.accumulators.parameters()),
              state_elements_per_example=model.state_elements_per_example,
              state_bytes_per_example=4*model.state_elements_per_example)
        del model, optimizer
    assert len(set(common)) == 1
    results["common_initialization_sha256"] = common[0]
    # Independent scalar/head matrix expression for KDA's exact recurrence.
    state = torch.randn(3,8,16,dtype=torch.float64)
    q=F.normalize(torch.randn(3,8,dtype=torch.float64),dim=-1)
    k=F.normalize(torch.randn(3,8,dtype=torch.float64),dim=-1)
    v=torch.randn(3,16,dtype=torch.float64)
    alpha=torch.rand(3,8,dtype=torch.float64)
    beta=torch.rand(3,1,dtype=torch.float64)
    observed,updated=kda_update(state,q,k,v,alpha,beta)
    for i in range(3):
        decayed=torch.diag(alpha[i]) @ state[i]
        expected=decayed+beta[i,0]*torch.outer(k[i],v[i]-decayed.T@k[i])
        torch.testing.assert_close(updated[i],expected,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(observed[i],expected.T@q[i],rtol=1e-12,atol=1e-12)
    results["kda_matches_independent_matrix_equation"]=True
    for cls in (SpatialKDA,SpatialConvGRU,OpponentAccumulator):
        core=cls()
        first=torch.randn(1,32,9,9,requires_grad=True)
        second=torch.randn(1,32,9,9,requires_grad=True)
        _,state=core(first)
        output,_=core(second,state)
        output.square().mean().backward()
        assert first.grad is not None and torch.isfinite(first.grad).all() and first.grad.abs().sum()>0
        assert second.grad is not None and torch.isfinite(second.grad).all() and second.grad.abs().sum()>0
    results["all_cores_backpropagate_through_both_steps"]=True
    core=OpponentAccumulator()
    first=torch.randn(1,32,13,13)
    second=torch.roll(first,1,-1)
    with torch.no_grad():
        _,initial=core(first)
        _,initial_opponent=core.energy_channels(*initial)
        assert initial_opponent.abs().max()==0
        _,static=core(first,initial)
        _,static_opponent=core.energy_channels(*static)
        torch.testing.assert_close(static_opponent,torch.zeros_like(static_opponent),rtol=0,atol=1e-5)
        _,forward=core(second,initial)
        _,reverse_initial=core(second)
        _,reverse=core(first,reverse_initial)
        _,forward_opponent=core.energy_channels(*forward)
        _,reverse_opponent=core.energy_channels(*reverse)
        assert forward_opponent.abs().max()>.01
        torch.testing.assert_close(forward_opponent,-reverse_opponent,rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(forward[0]-forward[1],.5*(second-first),rtol=1e-5,atol=1e-6)
    results["opponent_static_zero_and_swap_sign_reversal"]=True
    results["status"]="passed"
    results["wall_seconds"]=time.monotonic()-started
    target=Path(__file__).with_name("focused_checks.json")
    target.write_text(json.dumps(results,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(results,indent=2))


if __name__=="__main__":
    check()
