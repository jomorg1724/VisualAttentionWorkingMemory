"""Bounded CPU checks of spatial recurrence, CLS feedback and full-trial gradients."""
import json
from pathlib import Path
import torch
from SecondPass.SpatialReadout.model import FinalConvGRU
from .model import OutcomeContrastiveRViT, SpatialGRU
from .dataset import generate
from .loss import outcome_contrastive_loss, score_choices


def main():
    torch.set_num_threads(1)
    torch.manual_seed(193211)
    reference, adapted = FinalConvGRU(), SpatialGRU(64)
    adapted.load_state_dict(reference.state_dict())
    x, h = torch.randn(2, 64, 7, 7), torch.randn(2, 64, 7, 7)
    assert torch.equal(reference(x, h), adapted(x, h))
    rows = [generate(i) for i in range(4)]
    frames = torch.stack([r[0] for r in rows]).requires_grad_()
    labels = torch.tensor([r[1] for r in rows])
    model = OutcomeContrastiveRViT()
    logits, cls, history = model(frames, return_history=True)
    for field, memory_cls in history:
        field.retain_grad(); memory_cls.retain_grad()
    _, rewards = score_choices(logits, labels)
    loss, parts = outcome_contrastive_loss(cls, logits, rewards)
    loss.backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not missing, missing
    assert frames.grad[:, 0].abs().sum() > 0
    assert history[0][0].grad.abs().sum() > 0 and history[0][1].grad.abs().sum() > 0
    with torch.no_grad():
        state = None
        for frame in frames.unbind(1):
            state = model.step(frame, state)
        assert torch.equal(cls, state[1])
        changed = frames.detach().clone(); changed[:, 3:] = .5
        _, _, different_history = model(changed, return_history=True)
        assert all(torch.equal(history[t][1], different_history[t][1]) for t in range(3))
        assert not torch.equal(history[-1][1], different_history[-1][1])
        chunked = [model(chunk) for chunk in frames.detach().split(2)]
        assert torch.allclose(torch.cat([v[0] for v in chunked]), logits, atol=2e-6)
    result = dict(passed=True, cpu_only=True, parameter_count=sum(p.numel() for p in model.parameters()),
        parameter_tensors=len(list(model.parameters())), spatial_gru_matches_existing_equations_exactly=True,
        spatial_state=[128, 10, 10], cls_memory_width=128, frames=6,
        streaming_matches_batched=True, causal_prefix_independent_of_future=True,
        gradients_reach_first_frame_spatial_memory_and_cls=True,
        all_parameter_gradients_present_and_finite=True, chunked_forward_matches=True,
        training_performed=False)
    Path(__file__).with_name('model_check_results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
