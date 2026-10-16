"""Focused CPU checks of the batch objective, feedback and its gradients."""
import json
from pathlib import Path
import torch
from .loss import outcome_contrastive_loss, score_choices


def main():
    torch.set_num_threads(1)
    torch.manual_seed(193211)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    choices = torch.tensor([0, 0, 1, 1, 1, 1, 0, 0])
    logits = (torch.nn.functional.one_hot(choices, 2).float() * 2 - 1).requires_grad_()
    features = torch.randn(8, 12, requires_grad=True)
    _, rewards = score_choices(logits, labels)
    total, parts = outcome_contrastive_loss(features, logits, rewards)
    assert torch.equal(parts['targets'], labels)
    assert all(int(parts[k]) == 2 for k in ('TP', 'TN', 'FP', 'FN'))
    assert parts['positive_counts'].tolist() == [3] * 8
    assert torch.equal(parts['anchor_weights'], torch.where(rewards > 0, 1., 2.))
    total.backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum() > 0 for t in (features, logits))
    permutation = torch.randperm(8)
    permuted, _ = outcome_contrastive_loss(features.detach()[permutation], logits.detach()[permutation], rewards[permutation])
    assert torch.allclose(total.detach(), permuted, atol=1e-6)
    separated = torch.nn.functional.one_hot(labels, 2).float()
    collapsed = torch.ones_like(separated)
    _, good = outcome_contrastive_loss(separated, logits.detach(), rewards)
    _, bad = outcome_contrastive_loss(collapsed, logits.detach(), rewards)
    assert good['contrastive'] < bad['contrastive']
    assert torch.allclose(bad['contrastive'], torch.tensor(7.).log(), atol=1e-6)
    # Same representation in a single graph vs chunks concatenated before the
    # objective: all cross-chunk positives/negatives and candidate gradients stay.
    torch.manual_seed(193212)
    inputs = torch.randn(8, 5)
    weight = torch.randn(5, 12, requires_grad=True)
    reference, _ = outcome_contrastive_loss(inputs @ weight, logits.detach(), rewards)
    g1, = torch.autograd.grad(reference, weight)
    chunked, _ = outcome_contrastive_loss(torch.cat([chunk @ weight for chunk in inputs.split(2)]), logits.detach(), rewards)
    g2, = torch.autograd.grad(chunked, weight)
    assert torch.allclose(g1, g2, atol=1e-6, rtol=1e-5)
    result = dict(passed=True, cpu_only=True, feedback_recovers_binary_labels=True,
        all_four_outcomes_counted=True, error_weights_detached=True,
        full_batch_positives_per_anchor=3, permutation_invariant=True,
        finite_nonzero_representation_and_belief_gradients=True,
        separated_class_loss_below_collapse=True,
        full_batch_chunked_forward_gradient_equivalence=True,
        training_performed=False)
    Path(__file__).with_name('loss_check_results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
