"""Binary outcome-derived, mistake-weighted supervised contrastive learning.

Adapted from Khosla et al. (2020), Eq. 2. This is not a CPC objective or a
policy-gradient estimator. Similarities include the entire current batch.
"""
import torch
from torch.nn import functional as F


@torch.no_grad()
def score_choices(logits, labels):
    """Environment scores a terminal argmax choice; no labels enter the model."""
    choices = logits.argmax(dim=-1)
    rewards = 2 * choices.eq(labels).to(logits.dtype) - 1
    return choices, rewards


def outcome_contrastive_loss(memory_cls, logits, rewards, temperature=.1,
                             contrastive_weight=.1, error_weight=2.):
    """Recover the correct binary answer from choice +/- correctness feedback.

    Full-batch positive sets use that answer, not correctness alone. Argmax,
    feedback, class masks and mistake weights are detached; all representations
    remain differentiable as both anchors and comparison examples.
    """
    if memory_cls.ndim != 2 or logits.shape != (len(memory_cls), 2):
        raise ValueError('Need [batch,features] CLS memory and [batch,2] logits')
    if rewards.shape != (len(memory_cls),) or not bool(((rewards == 1) | (rewards == -1)).all()):
        raise ValueError('Each trial needs a +1 correct / -1 incorrect outcome')
    if temperature <= 0 or contrastive_weight < 0 or error_weight < 1:
        raise ValueError('Invalid objective constants')
    choice = logits.detach().argmax(dim=-1)
    correct = rewards.detach().gt(0)
    targets = torch.where(correct, choice, 1 - choice)
    n = len(targets)
    diagonal = torch.eye(n, dtype=torch.bool, device=logits.device)
    positives = targets[:, None].eq(targets[None, :]) & ~diagonal
    counts = positives.sum(dim=1)
    if not bool((counts > 0).all()) or len(targets.unique()) != 2:
        raise ValueError('Full batch must contain at least two trials of each answer')
    z = F.normalize(memory_cls, dim=-1, eps=1e-8)
    scores = (z @ z.T) / temperature
    denominator = torch.logsumexp(scores.masked_fill(diagonal, -torch.inf), dim=1)
    # Mask only when selecting positive terms: avoid zero * -inf -> NaN.
    log_probability = scores - denominator[:, None]
    per_anchor = -(log_probability * positives.to(scores.dtype)).sum(1) / counts
    weights = torch.where(correct, torch.ones_like(rewards), torch.full_like(rewards, error_weight)).detach()
    contrastive = (per_anchor * weights).sum() / weights.sum()
    decision = F.cross_entropy(logits, targets)
    total = decision + contrastive_weight * contrastive
    outcomes = dict(TP=((choice == 1) & (targets == 1)).sum(),
                    TN=((choice == 0) & (targets == 0)).sum(),
                    FP=((choice == 1) & (targets == 0)).sum(),
                    FN=((choice == 0) & (targets == 1)).sum())
    return total, dict(decision=decision, contrastive=contrastive,
                      targets=targets, choices=choice, rewards=rewards.detach(),
                      anchor_weights=weights, positive_counts=counts, **outcomes)
