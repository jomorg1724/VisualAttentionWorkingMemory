"""Choice + causal transport prediction + appearance-invariant dynamics."""
import torch
from torch.nn import functional as F


def objective(logits, details, labels):
    # Ordering: all B first appearances, then B independent second appearances.
    b = len(labels)
    decision = F.cross_entropy(logits, labels.repeat(2))
    z = details['transformations']
    a, other = z[:b].reshape(-1, 64), z[b:].reshape(-1, 64)
    # One positive per anchor: corresponding clip with identical dynamics.
    # Other clips are not forced apart (same trial may have unchanged motion).
    invariance = (a-other).square().mean()
    joined = torch.cat((a, other))
    std = (joined.var(0, unbiased=False)+1e-4).sqrt()
    variance = F.relu(.5-std).mean()
    centered = joined-joined.mean(0)
    covariance = centered.T@centered/max(1, len(joined)-1)
    covariance = (covariance.square().sum()-covariance.diagonal().square().sum())/64
    representation = invariance+variance+.04*covariance
    prediction = details['prediction_loss']
    loss = decision + .5*prediction + .1*representation
    return loss, dict(decision=decision, prediction=prediction, representation=representation,
                     invariance=invariance, variance=variance, covariance=covariance,
                     transform_std=std.mean())
