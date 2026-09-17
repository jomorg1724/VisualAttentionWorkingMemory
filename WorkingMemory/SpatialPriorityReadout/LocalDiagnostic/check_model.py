"""CPU smoke checks for the two diagnostic probe architectures."""
import json
from pathlib import Path

import torch

from WorkingMemory.SpatialPriorityReadout.LocalDiagnostic.model import (
    PooledProbe,
    SpatialPriorityProbe,
    parameter_count,
    target_region_mask,
)


def main():
    x = torch.randn(3, 192, 13, 13)
    pooled = PooledProbe()
    spatial = SpatialPriorityProbe()
    a = pooled(x)
    b, priority, evidence = spatial(x, True)
    mask = target_region_mask(torch.tensor([0, 1, 3]))
    loss = a.square().mean() + b.square().mean()
    loss.backward()
    result = {
        "status": "passed",
        "input_shape": list(x.shape),
        "output_shape": list(a.shape),
        "priority_shape": list(priority.shape),
        "evidence_shape": list(evidence.shape),
        "priority_normalized": bool(
            torch.allclose(priority.sum((1, 2)), torch.ones(3))
        ),
        "target_region_pixels_each": mask.sum((1, 2)).tolist(),
        "pooled_parameters": parameter_count(pooled),
        "spatial_parameters": parameter_count(spatial),
        "capacity_difference": parameter_count(spatial) - parameter_count(pooled),
        "finite_backward": all(
            p.grad is None or torch.isfinite(p.grad).all()
            for module in (pooled, spatial)
            for p in module.parameters()
        ),
    }
    assert result["priority_normalized"]
    assert result["target_region_pixels_each"] == [9, 9, 9]
    assert result["finite_backward"]
    out = Path(__file__).with_name("construction_checks.json")
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

