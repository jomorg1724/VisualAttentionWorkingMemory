"""CPU engineering checks for exact input, independent histories and gradients."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from . import dataset
from .model import WeightedSumTransformer, weighted_sum


def main():
    torch.set_num_threads(1)
    torch.manual_seed(191111)
    checks = {}
    frames = torch.stack([torch.full((3, 100, 100), x) for x in (.1, .2, .9)])[None]
    raw = weighted_sum(frames, normalized=False)
    assert torch.equal(raw, frames[:, 0] + 2 * frames[:, 1] + 3 * frames[:, 2])
    assert torch.allclose(raw, 6 * weighted_sum(frames), atol=3e-7)
    assert raw.max() > 1 and not torch.equal(weighted_sum(frames), weighted_sum(frames.flip(1)))
    checks['exact_ordered_123_sum_reversible_normalization_no_raw_clipping'] = True

    rows = [dataset.generate(i) for i in range(24)]
    assert sum(r[2] for r in rows) == 12
    for label in (0, 1):
        assert {r[3]['cell'] for r in rows if r[2] == label} == set(range(6))
    assert len(set(dataset.NAMESPACES.values())) == 3
    assert torch.equal(dataset.generate(5)[0], dataset.generate(5)[0])
    assert not torch.equal(dataset.generate(5)[0], dataset.generate(5, 'val')[0])
    assert not torch.equal(dataset.generate(5)[0], dataset.generate_evaluation(5, 'train')[0])
    assert not torch.equal(dataset.generate(5)[0], dataset.generate(4)[0])
    for i in range(12):
        a, b, label, meta = dataset.generate_evaluation(i, 'test')
        positions = np.asarray(meta['positions_unwrapped_xy'])
        assert np.allclose(np.diff(positions[:4], axis=0), np.asarray(meta['velocity_before_xy'])[None, None])
        assert np.allclose(np.diff(positions[3:], axis=0), np.asarray(meta['velocity_after_xy'])[None, None])
        if label:
            a0, b0, _, _ = dataset.generate_evaluation(i - 1, 'test')
            assert torch.equal(a, a0) and torch.equal(b[0], b0[0])
            assert not torch.equal(b[-1], b0[-1])
    checks['balanced_six_cells_disjoint_streams_continuous_movies_paired_raster_prefix'] = True

    model = WeightedSumTransformer()
    blocks = [m for m in model.modules() if isinstance(m, nn.TransformerEncoderLayer)]
    assert len(blocks) == 1 and blocks[0].self_attn.num_heads == 4
    assert blocks[0].norm_first and blocks[0].self_attn.batch_first
    assert blocks[0].linear1.in_features == 128 and blocks[0].linear1.out_features == 512
    assert all(m.p == 0 for m in model.modules() if isinstance(m, nn.Dropout))
    assert model.cls_token.shape == (1, 1, 128) and model.position_embedding.shape == (1, 101, 128)
    assert sum(isinstance(m, nn.Conv2d) for m in model.modules()) == 1
    assert model.patch_embedding.kernel_size == (10, 10) and model.patch_embedding.stride == (10, 10)
    assert all(p.requires_grad for p in model.parameters())
    checks['exact_one_shared_four_head_preLN_block_100patches_learned_CLS_positions'] = True

    a = torch.stack([r[0] for r in rows[:4]])
    b = torch.stack([r[1] for r in rows[:4]])
    images = weighted_sum(a)
    patches = F.unfold(images - .5, kernel_size=10, stride=10).transpose(1, 2)
    manual = F.linear(patches, model.patch_embedding.weight.flatten(1), model.patch_embedding.bias)
    actual = model.patch_embedding(images - .5).flatten(2).transpose(1, 2)
    assert torch.allclose(manual, actual, atol=2e-7)
    checks['patch_embedding_equals_unfold_plus_linear_no_local_feature_network'] = True

    with torch.no_grad():
        separate_a = model.encode_clips(a)
        separate_b = model.encode_clips(b)
        combined = model.encode_clips(torch.cat([a, b]))
        assert torch.allclose(separate_a, combined[:4], atol=2e-6)
        assert torch.allclose(separate_b, combined[4:], atol=2e-6)
        changed_b = b.flip(-1)
        changed_combined = model.encode_clips(torch.cat([a, changed_b]))
        assert torch.equal(combined[:4], changed_combined[:4])
        assert not torch.equal(combined[4:], changed_combined[4:])
        assert torch.equal(model.encode_clips(a), model.encode_images(weighted_sum(a)))
        reference = model(a, b)
        manual_logits = model.classifier(torch.cat([separate_a, separate_b], dim=1)).squeeze(-1)
        assert torch.allclose(reference, manual_logits, atol=2e-6)
        assert not torch.equal(reference, model(b, a))
        assert torch.equal(model(a, b), model(a, b))
    checks['CLS_depends_on_pixels_no_cross_clip_attention_shared_weights_ordered_FFN'] = True

    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    a_grad = a.clone().requires_grad_()
    b_grad = b.clone().requires_grad_()
    logits = model(a_grad, b_grad)
    loss = F.binary_cross_entropy_with_logits(logits, torch.tensor([r[2] for r in rows[:4]], dtype=torch.float32))
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    for grad in (a_grad.grad, b_grad.grad):
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
        assert torch.allclose(grad[:, 1], 2 * grad[:, 0], atol=1e-9)
        assert torch.allclose(grad[:, 2], 3 * grad[:, 0], atol=1e-9)
    optimizer.step()
    for name, parameter in model.named_parameters():
        assert not torch.equal(parameter, initial[name]), name
        assert int(optimizer.state[parameter]['step']) == 1, name
    checks['all_parameters_and_both_histories_finite_nonzero_gradients_Adam_weighted_chain_rule'] = True
    model.eval()
    with torch.no_grad():
        assert model(a[:1].expand(32, -1, -1, -1, -1), b[:1].expand(32, -1, -1, -1, -1)).shape == (32,)
    checks['batch32_output_shape'] = True

    directory = Path(__file__).parent
    result = dict(status='passed_cpu_engineering_only', torch_version=torch.__version__,
                  threads=torch.get_num_threads(), parameters=sum(p.numel() for p in model.parameters()),
                  parameter_tensors=len(list(model.parameters())), checks=checks,
                  source_sha256={name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                                 for name in ('model.py', 'dataset.py', 'check.py')})
    (directory / 'check_results.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
