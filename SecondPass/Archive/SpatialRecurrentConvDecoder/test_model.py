"""Bounded CPU tests on generated tensors; never open trained checkpoints."""
import torch

from PreAttentiveVision.decoder import ConvNormAct
from SecondPass.SpatialRecurrentConvDecoder.model import (
    SpatialRecurrentConvDecoder, ConvTransformerBlock, grid, tokens,
)
from SecondPass.TaskSuite.suite import task_classes


torch.set_num_threads(2)


def test_no_cls_recurrence_has_49_spatial_tokens():
    torch.manual_seed(73)
    model = SpatialRecurrentConvDecoder({'probe': 2})
    visual = torch.randn(2, 64, 7, 7)
    hidden = model.memory(visual)
    assert hidden.shape == (2, 49, 64)
    assert not any('cls' in name for name, _ in model.named_parameters())
    assert all(p.requires_grad for p in model.parameters())
    assert len(model.memory.layers) == 2
    assert all(layer.position.shape == (1, 49, 64) for layer in model.memory.layers)
    assert model.classify(hidden, 'probe').shape == (2, 2)


def test_attention_is_49_queries_over_98_joint_source_keys():
    torch.manual_seed(74)
    block = ConvTransformerBlock()
    hidden = torch.randn(2, 49, 64)
    visual = torch.randn(2, 64, 7, 7)
    h = grid(block.memory_norm(hidden) + block.position + block.source[1])
    z = grid(block.visual_norm(tokens(visual)) + block.position + block.source[0])
    q = tokens(block.spatial_q(h)).reshape(2, 49, 2, 32).transpose(1, 2)
    k = torch.cat([tokens(block.spatial_k(x)) for x in (z, h)], 1)
    v = torch.cat([tokens(block.spatial_v(x)) for x in (z, h)], 1)
    k = k.reshape(2, 98, 2, 32).transpose(1, 2)
    v = v.reshape(2, 98, 2, 32).transpose(1, 2)
    # Independent per-head expression checks the joint softmax/residual route.
    attended = torch.stack([
        ((q[:, i] @ k[:, i].transpose(1, 2)) / (32 ** .5)).softmax(-1) @ v[:, i]
        for i in range(2)
    ], 2).reshape(2, 49, 64)
    residual = hidden + tokens(block.spatial_out(grid(attended)))
    expected = residual + tokens(block.spatial_ffn(grid(block.ffn_norm(residual))))
    torch.testing.assert_close(block(hidden, visual), expected)


def test_early_frame_bptt_finite_gradients_and_decoder_update():
    torch.manual_seed(75)
    model = SpatialRecurrentConvDecoder({'probe': 2})
    # Five steps put frame zero outside the last raw stack-3 window.
    images = torch.rand(1, 5, 3, 100, 100, requires_grad=True)
    before = {n: p.detach().clone() for n, p in model.readout.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss = torch.nn.functional.cross_entropy(model(images, 'probe'), torch.tensor([1]))
    loss.backward()
    assert torch.isfinite(images.grad).all()
    assert images.grad[:, 0].abs().sum() > 0
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    optimizer.step()  # One disposable synthetic update, not an experiment.
    for name, parameter in model.readout.named_parameters():
        assert not torch.equal(before[name], parameter), name
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_causal_prefix_carry_and_sequence_reset():
    torch.manual_seed(76)
    model = SpatialRecurrentConvDecoder({'probe': 2}).eval()
    images = torch.rand(1, 4, 3, 100, 100)
    changed = images.clone()
    changed[:, 2:] = 0
    with torch.no_grad():
        history = model.recurrent_states(images)
        prefix = model.recurrent_states(images[:, :2])
        other = model.recurrent_states(changed)
        for step in range(2):
            assert torch.equal(history[step], prefix[step])
            assert torch.equal(history[step], other[step])
        assert not torch.equal(history[-1], other[-1])
        visual = torch.randn(1, 64, 7, 7)
        assert not torch.equal(model.memory(visual, history[0]), model.memory(visual))
        # Separate calls, including an intervening unrelated sequence, reset
        # both the KDA states and transformer H (no hidden module cache).
        expected = model(images, 'probe')
        model(changed, 'probe')
        assert torch.equal(expected, model(images, 'probe'))
        frames = model.frames(images)
        encoder_states = hidden = None
        for step in range(frames.shape[1]):
            field, encoder_states = model.encode_frame(frames[:, step], encoder_states)
            hidden = model.memory(model.spatial_input(field), hidden)
            assert torch.equal(hidden, history[step])


def test_classification_uses_only_final_updated_spatial_field():
    torch.manual_seed(77)
    model = SpatialRecurrentConvDecoder({'probe': 2}).eval()
    fixed_hidden = torch.randn(1, 49, 64)
    observed = []
    count = 0

    def fix_final(_module, _inputs, output):
        nonlocal count
        count += 1
        return fixed_hidden if count % 3 == 0 else output

    def record_field(_module, inputs):
        observed.append(inputs[0].detach().clone())

    memory_hook = model.memory.register_forward_hook(fix_final)
    readout_hook = model.readout.register_forward_pre_hook(record_field)
    try:
        images = torch.rand(1, 3, 3, 100, 100, requires_grad=True)
        first = model(images, 'probe')
        second = model(torch.zeros_like(images), 'probe')
        assert torch.equal(first, second)
        first.sum().backward()
        assert images.grad is None  # No sensory/earlier-state bypass.
        assert len(observed) == 2
        assert all(torch.equal(field, grid(fixed_hidden)) for field in observed)
        assert torch.equal(first, model.classify(fixed_hidden, 'probe'))
        altered = fixed_hidden.clone()
        altered[:, 15:30] = 0
        assert not torch.equal(first, model.classify(altered, 'probe'))
    finally:
        memory_hook.remove()
        readout_hook.remove()


def test_conv_before_pooling_and_all_native_task_heads():
    torch.manual_seed(78)
    classes = task_classes()
    assert len(classes) == 13
    model = SpatialRecurrentConvDecoder(classes)
    assert isinstance(model.readout.local, ConvNormAct)
    assert isinstance(model.readout.fusion, ConvNormAct)
    assert not any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    shapes = []
    handles = [module.register_forward_hook(
        lambda _module, inputs, output: shapes.append((inputs[0].shape, output.shape)))
        for module in (model.readout.local, model.readout.fusion, model.readout.trunk)]
    try:
        hidden = model.recurrent_states(torch.rand(2, 2, 3, 100, 100))[-1]
        features = model.readout(grid(hidden))
    finally:
        for handle in handles:
            handle.remove()
    assert shapes == [((2, 64, 7, 7), (2, 32, 7, 7)),
                      ((2, 32, 7, 7), (2, 64, 7, 7)),
                      ((2, 128), (2, 256))]
    field = model.readout.fusion(model.readout.local(grid(hidden)))
    expected = model.readout.trunk(torch.cat((field.mean((2, 3)), field.amax((2, 3))), 1))
    assert torch.equal(features, expected)
    assert set(model.heads) == set(classes)
    for task, count in classes.items():
        assert model.heads[task].in_features == 256
        logits = model.classify(hidden, task)
        assert logits.shape == (2, count)
        assert torch.isfinite(logits).all()
        assert torch.equal(logits, model.heads[task](features))
