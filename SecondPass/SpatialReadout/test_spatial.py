"""CPU behavior tests for the final-spatial-recurrence experiment."""
import importlib.util
import torch
import pytest

torch.set_num_threads(2)


def model_module():
    name = 'SecondPass.SpatialReadout.model'
    assert importlib.util.find_spec(name) is not None, 'Final spatial recurrence is not implemented'
    return __import__(name, fromlist=['SpatialReadout'])


def test_final_spatial_recurrence_shape_causality_and_gradient():
    module = model_module()
    model = module.SpatialReadout({'example': 2})
    assert len(model.acc) == 3 and not hasattr(model, 'gru') and not hasattr(model, 'feat')
    assert model.spatial_input.weight.shape == (64, 160, 1, 1)
    assert model.readout.weight.shape == (256, 3136)
    assert all(p.requires_grad for p in model.parameters())
    x = torch.rand(1, 3, 3, 100, 100, requires_grad=True)
    states = model.recurrent_states(x)
    assert len(states) == 3 and all(h.shape == (1, 64, 7, 7) for h in states)
    altered = x.detach().clone(); altered[:, 2] = 0
    other = model.recurrent_states(altered)
    assert torch.equal(states[0], other[0]) and torch.equal(states[1], other[1])
    assert torch.equal(model(x, 'example'), model.heads['example'](model.readout(states[-1].flatten(1)).relu()))
    states[-1].square().mean().backward()
    assert x.grad[:, 0].abs().sum() > 0
    assert model.spatial_gru.candidate.weight.grad.abs().sum() > 0
    assert model.acc[0].inputs.weight.grad.abs().sum() > 0
    assert torch.equal(model.recurrent_states(x.detach())[0], model.recurrent_states(x.detach())[0])


def test_convgru_equation_zero_bias_and_spatial_hidden_gradient():
    cell = model_module().FinalConvGRU()
    assert torch.count_nonzero(cell.gates.bias) == 0
    assert torch.count_nonzero(cell.candidate.bias) == 0
    x = torch.randn(2,64,7,7, requires_grad=True)
    h = torch.randn_like(x, requires_grad=True)
    write, reset = cell.gates(torch.cat((x,h),1)).sigmoid().chunk(2,1)
    expected = (1-write)*h + write*cell.candidate(torch.cat((x,reset*h),1)).tanh()
    assert torch.equal(cell(x,h), expected)
    assert torch.equal(cell(x), cell(x,torch.zeros_like(x)))
    cell(x,h).sum().backward()
    assert h.grad.abs().sum() > 0
