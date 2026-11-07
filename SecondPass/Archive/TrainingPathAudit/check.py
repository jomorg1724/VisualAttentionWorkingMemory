"""CPU-only targeted loss/decoder/temporal-gradient diagnostic of saved models.

No deployed weights, optimizers, samplers, source files or accelerator jobs change.
One native B12 movie/model; these gradient norms are descriptive, not causal tests.
"""
import contextlib
import copy
import io
import json
import math
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from SecondPass.TwoFrameRViTReplay import worker as replay
from SecondPass.SequenceKDA16 import worker as kda_worker
from SecondPass.TwoFrameRViT.model import TwoFrameRViT
from SecondPass.SequenceKDA16.model import SequenceKDA
from SecondPass.StructuredMotionRViT.model import StructuredMotionRViT

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).parent


def losses_and_accumulation():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 2)
        def forward(self, images, task=None):
            return self.linear(images.flatten(1))

    class Stream:
        pool_index = 0
        epoch = 0
        cursor = 1
        generation_seconds = 0.
        def __init__(self, x, y):
            self.x, self.y, self.used, self.offset = x, y, set(), 0
        def current_batch(self):
            return self.x, self.y, 'B12', list(range(len(self.y)))
        def commit(self, cell, indices):
            self.used.update((cell, i) for i in indices)
        def batch(self, n, task, cell):
            i = self.offset
            self.offset += n
            return self.x[i:i+n], self.y[i:i+n], [{}] * n

    results = []
    for policy, update, counts, micros in (
        ('replay', replay.Session.train_update, (13, 14, 32), (1, 4)),
        ('online_KDA', kda_worker.Session.train_update, (32,), (4,)),
    ):
        for n in counts:
            for micro in micros:
                torch.manual_seed(7)
                x, y = torch.randn(n, 1, 3, 1, 1), torch.arange(n) % 2
                a, b = Toy(), Toy()
                b.load_state_dict(a.state_dict())
                opt_a = torch.optim.Adam(a.parameters(), lr=1e-4)
                opt_b = torch.optim.Adam(b.parameters(), lr=1e-4)
                logits = b(x)
                logits.retain_grad()
                expected_loss = -logits.log_softmax(1).gather(1, y[:, None]).mean()
                expected_loss.backward()
                closed_grad = (logits.detach().softmax(1) - F.one_hot(y, 2)) / n
                torch.testing.assert_close(logits.grad, closed_grad, atol=2e-8, rtol=2e-6)
                expected_grads = [p.grad.clone() for p in b.parameters()]
                opt_b.step()
                with tempfile.TemporaryDirectory() as directory:
                    session = SimpleNamespace(model=a, optimizer=opt_a, device='cpu',
                        directory=Path(directory), config=dict(microbatch=micro, effective_batch=32,
                        disposable_profile=True), stream=Stream(x, y),
                        state=dict(step=0, episodes=0, frames=0, optimizer_seconds=0.,
                        exposure={replay.TASK:dict(updates=0, episodes=0, frames=0, cells={c:0 for c in replay.CELLS})}))
                    with contextlib.redirect_stdout(io.StringIO()):
                        row = update(session, replay.TASK, 'B12')
                assert abs(row['loss'] - expected_loss.item()) < 2e-7
                for p, q, expected in zip(a.parameters(), b.parameters(), expected_grads):
                    torch.testing.assert_close(p.grad, expected, atol=2e-7, rtol=3e-6)
                    torch.testing.assert_close(p, q, atol=2e-8, rtol=2e-6)
                results.append(dict(policy=policy, actual_batch=n, microbatch=micro,
                    loss_gradient_and_adam_match_full_batch=True))
    return results


def audit_model(name, checkpoint, constructor):
    started = time.monotonic()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = constructor()
    model.load_state_dict(saved['model'])
    model.train()
    stream = replay.base.FreshStream('val')
    images, labels, metadata = stream.batch(1, replay.TASK, 'B12')
    assert int(labels[0]) == int(metadata[0]['event_type'] == 'target')
    images.requires_grad_()
    states, projections, handles = [], {}, []
    if hasattr(model, 'recurrent_block'):
        def state_hook(module, args, output):
            output.retain_grad()
            states.append(output)
        handles.append(model.recurrent_block.register_forward_hook(state_hook))
    else:
        for kind in ('q', 'k', 'v'):
            def capture(module, args, output, kind=kind):
                output.retain_grad()
                projections[kind] = output
            handles.append(getattr(model.kda, kind + '_proj').register_forward_hook(capture))
    logits = model(images)
    logits.retain_grad()
    loss = F.cross_entropy(logits, labels)
    hand_loss = -logits.log_softmax(1)[0, int(labels[0])]
    torch.testing.assert_close(loss, hand_loss, rtol=0, atol=0)
    loss.backward()
    closed_grad = logits.detach().softmax(1) - F.one_hot(labels, 2)
    torch.testing.assert_close(logits.grad, closed_grad, atol=2e-8, rtol=2e-6)
    parameter_gradients = {n: float(p.grad.norm()) if p.grad is not None else None
                           for n, p in model.named_parameters()}
    assert all(g is not None and math.isfinite(g) for g in parameter_gradients.values())
    # Measure FP32 gradients in FP64 so squaring tiny values does not underflow.
    raw = images.grad.double().flatten(2).norm(dim=2)[0].tolist()
    result = dict(model=name, checkpoint=str(checkpoint), checkpoint_step=saved['state']['step'],
        native_frames=images.shape[1], event=metadata[0]['event_type'], label=int(labels[0]),
        logits=logits.detach().tolist(), loss=loss.item(),
        loss_matches_manual_log_softmax=True, logit_gradient_matches_closed_form=True,
        parameter_tensors=len(parameter_gradients), all_parameter_gradients_present_finite=True,
        zero_parameter_gradient_tensors=[n for n, g in parameter_gradients.items() if g == 0.],
        parameter_gradient_norms=parameter_gradients, input_frame_gradient_l2=raw,
        cue_gradient_nonzero_entries=[int(torch.count_nonzero(images.grad[:, i])) for i in (0, 1)],
        cue_gradient_l2=raw[:2], final_frame_gradient_l2=raw[-1],
        cue_to_final_gradient_ratio=max(raw[:2])/raw[-1] if raw[-1] else None)
    if states:
        memory = [float(state.grad.double().norm()) for state in states]
        result.update(memory_state_gradient_l2=memory,
                      first_memory_to_final_ratio=memory[0]/memory[-1] if memory[-1] else None)
    if projections:
        for kind, tensor in projections.items():
            g = tensor.grad
            result[kind + '_patch_gradient_l2_by_frame'] = g[:, :-1].double().reshape(1, 29, 100, -1).flatten(2).norm(dim=2)[0].tolist()
            result[kind + '_cls_gradient_l2'] = float(g[:, -1].norm())
    for handle in handles:
        handle.remove()
    result['seconds'] = time.monotonic() - started
    return result


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    started = time.monotonic()
    result = dict(device='cpu', threads=2, weights_unchanged=True,
        scope='seven exact-update accumulation comparisons and one native B12 backward per saved model',
        accumulation=losses_and_accumulation(), models=[])
    candidates = [
        ('KDA16', ROOT/'SecondPass/SequenceKDA16/CloudRuntime/artifacts/checkpoint_004100.pt', SequenceKDA),
        ('RViT replay', ROOT/'SecondPass/TwoFrameRViTReplay/CloudRuntime/artifacts/checkpoint_001500.pt', lambda:TwoFrameRViT(checkpoint_encoder=False)),
        ('Motion RViT', Path('/Users/jonathanmorgan/VAWMRuntime/structured_motion_rvit_local01/run'), lambda:StructuredMotionRViT(checkpoint_encoder=False)),
    ]
    for name, checkpoint, constructor in candidates:
        if checkpoint.is_dir():
            checkpoint = Path(json.loads((checkpoint/'latest_checkpoint.json').read_text())['path'])
        item = audit_model(name, checkpoint, constructor)
        result['models'].append(item)
        (OUTPUT/'results.partial.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k:item[k] for k in ('model','checkpoint_step','loss','cue_to_final_gradient_ratio','zero_parameter_gradient_tensors','seconds')}),flush=True)
    result['seconds'] = time.monotonic() - started
    result['complete'] = True
    (OUTPUT/'results.json').write_text(json.dumps(result, indent=2)+'\n')

if __name__ == '__main__':
    main()
