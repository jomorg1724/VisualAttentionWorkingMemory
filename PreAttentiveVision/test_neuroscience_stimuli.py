"""Focused direction-label, stream-resume and available-motion checks."""
import numpy as np
import torch
from .neuroscience_stimuli import TaskStream, DIRECTION_VECTORS


def test_motion():
    stream = TaskStream(66201, 'val')
    state = stream.state_dict()
    x, y, meta = stream.batch(48)
    assert x.shape == (48, 2, 3, 100, 100) and x.dtype == torch.float32
    assert torch.isfinite(x).all() and x.min() >= 0 and x.max() <= 1
    assert torch.bincount(y).tolist() == [12, 12, 12, 12]
    assert all(m['coherence'] == 1 and m['survivor_count_domain'] == 128 for m in meta)
    stream.load_state_dict(state)
    xx, yy, mm = stream.batch(48)
    assert torch.equal(x, xx) and torch.equal(y, yy) and meta == mm
    # Renderers must preserve cardinal correspondence; this is a diagnostic of
    # task observability, not a claim about a trained encoder's performance.
    predicted = []
    for pair, m in zip(x, meta):
        a, b = pair[:, 0].numpy()
        step = int(m['displacement_pixels'])
        scores = []
        for dx, dy in DIRECTION_VECTORS.astype(int) * step:
            shifted = np.roll(a, (dy,dx), (0,1))
            scores.append(np.mean((shifted[12:88,12:88]-.2) * (b[12:88,12:88]-.2)))
        predicted.append(int(np.argmax(scores)))
    accuracy = float(np.mean(np.array(predicted) == y.numpy()))
    assert accuracy >= .9, accuracy
    return {'trials':48, 'direction_counts':[12,12,12,12], 'rendered_cross_correlation_accuracy':accuracy, 'exact_stream_resume':True}


if __name__ == '__main__':
    import json
    from pathlib import Path
    result = test_motion()
    result['command'] = 'C:/Python310/python.exe -X utf8 -B -m PreAttentiveVision.test_neuroscience_stimuli'
    result['status'] = 'passed'
    Path(__file__).with_name('cardinal_motion_check.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))
