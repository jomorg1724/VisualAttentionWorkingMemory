"""One focused check of the two-frame protocol, stream replay and split law."""
import numpy as np
import torch
from .stimuli import PairStream, FAMILIES


def test_pair_protocol():
    stream = PairStream(819230, 'val')
    for family in FAMILIES:
        before = stream.state_dict()
        x, y, metadata = stream.batch(16, family=family)
        assert x.shape == (16, 2, 3, 100, 100) and x.dtype == torch.float32
        assert y.dtype == torch.int64 and int(y.sum()) == 8
        assert torch.isfinite(x).all() and float(x.min()) >= 0 and float(x.max()) <= 1
        # A/B variant choice supplies both marginals for either label. The
        # bitwise label/variant relation is the explicit matched-marginal law.
        assert all((m['variant_pair'][0] != m['variant_pair'][1]) == bool(label)
                   for label, m in zip(y, metadata))
        assert all(not torch.equal(pair[0], pair[1]) for pair in x[y == 0])
        assert all(m['magnitude'] > 0 for m in metadata)  # edit drawn even for negatives
        stream.load_state_dict(before)
        xx, yy, mm = stream.batch(16, family=family)
        assert torch.equal(x, xx) and torch.equal(y, yy) and metadata == mm
        if family == 'natural':
            assert all(45000 <= int(m['base_id'].rsplit('/', 1)[1]) < 50000 for m in metadata)
    for split in ('train', 'test'):
        _, _, metadata = PairStream(51, split).batch(4, 'natural')
        if split == 'train':
            assert all(m['base_id'].startswith('cifar10/train/') and int(m['base_id'].rsplit('/', 1)[1]) < 45000 for m in metadata)
        else:
            assert all(m['base_id'].startswith('cifar10/test/') for m in metadata)


if __name__ == '__main__':
    test_pair_protocol()
    print('PASS: shapes, finite values, balanced labels, matched-marginal variant law, independent nuisance, exact stream replay, natural split IDs')
