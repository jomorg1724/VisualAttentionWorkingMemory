"""CPU-only raw-target assay; deliberately does not open the final test split."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from .dataset import generate_evaluation
from .model import decayed_distance


def auc(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    differences = scores[labels == 1, None] - scores[labels == 0][None, :]
    return float(((differences > 0) + .5 * (differences == 0)).mean())


def main():
    torch.set_num_threads(2)
    started = time.monotonic()
    cells = {}
    # Separate high indices in the training namespace, not validation/test draws.
    for index in range(12000000, 12000000 + 1536):
        first, second, label, metadata = generate_evaluation(index, 'train')
        key = (metadata['speed'], metadata['angle'])
        score = float(decayed_distance(first[None], second[None]))
        cells.setdefault(key, []).append((label, score))
    results = []
    for (speed, angle), rows in sorted(cells.items()):
        labels, scores = np.array(rows).T
        results.append(dict(speed=speed, angle=angle, presentations=len(rows),
                            paired_contexts=len(rows)//2, direct_auc=auc(labels, scores),
                            inverse_auc=auc(labels, -scores),
                            unchanged_mean=float(scores[labels == 0].mean()),
                            changed_mean=float(scores[labels == 1].mean())))
    record = dict(device='cpu', model_inference=False, split='train',
                  start_index=12000000, used_final_test=False, cells=results,
                  seconds=time.monotonic()-started)
    Path(__file__).with_name('raw_pixel_audit.json').write_text(
        json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
