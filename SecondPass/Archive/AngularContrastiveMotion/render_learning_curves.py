"""Plot saved training/validation results without running the model."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(run, output):
    rows = [json.loads(line) for line in (run / 'progress.jsonl').read_text().splitlines()]
    validations = [json.loads(p.read_text()) for p in sorted(run.glob('validation_*.json'))]
    steps = np.array([r['step'] for r in rows])
    loss = np.array([r['loss'] for r in rows])
    window = min(256, len(rows))
    rolling = np.convolve(loss, np.ones(window) / window, 'valid')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    axes[0].plot(steps[window-1:], rolling, label='Training (256-update mean)', linewidth=1.5)
    axes[0].plot([v['step'] for v in validations], [v['angular_loss'] for v in validations],
                 '.-', label='Fixed angular validation', linewidth=1)
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Optimizer updates')
    axes[0].set_ylabel('Angular spring loss')
    axes[0].legend(fontsize=8)
    axes[0].set_title('Direction-similarity learning')
    axes[1].plot([v['step'] for v in validations], [v['mean_balanced_accuracy'] for v in validations],
                 '.-', label='Balanced accuracy')
    axes[1].plot([v['step'] for v in validations], [v['mean_auc'] for v in validations],
                 '.-', label='AUC')
    axes[1].axhline(.5, color='gray', linestyle=':', linewidth=1)
    axes[1].set_ylim(.45, 1.025)
    axes[1].set_xlabel('Optimizer updates')
    axes[1].set_ylabel('Mean six-cell validation score')
    axes[1].legend(fontsize=8)
    axes[1].set_title('Simplified before/after comparison')
    for ax in axes:
        ax.grid(alpha=.2)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / 'learning_curves.png', dpi=180)
    fig.savefig(output / 'learning_curves.svg')
    plt.close(fig)


if __name__ == '__main__':
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=base / 'LocalRuntime/run')
    parser.add_argument('--output', type=Path, default=base / 'figures')
    args = parser.parse_args()
    main(args.run, args.output)
