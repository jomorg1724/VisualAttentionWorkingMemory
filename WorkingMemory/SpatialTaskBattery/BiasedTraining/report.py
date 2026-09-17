"""Saved-artifact summary only; no GPU, fitting or additional evaluation."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
def main():
    receipt=json.loads((HERE/'retrieval_receipt.json').read_text());root=Path(receipt['results']);a=json.loads((root/'aggregate.json').read_text());lines=['# Five-task training with retained attention biases','',f"Status: {a['status']}.",'','The original trained source and locality terms were retained, together with every compatible parent weight and Adam state. Five new semantic heads use the same initialization as the cancelled bias-free arm. That arm was cancelled before a comparable exposure or evaluation and is not an equal-training comparator.','']
    for name,r in a['runs'].items():
        stop=r.get('user_stop')
        if stop:
            lines += [f"Stopped at the user's request after {stop['logged_added_updates']:,} logged updates / {stop['logged_added_episodes']:,} episodes (global step {stop['logged_global_step']}). The latest durable checkpoint is step {stop['durable_global_step']}, preserving {stop['durable_added_updates']:,} updates / {stop['durable_added_episodes']:,} episodes. The last {stop['uncheckpointed_logged_updates']} logged updates were not checkpointed. No final held-out evaluation was run.",'',f"Validation-selected so far: global step {r.get('selected_step')}. The table below is the last completed validation, not a final held-out result.",'']
            if r.get('validation'):
                v=r['validation'][-1]
                lines += ['| Last validation condition | BA | Accuracy | AUC |','|---|---:|---:|---:|']
                for c in v['cells'].values():
                    m=c['overall'];fmt=lambda x:'undefined' if x is None else f'{100*x:.2f}%';lines.append(f"| {c['condition']} | {fmt(m['balanced_accuracy'])} | {fmt(m['accuracy'])} | {m['macro_ovr_auc']} |")
                lines.append('')
        else:
            lines += [f"Selected global step: {r.get('selected_step')}. Recorded training episodes: {r.get('training',{}).get('counts',{}).get('episodes','unavailable')}.",'']
        for stage in ('selected_test','terminal_test'):
            if stage not in r:continue
            lines += [f'## {stage}','','| Condition | BA | Accuracy | AUC |','|---|---:|---:|---:|']
            for c in r[stage]['cells'].values():
                m=c['overall'];fmt=lambda x:'undefined' if x is None else f'{100*x:.2f}%';lines.append(f"| {c['condition']} | {fmt(m['balanced_accuracy'])} | {fmt(m['accuracy'])} | {m['macro_ovr_auc']} |")
            lines.append('')
    lines += ['Recognition load0 is a specificity/accuracy control with no positive class; BA/AUC are undefined and excluded from selection. All class confusions, predictions, validation trajectories, gradient logs and per-head attention summaries are preserved. One training continuation does not establish biological plausibility or population-level model superiority.']
    (Path(receipt['archive']).parent/'report.md').write_text('\n'.join(lines));print('Saved report.md')
if __name__=='__main__':main()
