"""Collect receipts from a rung directory into one Markdown table.

Usage: python -m WorkingMemory.PlainBaseline.summarize <run_dir>  (writes <run_dir>/summary.md)
"""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np

def main():
    root=Path(sys.argv[1]);rows=[];curves=[]
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        rp=d/'receipt.json'
        if not rp.exists():continue
        r=json.loads(rp.read_text());status='completed' if 'final' in r else 'incomplete'
        hist=json.loads((d/'validation.json').read_text()) if (d/'validation.json').exists() else []
        if status=='completed':
            for tag in ('best','terminal'):
                for cell,m in r['final'][tag]['test'].items():
                    rows.append(dict(task=r['task'],cell=cell,tag=tag,step=r['final'][tag]['step'],episodes=r['final'][tag]['step']*r['args']['batch'],ba=m['balanced_accuracy'],chance=m['chance'],auc=m['macro_ovr_auc'],n=m['n'],confusion=m['confusion']))
        curves.append(dict(task=r['task'],status=status,points=[(h['update'],h['episodes'],h['mean_ba']) for h in hist],seconds=r.get('seconds'),params=r['params']))
    L=['# Rung 1 summary: plain baseline, one task per run\n',f'Run directory `{root.as_posix()}`. Test seed 64973001, 512 trials per cell, fixed argmax. BA = balanced accuracy; normalised BA = (BA - chance)/(1 - chance).\n',
       '| task | cell | checkpoint | update | episodes | BA | chance | normalised BA | AUC | n | confusion |','|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for x in rows:L.append(f"| {x['task']} | {x['cell']} | {x['tag']} | {x['step']} | {x['episodes']} | {x['ba']:.3f} | {x['chance']:.2f} | {(x['ba']-x['chance'])/(1-x['chance']):.3f} | {x['auc']:.3f} | {x['n']} | {x['confusion']} |")
    L.append('\n## Validation curves (mean BA over cells, 256 trials per cell, val seed 63973001)\n')
    for c in curves:
        secs=f"{c['seconds']:.0f} s" if c['seconds'] is not None else 'running/killed'
        L.append(f"**{c['task']}** ({c['status']}, {c['params']} params, {secs}): "+', '.join(f'{u}:{b:.3f}' for u,e,b in c['points']))
    (root/'summary.md').write_text('\n'.join(L),encoding='utf-8');print('\n'.join(L))

if __name__=='__main__':main()
