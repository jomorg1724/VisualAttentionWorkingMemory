"""Sequential program: for each (arm, seed), gate -> curriculum -> delay ladder, all on the orientation family.

Stages (each a `baseline.py` run; a stage is skipped if its receipt already has a final result, so the program resumes):
  ring   : orientation_ring from scratch, 100k episodes.   Gate: terminal test BA >= 0.90 or the arm stops here.
  cued   : orientation_cued D0 from the ring terminal, 100k.
  delayA : orientation_cued cells D0,D1,D2 from cued, 100k.
  delayB : cells D0,D2,D4 from delayA, 100k.
  delayC : cells D0,D4,D12,D24 from delayB, 200k.  Its test at all four delays is the arm's delay curve.
Arms: plain (PlainBaseline), and the accumulator encoder with convgru / opponent / kda. Same recipe for all:
centred input, stack 3, lr 1e-4, batch 64, no clipping. Everything trainable from scratch.

Usage: python -m WorkingMemory.PlainBaseline.program --out <dir> --arms plain,convgru,opponent,kda --seeds 1,2 --deadline <unix> [--workers 3] [--chunk 32] [--scale 1.0]
Writes <out>/program_receipt.json after every stage and <out>/summary.md at the end.
"""
from __future__ import annotations
import argparse,json,subprocess,sys,time
from pathlib import Path

STAGES=[('ring',dict(task='orientation_ring',cells='rung1',episodes=100000,init=None)),
        ('cued',dict(task='orientation_cued',cells='D0',episodes=100000,init='ring')),
        ('delayA',dict(task='orientation_cued',cells='D0,D1,D2',episodes=100000,init='cued')),
        ('delayB',dict(task='orientation_cued',cells='D0,D2,D4',episodes=100000,init='delayA')),
        ('delayC',dict(task='orientation_cued',cells='D0,D4,D12,D24',episodes=200000,init='delayB'))]
GATE=0.90

def arm_flags(arm):
    return ['--encoder','plain'] if arm=='plain' else ['--encoder','accum','--accumulator',arm]

def stage_result(d):
    r=d/'receipt.json'
    if not r.exists():return None
    j=json.loads(r.read_text())
    return j if 'final' in j else None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--arms',default='plain,convgru,opponent,kda');ap.add_argument('--seeds',default='1,2')
    ap.add_argument('--deadline',type=float,required=True,help='unix time; no new stage starts within 20 min of it');ap.add_argument('--workers',type=int,default=3);ap.add_argument('--chunk',type=int,default=32)
    ap.add_argument('--scale',type=float,default=1.0,help='multiply every stage episode count (smoke tests)');ap.add_argument('--gate',type=float,default=GATE);ap.add_argument('--val-n',type=int,default=256);ap.add_argument('--test-n',type=int,default=512);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);receipt_path=out/'program_receipt.json'
    receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else dict(started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),args=vars(a),stages={})
    def save(status=None):
        if status:receipt['status']=status
        receipt['updated']=time.strftime('%Y-%m-%dT%H:%M:%S%z');receipt_path.write_text(json.dumps(receipt,indent=1))
    for arm in a.arms.split(','):
        for seed in [int(s) for s in a.seeds.split(',')]:
            lane=f'{arm}_s{seed}';lane_dir=out/lane;lane_dir.mkdir(exist_ok=True);prev={}
            for name,spec in STAGES:
                key=f'{lane}/{name}';d=lane_dir/name
                done=stage_result(d)
                if done is None:
                    if time.time()>a.deadline-1200:receipt['stages'][key]=dict(status='not_started_deadline');save('deadline');continue
                    cmd=[sys.executable,'-B','-m','WorkingMemory.PlainBaseline.baseline','--task',spec['task'],'--cells',spec['cells'],'--out',str(d),'--episodes',str(int(spec['episodes']*a.scale)),
                         '--workers',str(a.workers),'--chunk',str(a.chunk),'--stack','3','--center','--lr','0.0001','--seed',str(seed),'--val-n',str(a.val_n),'--test-n',str(a.test_n),*arm_flags(arm)]
                    if spec['init']:
                        parent=lane_dir/spec['init']/'terminal.pt'
                        if not parent.exists():receipt['stages'][key]=dict(status='skipped_no_parent');save();continue
                        cmd+=['--init',str(parent)]
                    t=time.time();d.mkdir(parents=True,exist_ok=True);log=open(d/'stdout.log','a')
                    proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=str(Path(__file__).resolve().parents[2]));log.close()
                    if proc.returncode:receipt['stages'][key]=dict(status='failed',returncode=proc.returncode,seconds=time.time()-t);save();break
                    done=stage_result(d)
                    if done is None:receipt['stages'][key]=dict(status='no_receipt');save();break
                test={c:dict(ba=m['balanced_accuracy'],auc=m['macro_ovr_auc']) for c,m in done['final']['terminal']['test'].items()}
                receipt['stages'][key]=dict(status='completed',episodes=done['episodes'],seconds=done.get('seconds'),test_terminal=test,best_val=done['best_val'],params=done['params'],init=done.get('init'));save()
                if name=='ring' and min(v['ba'] for v in test.values())<a.gate:receipt['stages'][key]['status']='completed_gate_failed';save();break
    # Summary table.
    L=['# Accumulator program summary\n','| lane | stage | status | episodes | test BA per cell |','|---|---|---|---:|---|']
    for key,v in receipt['stages'].items():
        cells=' '.join(f"{c.split('_')[-1]}={m['ba']:.3f}" for c,m in v.get('test_terminal',{}).items())
        L.append(f"| {key.split('/')[0]} | {key.split('/')[1]} | {v['status']} | {v.get('episodes','')} | {cells} |")
    (out/'summary.md').write_text('\n'.join(L));save(receipt.get('status') or 'completed');print('\n'.join(L))

if __name__=='__main__':main()
