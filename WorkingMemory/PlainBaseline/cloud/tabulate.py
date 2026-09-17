"""Tabulate test results from the pulled program receipts. Usage: python -m WorkingMemory.PlainBaseline.cloud.tabulate [run_dir]"""
import json,glob,os,sys
R=sys.argv[1] if len(sys.argv)>1 else json.load(open('WorkingMemory/PlainBaseline/runs/latest_cloud.json'))['run']
rows=[]
for f in sorted(glob.glob(os.path.join(R,'pulled','results','*','program_receipt.json'))):
    j=json.load(open(f))
    for key,v in j['stages'].items():
        lane,stage=key.split('/');arm,seed=lane.rsplit('_s',1)
        cells=v.get('test_terminal',{});rows.append((arm,int(seed),stage,v['status'],v.get('episodes'),{c.split('_')[-1]:m['ba'] for c,m in cells.items()}))
order={'ring':0,'cued':1,'delayA':2,'delayB':3,'delayC':4};rows.sort(key=lambda r:(r[0],r[1],order[r[2]]))
print('| arm | seed | stage | status | test BA per cell (512 trials, seed 64973001) |');print('|---|---:|---|---|---|')
for arm,seed,stage,status,ep,cells in rows:print(f"| {arm} | {seed} | {stage} | {status} | "+' '.join(f'{k}={v:.3f}' for k,v in cells.items())+' |')
print('\nDelay curves (delayC terminal test):');print('| arm | seed | D0 | D4 | D12 | D24 |');print('|---|---:|---:|---:|---:|---:|')
for arm,seed,stage,status,ep,cells in rows:
    if stage=='delayC' and cells:print(f"| {arm} | {seed} | "+' | '.join(f"{cells.get(d,float('nan')):.3f}" for d in ('D0','D4','D12','D24'))+' |')
