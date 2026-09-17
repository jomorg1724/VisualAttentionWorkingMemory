"""Read-only trajectory comparison after local completion; never launches GPU work."""
import json,time,subprocess,shlex,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;CLOUD=HERE.parent/'BiasedTraining';ROOT=HERE.parents[2]
def read(p):return json.loads(Path(p).read_text())
def cloud_motion():
    receipt=CLOUD/'retrieval_receipt.json'
    if receipt.exists():
        a=read(Path(read(receipt)['results'])/'aggregate.json');return extract(a)
    cfg=read(CLOUD/'cloud_provisioning.json')
    code="import json; a=json.load(open("+repr(cfg['remote_results']+'/aggregate.json')+")); r=a['runs']['spatial_biased']; print(json.dumps([dict(step=e['step'],cells={c['condition']:c['overall']['balanced_accuracy'] for c in e['cells'].values() if c['family']=='motion_duration_cued'}) for e in r.get('validation',[]) if e['step']>=10000]))"
    command=['ssh','-n','-i',cfg['ssh_identity'],'-o','UserKnownHostsFile='+cfg['known_hosts'],'-o','BatchMode=yes','-o','ConnectTimeout=10','-p',str(cfg['ssh_port']),'root@'+cfg['ssh_host'],'python -c '+shlex.quote(code)]
    return json.loads(subprocess.run(command,capture_output=True,text=True,timeout=30,check=True).stdout)
def extract(a):
    return [dict(step=e['step'],cells={c['condition']:c['overall']['balanced_accuracy'] for c in e['cells'].values() if c['family']=='motion_duration_cued'}) for e in a['runs']['spatial_biased'].get('validation',[]) if e['step']>=10000]
def compare():
    run=Path(read(HERE/'local_run.json')['run']);a=read(run/'aggregate.json');cloud=cloud_motion();records=[]
    for v in [a['parent_validation']]+a['validation']:records.append(dict(arm='Local motion only',step=v['step'],added_motion_episodes=(v['step']-10000)*8,cells={n:c['balanced_accuracy'] for n,c in v['cells'].items()}))
    for v in cloud:records.append(dict(arm='Cloud five tasks',step=v['step'],added_motion_episodes=(v['step']-10000)*8,cells={n.rsplit('_',1)[-1]:x for n,x in v['cells'].items()}))
    lines=['# Motion validation trajectories','', 'Both branches start at the same checkpoint10000. Each optimizer update exposes eight motion episodes, so the horizontal variable below is added motion episodes, not total five-task episodes. Validation draws differ between branches; these are descriptive trajectories, not paired accuracy differences.','', '| Branch | Step | Added motion episodes | D0 | D4 | D12 | D24 |','|---|---:|---:|---:|---:|---:|---:|']
    for r in sorted(records,key=lambda r:(r['added_motion_episodes'],r['arm'])):lines.append('| '+r['arm']+f" | {r['step']} | {r['added_motion_episodes']} | "+' | '.join(f"{100*r['cells'][n]:.2f}%" for n in ('D0','D4','D12','D24'))+' |')
    lines += ['','The local branch uses full motion CE; the cloud averages five task losses and combines their gradients before clipping/Adam. These results cannot isolate gradient interference. Cloud points beyond the local exposure range are outside the matched exposure overlap. Missing later cloud checkpoints may still be training; this report reads only completed scheduled validations. The local final parent/selected/terminal test remains the paired evidence for local acquisition.']
    (HERE/'trajectory_comparison.md').write_text('\n'.join(lines));(HERE/'trajectory_comparison.json').write_text(json.dumps(dict(observed_unix=time.time(),records=records,local_run=str(run)),indent=2))
def watch():
    cfg=read(HERE/'local_run.json');run=Path(cfg['run'])
    while time.time()<cfg['deadline_unix']+180:
        if (run/'exit.json').exists():
            if read(run/'exit.json')['status']=='completed':compare()
            return
        time.sleep(30)
if __name__=='__main__':watch() if '--watch' in sys.argv else compare()
