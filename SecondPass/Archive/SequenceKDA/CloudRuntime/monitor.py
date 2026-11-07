"""Read-only exact-pod lifecycle, private status, and persisted scientific data."""
import json
from deploy import ROOT,REMOTE,api,remote_python,save


def inspect():
    code='''import json,time
from pathlib import Path
r=Path(ROOT_VALUE)/'run';out={'observed':time.time()}
for n in ('live_status.json','latest_checkpoint.json','allocation.json','failure.json','cloud_completion.json','supervisor_result.json','owner_run.json','profile/profile.json','persisted_progress_verification.json','first_cycle_timing.json'):
 p=r/n
 if p.exists(): out[n]={k:v for k,v in json.loads(p.read_text()).items() if k!='artifact_manifest'}
p=r/'progress.jsonl';rows=[]
if p.exists():
 for line in p.read_text().splitlines():
  try: rows.append(json.loads(line))
  except ValueError: pass
out['progress']=rows[-1] if rows else None
out['evaluations']=[]
for p in sorted(list(r.glob('validation_*.json'))+list(r.glob('test_*.json'))):
 if 'partial' in p.name: continue
 d=json.loads(p.read_text());d=d.get('results',d)
 out['evaluations'].append({'file':p.name,'complete':d.get('complete'),'summary':d.get('summary'),'cells':d.get('cells')})
print(json.dumps(out))
'''.replace('ROOT_VALUE',repr(REMOTE))
    data=json.loads(remote_python(code));save('remote_status.json',data);return data


def status():
    ident=json.loads((ROOT/'pod.json').read_text())['id'];p=api('/v2/pods/'+ident)
    record=json.loads((ROOT/'status_record_verified.json').read_text());t=api('/v2/templates/'+record['id'])
    if t['public'] is not False: raise ValueError('Status record no longer private')
    out=dict(pod={k:p.get(k) for k in ('id','name','status','cost')},public_status=json.loads(t['env']['VAWM_PUBLIC_STATUS']),disk_retained=True)
    if p['status']=='RUNNING':
        try: out['remote']=inspect()
        except Exception as e: out['remote_unavailable']=type(e).__name__
    else: out['retrieval']='No automatic restart; stopped disk recovery needs separately authorized zero-GPU retrieval'
    save('status.json',out);return out

if __name__=='__main__': print(json.dumps(status(),indent=2))
