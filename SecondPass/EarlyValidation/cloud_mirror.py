"""Bounded parent mirror for one CPU snapshot; never rents or restarts a pod."""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('cloud_runtime',type=Path)
    args=parser.parse_args()
    root=args.cloud_runtime.resolve()
    sys.path.insert(0,str(root))
    import deploy
    start=json.loads((root/'early_validation_started.json').read_text())
    out=root/'early_validation';out.mkdir(exist_ok=True)
    code='''import json
from pathlib import Path
root=Path(REMOTE_VALUE)/'early_validation';out={}
for name in ('latest.json.profile.json','latest.json.partial.json','latest.json'):
    path=root/name
    if path.exists():out[name]=json.loads(path.read_text())
print(json.dumps(out))
'''.replace('REMOTE_VALUE',repr(deploy.REMOTE))
    end=start['deadline']+60
    while time.time()<end:
        try:
            rows=json.loads(deploy.remote_python(code,timeout=20))
            for name,data in rows.items():
                tmp=out/(name+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(out/name)
            if 'latest.json' in rows:
                data=rows['latest.json']
                if data.get('diagnostic') is not True or data.get('affects_selection') is not False:
                    raise ValueError('Snapshot scope mismatch')
                (out/'mirror_result.json').write_text(json.dumps(dict(retrieved=True,
                    complete=data.get('complete'),checkpoint_step=data.get('checkpoint_step'),
                    observed=time.time()),indent=2)+'\n')
                return
        except Exception as exc:
            (out/'mirror_last_error.json').write_text(json.dumps(dict(error_type=type(exc).__name__,observed=time.time()))+'\n')
        time.sleep(15)
    (out/'mirror_result.json').write_text(json.dumps(dict(retrieved=False,deadline_elapsed=True,observed=time.time()))+'\n')


if __name__=='__main__':main()
