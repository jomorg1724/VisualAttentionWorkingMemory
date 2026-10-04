"""Independent read-only CPU checkpoint audit; generates a verification receipt."""
import argparse
import json
from pathlib import Path
import torch
from SecondPass.JointTraining.core import tree_equal,atomic_json
from SecondPass.JointTraining.worker import utc
from SecondPass.SpatialReadout.state import verify_parent,migrate,load_verified,verify_updated


def main():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1: torch.set_num_interop_threads(1)
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['profile','production']); p.add_argument('directory')
    args=p.parse_args(); directory=Path(args.directory).resolve()
    prep=json.loads((directory/'preparation.json').read_text())
    source,receipt=verify_parent(prep['parent_pointer']); expected=migrate(source,receipt)
    if args.mode=='profile':
        profile=json.loads((directory/'profile.json').read_text())
        assert profile['complete'] and len(profile['rows'])==35
        migration_receipt=profile['first_migration']; updated_receipt=profile['checkpoint']
        updated=load_verified(updated_receipt)
        names=updated['optimizer_names']
        adam={n:float(updated['optimizer']['state'][names.index(n)]['step']) for n in expected['migration']['fresh_names']}
        assert all(step==35 for step in adam.values())
        changed=[n for n in expected['migration']['fresh_names'] if not torch.equal(expected['model'][n],updated['model'][n])]
        assert len(changed)==8
        checks=dict(verified=True,disposable=True,production_updates=0,profile_updates=updated['state']['step'],
            fresh_adam_steps=adam,changed_fresh_tensors=changed,checkpoint=updated_receipt)
    else:
        integrity=json.loads((directory/'migration_integrity.json').read_text())
        migration_receipt=integrity['migration']
        updated_receipt=json.loads((directory/'latest_checkpoint.json').read_text())
        checks=verify_updated(migration_receipt,updated_receipt)
        progress=[json.loads(line) for line in (directory/'progress.jsonl').read_text().splitlines() if line.strip()]
        assert progress and progress[-1]['step']>=checks['branch_step']
        checks['latest_logged_step']=progress[-1]['step']
    actual=load_verified(migration_receipt)
    equal={key:tree_equal(expected[key],actual[key]) for key in ('model','optimizer','optimizer_names','scheduler','stream','rng','migration')}
    assert all(equal.values()),equal
    checks.update(migration_exact_checks=equal,parent=receipt,verified_utc=utc(),mode=args.mode)
    output=directory/('independent_'+args.mode+'_verification.json'); atomic_json(output,checks)
    print(json.dumps(checks,indent=2))


if __name__=='__main__': main()
