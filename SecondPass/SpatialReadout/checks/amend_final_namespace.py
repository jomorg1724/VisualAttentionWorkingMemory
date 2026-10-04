"""Preproduction-only seed hygiene amendment; preserve the original profile.

A tiny untrained CPU interface test consumed test namespace94592763 in two
cells. Reserve94692763 for production final tests and move that CPU test to
validation. This changes no profiled training/evaluation cost or checkpoint.
"""
import json
from pathlib import Path
import shutil
from SecondPass.JointTraining.core import atomic_json
from SecondPass.SpatialReadout.protocol import digest


def main():
    root=Path(__file__).resolve().parents[3]
    runtime=Path('/Users/jonathanmorgan/VAWMRuntime/final_convgru_01/repo')
    run=runtime.parent/'run'
    assert json.loads((run/'profile.json').read_text())['complete']
    assert not (run/'config.json').exists() and not (run/'migration.pt').exists()
    backup=run/'preparation.before_final_namespace.json'
    assert not backup.exists()
    previous=json.loads((run/'preparation.json').read_text())
    allowed=['SecondPass/SpatialReadout/protocol.py','SecondPass/SpatialReadout/test_launch.py']
    changes=[]
    for relative in allowed:
        old=runtime/relative; new=root/relative
        assert digest(old)==previous['source_hashes'][str(old)]
        if relative.endswith('protocol.py'):
            assert old.read_text().replace('FINAL_TEST_NAMESPACE=94592763','FINAL_TEST_NAMESPACE=94692763')==new.read_text()
        changes.append(dict(path=relative,old_sha256=digest(old),new_sha256=digest(new)))
    shutil.copy2(run/'preparation.json',backup)
    for relative in allowed:
        shutil.copy2(root/relative,runtime/relative)
        previous['source_hashes'][str(runtime/relative)]=digest(runtime/relative)
    amendment=dict(old_namespace=94592763,new_namespace=94692763,changes=changes,
        reason='Reserve untouched final draws after an untrained CPU interface smoke; future evaluator smoke uses validation',
        config_not_yet_pinned=True,production_updates=0,profile_costs_unchanged=True,
        cap_origin_and_deadline_unchanged=True,profile_artifacts_preserved=True)
    previous['final_namespace_amendment']=amendment
    atomic_json(run/'preparation.json',previous)
    atomic_json(run/'final_namespace_amendment.json',amendment)
    for path in previous['source_hashes']:
        target=run/'production_locked_source'/Path(path).relative_to(runtime)
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,target)
    print(json.dumps(amendment,indent=2))


if __name__=='__main__': main()
