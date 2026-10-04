"""Real CPU batch32/micro4 native renderer -> production launcher handshake."""
import json
import time
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
import sys
sys.path.insert(0,str(Path(__file__).parent))
import deploy
from SecondPass.TwoFrameRViT import worker as w


def test_native_session_checkpoint3_readiness(tmp_path):
    torch.set_num_threads(2)
    cfg=dict(device='cpu',cap_started=time.time(),deadline=time.time()+1800,
        effective_batch=32,microbatch=4,cpu_threads=2,source_hashes={})
    profile=tmp_path/'profile';profile.mkdir()
    w.atomic_json(profile/'profile_config.json',dict(cfg,disposable_profile=True))
    w.profile(profile)
    measured=json.loads((profile/'profile.json').read_text())
    assert len(measured['rows'])==6 and len(measured['evaluation_cells'])==3
    assert {r['cell'] for r in measured['rows']}==set(w.CELLS)
    with patch('torch.load',side_effect=AssertionError('checkpoint initialization forbidden')):
        s=w.Session(tmp_path,cfg)
    s.checkpoint('initial.pt')
    assert json.loads((tmp_path/'constructor_equality.json').read_text())['verified']
    for _ in range(3):
        s.train_update(*s.scheduler.next())
        s.checkpoint(f"checkpoint_{s.state['step']:06d}.pt")
    data={n:json.loads((tmp_path/n).read_text()) for n in
          ('latest_checkpoint.json','persisted_progress_verification.json','first_cycle_timing.json')}
    data['progress']=json.loads((tmp_path/'progress.jsonl').read_text().splitlines()[-1])
    # This is the historical failure: the actual inherited save verified 1,13, not3.
    assert data['persisted_progress_verification.json']['step']==3
    assert deploy.production_ready(data)
    cp=data['latest_checkpoint.json'];saved=w.load_verified(cp)
    assert saved['state']['step']==3 and saved['state']['episodes']==96
    assert saved['scheduler']['updates']==3 and saved['stream']['streams']
    assert set(saved['rng'])=={'cpu','numpy','python'}
    verification=data['persisted_progress_verification.json']
    assert all(verification['changed_by_module'].values())
    assert verification['all_active_parameters_changed']
    assert w.tree_equal(saved['model'],w.cpu_tree(s.model.state_dict()))
    assert w.tree_equal(saved['optimizer'],w.cpu_tree(s.optimizer.state_dict()))
    assert w.tree_equal(saved['stream'],s.stream.state_dict())
    assert torch.equal(saved['rng']['cpu'],torch.get_rng_state())
    bad=dict(cp,sha256='0'*64)
    with pytest.raises((AssertionError,ValueError,RuntimeError)): w.load_verified(bad)
    with pytest.raises(FileExistsError): s.checkpoint('checkpoint_000003.pt')
    # Exercise the launcher's native-repair path too. Only SSH transport is
    # replaced; the actual verify CLI reads these real checkpoint bytes.
    import subprocess,sys,shlex,shutil
    def local_remote(command,timeout):
        args=shlex.split(command)
        result=subprocess.run([sys.executable,'-m',w.MODULE,'verify',str(tmp_path)],
            capture_output=True,text=True,timeout=timeout,check=True)
        assert args[-2]=='verify'
        return result.stdout
    data['persisted_progress_verification.json']=dict(verified=True,step=1)
    with patch.object(deploy,'remote',side_effect=local_remote):
        assert deploy.reconcile_verification(data)
    assert deploy.production_ready(data)
    parent=tmp_path/'parent';parent.mkdir()
    downloaded_cp=dict(cp,path='/workspace/test/run/'+Path(cp['path']).name)
    def transfer(args,**kwargs):
        shutil.copy2(cp['path'],args[-1])
    with patch.object(deploy,'ROOT',parent),patch.object(deploy,'REMOTE','/workspace/test'),patch.object(deploy,'ssh_command',return_value=['ssh','host']),patch.object(deploy.subprocess,'run',side_effect=transfer):
        deploy.download_checkpoint({'latest_checkpoint.json':downloaded_cp})
    assert json.loads((parent/'downloaded_checkpoint_verified.json').read_text())['verified']
    out=dict(production_ready=True,checkpoint=cp,verification=verification,
        timing=data['first_cycle_timing.json'],profile_rows=measured['rows'],
        actual_device='cpu',cpu_threads=torch.get_num_threads(),fresh_constructor=True,
        checkpoint_integrity=True,corruption_rejected=True)
    Path(__file__).with_name('native_cpu_evidence.json').write_text(json.dumps(out,indent=2))
    print(json.dumps({k:out[k] for k in ('production_ready','actual_device','cpu_threads','checkpoint_integrity')}))
