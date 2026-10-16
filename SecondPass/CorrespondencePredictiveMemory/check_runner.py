"""Exercise real production save/select/evaluate lifecycle on a disposable CPU fixture."""
import argparse
import json
from pathlib import Path
import tempfile
import time
import torch
from . import train


def main():
    train.POOL_SIZE=8; train.POOL_EPOCHS=1; train.POOL_UPDATES=2; train.BATCH=4
    train.TRANSFER_COUNT=12
    with tempfile.TemporaryDirectory(prefix='correspondence_cpu_check_') as temp:
        run=Path(temp)
        args=argparse.Namespace(device='cpu',val_count=12,test_count=12)
        train.atomic_json(run/'profile.json',dict(started=time.time(),deadline=time.time()+600,
            target_updates=2,device='cpu',val_count=12,test_count=12,source_hashes=train.source_hashes()))
        train.production(args,run)
        saved=torch.load(run/'latest.pt',map_location='cpu')
        adam=saved['optimizer']['state']
        assert len(adam)==43 and all(float(s['step'])==2 for s in adam.values())
        final=json.loads((run/'final_test.json').read_text())
        transfer=json.loads((run/'transfer_test.json').read_text())
        assert final['n']==transfer['n']==12 and len(final['cells'])==len(transfer['cells'])==6
        assert transfer['domain']=='transfer' and not transfer['used_for_selection']
        result=dict(status='passed',cpu_only=True,fixture_not_science_result=True,updates=2,
            finite_aligned_adam_states=len(adam),standard_and_transfer_test_paths_verified=True,
            best_latest_selection_and_completion_verified=True)
    Path(__file__).with_name('runner_check_results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
