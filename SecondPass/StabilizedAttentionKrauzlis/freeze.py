"""Preserve source and checkpoint provenance before the one authorized launch."""
import hashlib,json,shutil
from pathlib import Path
from .train import ROOT,source_hashes

def main():
    run=Path(__file__).resolve().parent/'LocalRuntime/run01'
    assert not (run/'budget.json').exists()
    run.mkdir(parents=True,exist_ok=True)
    hashes=source_hashes()
    for relative in hashes:
        target=run/'source'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,target)
    old=ROOT/'SecondPass/LocalComparisonKrauzlis/LocalRuntime/run01'
    preserved={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (old/'best.pt',old/'latest.pt')}
    (run/'prelaunch.json').write_text(json.dumps(dict(source_hashes=hashes,preserved_checkpoints=preserved,
        authorized='user requested suggested stabilization and clipping100; fresh local model',
        whole_model_fresh=True,checkpoint_inputs=[],wall_seconds=14400,requested_updates=2244,
        gradient_normalization_after_clipping=False,activation_normalization='forward gate/candidate LayerNorm and QK L2'),indent=2),encoding='utf-8')
    print(str(run))

if __name__=='__main__':main()
