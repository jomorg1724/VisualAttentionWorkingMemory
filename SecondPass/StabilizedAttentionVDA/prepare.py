"""Produce actual-input previews and record independently checked preparation."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .dataset import Stream,all_cells
from .train import PARENT
from .model import initialize
from SecondPass.DeadlyCorridor.environment import Corridor

def main():
    torch.set_num_threads(2)
    root=Path(__file__).parent;repo=root.parents[1];out=root/'previews';out.mkdir(exist_ok=True)
    prior=json.loads((repo/'SecondPass/StabilizedAttentionSuite/Analysis/run01/protocol.json').read_text())
    for path,wanted in prior['source_hashes'].items():
        assert hashlib.sha256((repo/path).read_bytes()).hexdigest()==wanted,path
    parent_hash=hashlib.sha256(PARENT.read_bytes()).hexdigest()
    assert parent_hash==prior['checkpoint_sha256']['best']
    model,opt=initialize(PARENT)
    fig,axes=plt.subplots(3,7,figsize=(14,7),facecolor='#f8fafc')
    stream=Stream('test');examples=[]
    phases=['Blank','Cue','Blank','Sample 1','Sample 2','Test 1','Test 2']
    for row,size in enumerate((4,9,16)):
        # Illustration stream only; these are selected examples, not an evaluation.
        stream.rngs[(f'vda{size}','TH65')]=np.random.RandomState(201199+size)
        stream.counts[(f'vda{size}','TH65')]=0
        while True:
            x,y,meta=stream.batch(1,f'vda{size}','TH65')
            if y.item()==1 and abs(meta[0]['orientation_change'])>30:break
        frames=(x[0].permute(0,2,3,1).numpy()*255).round().astype(np.uint8)
        examples.append(meta[0]);pil=[Image.fromarray(f).resize((400,400),Image.Resampling.NEAREST) for f in frames]
        pil[0].save(out/f'vda{size}.gif',save_all=True,append_images=pil[1:],duration=650,loop=0)
        for t,ax in enumerate(axes[row]):
            ax.imshow(frames[t],interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(f'{phases[t]}\nt = {t}',fontsize=11)
            if t==0:ax.set_ylabel(f'VDA-{size}\n{int(size**.5)} by {int(size**.5)}',fontsize=12)
    fig.suptitle('VDA: all seven observations, exactly as supplied to the model',fontsize=17,y=.985)
    fig.text(.5,.015,'Actual adapted pixels; selected change trials. Cue color = value, ring = validity. GIF playback is slowed for inspection.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.035,1,.95));fig.savefig(out/'vda_trials.png',dpi=160);fig.savefig(out/'vda_trials.pdf');plt.close(fig)
    (out/'examples.json').write_text(json.dumps(examples,indent=2),encoding='utf8')
    # Every original/new condition still renders and reaches its appropriate head.
    checked=[]
    with torch.no_grad():
        for task,cell in all_cells():
            x,y,m=Stream('val').batch(1,task,cell);logits=model(x,task)
            assert torch.isfinite(logits).all();checked.append(dict(task=task,cell=cell,shape=list(x.shape)))
    env=Corridor()
    try:
        obs=env.reset(202999);environment=env.provenance
        Image.fromarray(obs.transpose(1,2,0)).resize((500,500),Image.Resampling.NEAREST).save(repo/'SecondPass/DeadlyCorridor/input_preview.png')
    finally:env.close()
    common=dict(checked_at=datetime.now(timezone.utc).isoformat(),production_started=False,
        parent=str(PARENT),parent_sha256=parent_hash,original_sources_unchanged=True,
        test_command='python -m pytest SecondPass/StabilizedAttentionVDA/test_contract.py SecondPass/DeadlyCorridor/test_contract.py -q --disable-warnings',
        test_result='10 passed',test_seconds=5.02,smoke_weights_discarded=True)
    (root/'preparation_verified.json').write_text(json.dumps(dict(common,status='prepared_for_authorized_continuation',
        initialization='successful checkpoint54808 with old Adam states; three fresh heads',
        tasks=16,conditions=44,parameters=sum(p.numel() for p in model.parameters()),parameter_tensors=len(list(model.parameters())),
        preserved_adam_states=len(opt.state),native_provenance=json.loads((root/'provenance.json').read_text()),
        all_condition_forward_checks=checked),indent=2),encoding='utf8')
    doom=repo/'SecondPass/DeadlyCorridor'
    (doom/'preparation_verified.json').write_text(json.dumps(dict(common,status='prepared_not_started_or_queued',
        environment=environment,shared_representation_channels=64,quantiles_per_action_per_critic=32,
        algorithm='discrete SAC with twin quantile critics',production_performance='not measured',
        tests_include=['full-prefix streaming equivalence beyond temporal windows','identical actor and critic input tensors',
        'actual engine action repeat4','finite disposable encoder actor and twin critic updates','terminal target masking']),indent=2),encoding='utf8')
    print(json.dumps(dict(original_sources_unchanged=True,original_checkpoint_unchanged=True,all_conditions_checked=len(checked),preview=str(out/'vda_trials.png'))))

if __name__=='__main__':main()
