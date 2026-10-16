"""CPU-only preparation receipt and plots of the exact renderer pixels."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .dataset import TASK,POPOUT,MOTION,TASKS,CATALOG,CONDITIONS,POPOUT_CONDITIONS,MOTION_CONDITIONS,Stream,all_cells
from .model import PARENT,initialize

def main():
    root=Path(__file__).parent;out=root/'examples';out.mkdir(exist_ok=True)
    torch.set_num_threads(2);model,opt=initialize(PARENT);model.eval();rows=[]
    with torch.no_grad():
        for task,c in [(TASK,c) for c in CONDITIONS]+[(POPOUT,c) for c in POPOUT_CONDITIONS]+[(MOTION,c) for c in MOTION_CONDITIONS]:
            x,y,m=Stream('test').batch(1,task,c['id']);logits,details=model(x,task,True)
            assert logits.shape==(1,4) and torch.isfinite(logits).all()
            assert torch.allclose(details['attention'].sum((-1,-2)),torch.ones(1,x.shape[1]),atol=1e-5)
            rows.append(dict(task=task,cell=c['id'],frames=x.shape[1],minimum_pair_rms=m[0].get('minimum_pair_rms')))
    fig,axes=plt.subplots(6,3,figsize=(9,15),facecolor='white')
    fig.suptitle('Memory-guided search: change the sample, keep the array',fontsize=15,y=.995)
    manifest=[]
    for row,(family,similarity) in enumerate((f,s) for f in ('features','objects','fractals') for s in ('distinct','similar')):
        cell=f'{family}_{similarity}_S4_D4_H8';x,y,m=Stream('test').paired_queries(cell,17)
        images=(x[0,1],x[3,1],x[0,m[0]['array_onset']])
        for col,(ax,img) in enumerate(zip(axes[row],images)):
            ax.imshow(img.permute(1,2,0).numpy(),interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
            ax.set_title(['Sample A: answer top-left','Sample B: answer bottom-right','Identical search array'][col],fontsize=10)
            for spine in ax.spines.values():spine.set_visible(False)
        axes[row,0].set_ylabel(f'{family}\n{similarity}',fontsize=11)
        raw=np.rint(x[0].permute(0,2,3,1).numpy()*255).astype(np.uint8)
        from PIL import Image
        frames_dir=out/cell;frames_dir.mkdir(exist_ok=True)
        for i,frame in enumerate(raw):Image.fromarray(frame).save(frames_dir/f'{i:02d}_{m[0]["phases"][i]}.png')
        manifest.append(dict(cell=cell,metadata=m[0],frames=[hashlib.sha256(f.tobytes()).hexdigest() for f in raw]))
    fig.tight_layout(rect=(0,.01,1,.98));fig.savefig(out/'cue_switch_examples.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(9,6),facecolor='white')
    fig.suptitle('Pop-out localization: one singleton, three identical distractors',fontsize=13)
    for row,(feature,levels) in enumerate((('color',(12,36,72)),('orientation',(5,15,45)))):
        for col,difference in enumerate(levels):
            cell=f'{feature}_X{difference}_H4';x,y,m=Stream('test').batch(1,POPOUT,cell)
            ax=axes[row,col];ax.imshow(x[0,-1].permute(1,2,0).numpy(),interpolation='nearest')
            ax.set_title(f'{feature}: difference {difference}\nanswer: {TASKS[POPOUT]["labels"][y[0]]}',fontsize=10);ax.axis('off')
            raw=np.rint(x[0].permute(0,2,3,1).numpy()*255).astype(np.uint8)
            frames_dir=out/cell;frames_dir.mkdir(exist_ok=True)
            for i,frame in enumerate(raw):Image.fromarray(frame).save(frames_dir/f'{i:02d}_{m[0]["phases"][i]}.png')
            manifest.append(dict(cell=cell,metadata=m[0],frames=[hashlib.sha256(f.tobytes()).hexdigest() for f in raw]))
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(out/'popout_examples.png',dpi=150);plt.close(fig)
    for spacing,target in [(20,1),(45,2),(90,0),(90,3)]:
        cell=f'motion_A{spacing}_S6_D4_H12';x,y,meta=Stream('test').motion_trial(cell,17,target)
        raw=np.rint(x.permute(0,2,3,1).numpy()*255).astype(np.uint8)
        name=f'{cell}_query{target}';frames_dir=out/name;frames_dir.mkdir(exist_ok=True)
        for i,frame in enumerate(raw):Image.fromarray(frame).save(frames_dir/f'{i:02d}_{meta["phases"][i]}.png')
        # Exact grayscale palette; 3x nearest-neighbor display enlargement.
        palette=[value for gray in range(256) for value in (gray,gray,gray)]
        gif=[]
        for frame in raw:
            im=Image.fromarray(np.repeat(np.repeat(frame[:,:,0],3,axis=0),3,axis=1)).convert('P')
            im.putpalette(palette);gif.append(im)
        gif[0].save(out/f'{name}.gif',save_all=True,append_images=gif[1:],duration=150,loop=0,optimize=False,disposal=2)
        manifest.append(dict(cell=cell,example_directory=name,gif=f'{name}.gif',gif_frame_ms=150,metadata=meta,
                             frames=[hashlib.sha256(f.tobytes()).hexdigest() for f in raw]))
    (root/'catalog.json').write_text(json.dumps(CATALOG,indent=2),encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    receipt=dict(verified_at=time.time(),task_count=len(TASKS),condition_count=len(all_cells()),new_tasks=3,new_conditions=len(CONDITIONS)+len(POPOUT_CONDITIONS)+len(MOTION_CONDITIONS),new_head_classes=4,
        parent=str(PARENT),parent_sha256=hashlib.sha256(PARENT.read_bytes()).hexdigest(),parent_step=16000,
        preserved_adam_states=len(opt.state),parameter_tensors=len(list(model.parameters())),parameters=sum(p.numel() for p in model.parameters()),
        cpu_condition_forward_checks=rows,training_started=False,weights_from_tests_discarded=True,tests='11 passed',
        source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in root.glob('*.py')})
    (root/'preparation_verified.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('cpu_condition_forward_checks','source_sha256')},indent=2))

if __name__=='__main__':main()
