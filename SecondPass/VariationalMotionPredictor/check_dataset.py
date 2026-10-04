"""Focused CPU dataset checks and reproducible example panel."""
import json
import math
from pathlib import Path
import random
import numpy as np
import torch
from PIL import Image,ImageDraw
from . import dataset as d


def check():
    a,target,meta=d.generate(7,'train'); duplicate=d.generate(7,'train')
    assert tuple(a.shape)==(3,3,100,100) and tuple(target.shape)==(3,100,100) and a.dtype==torch.float32
    assert torch.equal(a,duplicate[0]) and torch.equal(target,duplicate[1]) and meta==duplicate[2]
    unwrapped=np.array(meta['positions_unwrapped_xy']); wrapped=np.array(meta['positions_wrapped_xy'])
    assert np.allclose(np.diff(unwrapped,axis=0),np.array(meta['velocity_xy'])[None,None,:],atol=1e-13)
    assert np.array_equal(wrapped,unwrapped%100)
    for frame_index,frame in enumerate(list(a)+[target]):
        assert np.array_equal(frame.numpy(),d.render_positions(wrapped[frame_index]))
    # Recreate exactly the unconditioned whole-field initial-position draw.
    rng=np.random.default_rng(np.random.SeedSequence([d.NAMESPACES['train'],7]))
    rng.uniform(0.,2*math.pi); count=int(rng.integers(16,49)); expected=rng.uniform(0.,100.,size=(count,2))
    assert np.array_equal(expected,np.array(meta['initial_positions_xy']))
    assert len({d.generate(7,split)[2]['trial_id'] for split in ('train','val','test')})==3
    stream=d.MotionStream(); stream.batch(3); state=stream.state_dict(); expected=stream.batch(2)
    restored=d.MotionStream(); restored.load_state_dict(state); actual=restored.batch(2)
    assert torch.equal(expected[0],actual[0]) and torch.equal(expected[1],actual[1]) and expected[2]==actual[2]
    assert np.array_equal(d.render_positions([[99.8,.2]]),d.render_positions([[-.2,100.2]]))
    class FakePool(d.MotionPool):
        def _new_pool(self):
            self.pool_index+=1; self.epoch=0; self.cursor=0; self.pool_start=self.stream.next_index
            self.stream.next_index+=d.POOL_SIZE; self.order=list(range(d.POOL_SIZE)); self.rng.shuffle(self.order)
    pool=FakePool(); seen=[set(),set()]; sizes=[]
    for update in range(64):
        ix=pool.next_indices(); assert pool.pool_index==0 and pool.epoch==update//32
        assert not seen[pool.epoch].intersection(ix); seen[pool.epoch].update(ix); sizes.append(len(ix))
    assert len(seen[0])==len(seen[1])==1000 and sum(sizes)==2000 and sizes[31]==sizes[63]==8
    pool.next_indices(); assert pool.pool_index==1 and pool.epoch==0 and pool.updates==65
    return dict(verified=True,checks=['constant unwrapped velocity/persistent identities','periodic7x7 rendering',
        'unconditioned iid whole-field starts','deterministic disjoint streams and exact counter resume',
        'past3 versus fourthtarget separation','1000 movies twice/64updates/2000presentations then refresh65'],
        gpu_used=False,stimulus='new synthetic benchmark, not Krauzlis replication')


def examples(path):
    canvas=Image.new('RGB',(400,360),(128,128,128)); drawing=ImageDraw.Draw(canvas)
    for row,index in enumerate((0,1,2)):
        past,target,meta=d.generate(index,'train')
        for column,frame in enumerate(list(past)+[target]):
            array=(frame.permute(1,2,0).numpy()*255).round().astype(np.uint8)
            canvas.paste(Image.fromarray(array),(column*100,row*120+20))
        drawing.text((3,row*120+3),f"speed={meta['speed_pixels_per_frame']} theta={meta['angle_radians']:.2f}; inputs0,1,2 | target3",fill=(0,0,0))
    canvas.save(path)

if __name__=='__main__':
    torch.set_num_threads(2); directory=Path(__file__).parent
    result=check(); examples(directory/'dataset_examples.png')
    (directory/'dataset_check.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))
