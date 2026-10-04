"""Focused native timing/pixel/label check; one whole-movie CPU backward."""
from collections import Counter
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from SecondPass.SingleStimulusRViT.model import SingleStimulusRViT
from SecondPass.SingleStimulusRViT.stimuli import single_stimulus_movies, SingleStimulusStream
from SecondPass.TwoFrameRViT.worker import FreshStream, TASK


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    started = time.monotonic()
    cells = []
    for cell, length in (('B12',29),('B20',37),('B28',45)):
        native, labels, metadata = FreshStream('val').batch(100, TASK, cell)
        images, labels, records = single_stimulus_movies(native, labels, metadata)
        assert images.shape == native.shape == (100,length,3,100,100)
        assert int(labels.sum()) == 57
        assert Counter(r['event_type'] for r in records) == {'target':57,'catch':43}
        assert Counter(r['source_event_type'] for r in records) == {'target':57,'foil':29,'catch':14}
        for index, record in enumerate(records):
            assert not record['cue_frames'] and record['stimulus_count'] == 1
            assert record['stimulus_location_xy'] in ([20.,50.],[80.,50.])
            assert torch.equal(images[index,:7],native[index,2].expand(7,-1,-1,-1))
            side=int(record['target_location']); cx=int(record['stimulus_location_xy'][0])
            assert torch.equal(images[index,7:,:,38:63,cx-12:cx+13],native[index,7:,:,38:63,cx-12:cx+13])
            other=int(metadata[index]['positions_xy'][1-side][0])
            assert bool((images[index,:,:,38:63,other-12:other+13] == .5).all())
            baseline=record['baseline_means_degrees'][0];post=record['postevent_means_degrees'][0]
            assert bool(abs(post-baseline)>1e-6) == bool(labels[index])
        cells.append(dict(cell=cell,frames=length,change_count=57,no_change_count=43,
                          original_target_pixels_identical=True,no_cues=True,other_patch_absent=True))
        del native,images
    stream=SingleStimulusStream('train');state=stream.state_dict()
    x,y,meta=stream.batch(2,TASK,'B12')
    restored=SingleStimulusStream('train');restored.load_state_dict(state)
    xx,yy,mm=restored.batch(2,TASK,'B12')
    assert torch.equal(x,xx) and torch.equal(y,yy) and meta==mm
    model=SingleStimulusRViT(checkpoint_encoder=False)
    logits=model(x[:1]);F.cross_entropy(logits,y[:1]).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    result=dict(complete=True,cells=cells,stream_restore_exact=True,
                full_native_cpu_backward=True,parameter_tensors=92,
                learned_parameters=sum(p.numel() for p in model.parameters()),
                no_optimizer_steps=True,seconds=time.monotonic()-started)
    Path(__file__).with_name('check_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
