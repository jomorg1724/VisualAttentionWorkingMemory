"""Focused CPU tests for the declared stimulus signals, not model performance."""
import json
from pathlib import Path
import numpy as np
import torch
from .neuroscience_stimuli import TaskStream


def check():
    stream = TaskStream(7142801,'val')
    result = {}
    for task in ('orientation','contrast','spatial_frequency','chromatic_increment','contour'):
        state = stream.state_dict()
        x,y,meta = stream.batch(24,task)
        assert x.shape == (24,2,3,100,100)
        assert torch.isfinite(x).all() and x.min() >= 0 and x.max() <= 1
        assert torch.bincount(y).tolist() == [12,12]
        stream.load_state_dict(state)
        xx,yy,mm = stream.batch(24,task)
        assert torch.equal(x,xx) and torch.equal(y,yy) and meta == mm
        evidence = {'finite_in_gamut':True,'balanced_labels':True,'exact_replay':True}
        if task == 'contrast':
            predicted = np.argmax(np.mean((x[:,:,0].numpy()-.5)**2,axis=(-1,-2)),axis=1)
            assert np.all(predicted == y.numpy())
            evidence['pixel_energy_label_accuracy'] = 1.
        elif task == 'spatial_frequency':
            image = x[:,:,0].numpy()-.5
            energy = np.mean(image**2,axis=(-1,-2))
            dx,dy = np.gradient(image,axis=(-2,-1))
            frequency_score = np.mean(dx**2+dy**2,axis=(-1,-2))/energy
            predicted = np.argmax(frequency_score,axis=1)
            assert np.mean(predicted == y.numpy()) >= .95
            evidence['gradient_energy_label_accuracy'] = float(np.mean(predicted==y.numpy()))
        elif task == 'chromatic_increment':
            axis=np.array(meta[0]['axis_linear_rgb'])
            means=x.numpy().mean((-1,-2))
            predicted=np.argmax(means@axis,axis=1)
            assert np.all(predicted==y.numpy())
            assert max(abs(m['frame_luminances'][0]-m['frame_luminances'][1]) for m in meta) < 1e-6
            evidence['chromatic_axis_label_accuracy'] = 1.
            evidence['numerical_luminance_matched'] = True
        elif task == 'orientation':
            angles=[]
            for pair in x[:,:,0].numpy():
                aa=[]
                for image in pair:
                    dy,dx=np.gradient(image)
                    aa.append(.5*np.arctan2(2*np.sum(dx*dy),np.sum(dx*dx-dy*dy)))
                delta=.5*np.arctan2(np.sin(2*(aa[1]-aa[0])),np.cos(2*(aa[1]-aa[0])))
                angles.append(int(delta>0))
            assert np.mean(np.array(angles)==y.numpy()) >= .95
            evidence['gradient_orientation_label_accuracy'] = float(np.mean(np.array(angles)==y.numpy()))
        else:
            state=stream.state_dict()
            first,_=stream._contour(0,1)
            stream.load_state_dict(state)
            second,_=stream._contour(1,1)
            assert np.array_equal(first,second[::-1])
            evidence['structured_interval_swap_exact'] = True
        result[task]=evidence
    return result


if __name__ == '__main__':
    result=check()
    Path(__file__).with_name('sensory_battery_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))
