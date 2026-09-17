"""Contact sheet with the actual two inputs, generated from held-out seed."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .neuroscience_stimuli import TaskStream,TASK_CLASSES,DIRECTION_NAMES


def main():
    stream=TaskStream(2026091203,'val')
    canvas=Image.new('RGB',(670,len(TASK_CLASSES)*125),'white')
    draw=ImageDraw.Draw(canvas)
    records=[]
    for row,task in enumerate(TASK_CLASSES):
        frames,labels,meta=stream.batch(2,task)
        draw.text((5,row*125+10),task,fill='black')
        for example in range(2):
            label=int(labels[example])
            answer=DIRECTION_NAMES[label] if task=='motion_direction' else ('second CW' if label else 'second CCW') if task=='orientation' else f'target frame{label+1}'
            draw.text((220+example*225,row*125+3),answer,fill='black')
            for frame in range(2):
                pixels=(frames[example,frame].numpy().transpose(1,2,0)*255).astype(np.uint8)
                canvas.paste(Image.fromarray(pixels),(220+example*225+frame*105,row*125+20))
        records.extend(meta)
    root=Path(__file__).parent
    canvas.save(root/'sensory_battery_examples.png')
    (root/'sensory_battery_examples.json').write_text(json.dumps(records,indent=2),encoding='utf-8')


if __name__=='__main__': main()
