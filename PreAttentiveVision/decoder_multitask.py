"""Order-sensitive, spatial multiscale comparison shared by all five encoders."""
import torch
from torch import nn
from torch.nn import functional as F
from PreAttentiveVision.decoder import ConvNormAct


def local_correlation(first, second, radius=2):
    """Channel-normalized ordered match a(x,y) with b(x+dx,y+dy).

    Zero padding gives no cyclic image shortcut. Channel order is dy-major,
    then dx, each from -radius through radius. These are learned feature
    similarities, not privileged motion labels or optical-flow targets.
    """
    a=F.normalize(first,dim=1,eps=1e-6)
    b=F.pad(F.normalize(second,dim=1,eps=1e-6),(radius,)*4)
    h,w=first.shape[-2:]
    return torch.cat([(a*b[:,:,radius+dy:radius+dy+h,radius+dx:radius+dx+w]).sum(1,keepdim=True)
                      for dy in range(-radius,radius+1) for dx in range(-radius,radius+1)],1)


class OrderedTaskDecoder(nn.Module):
    def __init__(self, task_classes, channels=(24,48,96), dropout=.1):
        super().__init__()
        self.task_classes=dict(task_classes)
        self.projections=nn.ModuleList([ConvNormAct(c,32,1) for c in channels])
        self.local=nn.ModuleList([ConvNormAct(4*32+25,32) for _ in channels])
        self.fusion=ConvNormAct(32*len(channels),64)
        self.trunk=nn.Sequential(nn.Linear(128,128),nn.SiLU(),nn.Dropout(dropout))
        self.heads=nn.ModuleDict({name:nn.Linear(128,n) for name,n in task_classes.items()})

    def forward(self, first, second, task):
        scales=[]
        for a,b,project,local in zip(first,second,self.projections,self.local):
            pa,pb=project(torch.cat((a,b),0)).split(len(a),0)
            interactions=torch.cat((pb-pa,(pb-pa).abs(),.5*(pa+pb),pa*pb,
                                    local_correlation(pa,pb)),1)
            scales.append(F.adaptive_avg_pool2d(local(interactions),(13,13)))
        field=self.fusion(torch.cat(scales,1))
        pooled=torch.cat((field.mean((2,3)),field.amax((2,3))),1)
        return self.heads[task](self.trunk(pooled))


class MultitaskPairClassifier(nn.Module):
    """Task identity selects a supervised readout; it is not a feature input."""
    def __init__(self, encoder, task_classes):
        super().__init__()
        self.encoder=encoder
        self.decoder=OrderedTaskDecoder(task_classes,encoder.out_channels)

    def forward(self, images, task):
        if images.ndim!=5 or tuple(images.shape[1:])!=(2,3,100,100):
            raise ValueError('Expected ordered RGB pair [B,2,3,100,100]')
        batch=len(images)
        fields=self.encoder((torch.cat((images[:,0],images[:,1]),0)-.5)/.5)
        return self.decoder([f[:batch] for f in fields],[f[batch:] for f in fields],task)
