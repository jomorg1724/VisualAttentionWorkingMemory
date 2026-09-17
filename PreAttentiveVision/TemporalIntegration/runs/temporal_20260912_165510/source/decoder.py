"""One expressive, swap-symmetric spatial decoder shared by all PAV encoders."""
import torch
from torch import nn
from torch.nn import functional as F


class ConvNormAct(nn.Sequential):
    def __init__(self, cin, cout, kernel=3):
        super().__init__(nn.Conv2d(cin, cout, kernel, padding=kernel//2, bias=False),
                         nn.GroupNorm(8, cout), nn.SiLU())


class CommonPairDecoder(nn.Module):
    """Learn local comparisons before pooling; no image or metadata bypass.

    Each scale uses one shared projection P for both frames. Features
    [|P(a)-P(b)|, (P(a)+P(b))/2, P(a)*P(b)] are symmetric in the frame order.
    Local convolutions mix channels/interactions while retaining the spatial
    field. Aligned multiscale fusion precedes global mean/max and an MLP.
    """
    def __init__(self, channels=(24, 48, 96), dropout=.1):
        super().__init__()
        self.projections = nn.ModuleList([ConvNormAct(c, 32, 1) for c in channels])
        self.local = nn.ModuleList([ConvNormAct(96, 32) for _ in channels])
        self.fusion = ConvNormAct(32*len(channels), 64)
        self.classifier = nn.Sequential(nn.Linear(128, 128), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(128, 2))

    def forward(self, first, second):
        if len(first) != len(self.projections) or len(second) != len(first):
            raise ValueError('Encoder must supply the declared spatial scales')
        scales=[]
        for a,b,project,local in zip(first,second,self.projections,self.local):
            if a.shape != b.shape:
                raise ValueError('Paired encoder fields must have matching shapes')
            size=len(a)
            pa,pb=project(torch.cat((a,b),dim=0)).split(size,dim=0)
            interaction=torch.cat(((pa-pb).abs(),.5*(pa+pb),pa*pb),dim=1)
            scales.append(F.adaptive_avg_pool2d(local(interaction),(13,13)))
        field=self.fusion(torch.cat(scales,dim=1))
        pooled=torch.cat((field.mean((2,3)),field.amax((2,3))),dim=1)
        return self.classifier(pooled)


class PairClassifier(nn.Module):
    """Same preprocessing and pair decoder for every independent encoder."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder=encoder
        self.decoder=CommonPairDecoder(encoder.out_channels)

    def forward(self, images):
        if images.ndim!=5 or tuple(images.shape[1:])!=(2,3,100,100):
            raise ValueError('Expected RGB pair tensor[B,2,3,100,100]')
        batch=len(images)
        pair=torch.cat((images[:,0],images[:,1]),dim=0)
        # Stimulus generator delivers native RGB in[0,1], including sensory noise.
        fields=self.encoder((pair-.5)/.5)
        return self.decoder([f[:batch] for f in fields],[f[batch:] for f in fields])
