"""Fresh weighted RGB -> one shared four-head Transformer -> change decision.

Each clip is summarized as (Xold+2Xmiddle+3Xnew)/6. Ten-by-ten patches are
linearly embedded; learned CLS and position embeddings feed exactly one
pre-normalized Transformer layer. The two clips never share attention tokens.
"""
import torch
from torch import nn


def weighted_sum(clips, normalized=True):
    if clips.ndim != 5 or tuple(clips.shape[1:]) != (3, 3, 100, 100):
        raise ValueError('Expected batch x3frames x3channels x100 x100')
    weights = clips.new_tensor([1., 2., 3.]).reshape(1, 3, 1, 1, 1)
    summed = (clips * weights).sum(1)
    return summed / 6. if normalized else summed


class WeightedSumTransformer(nn.Module):
    def __init__(self, embedding_dim=128, patch_size=10):
        super().__init__()
        if embedding_dim != 128 or patch_size != 10:
            raise ValueError('Pinned architecture requires patch10 and embedding128')
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.patch_embedding = nn.Conv2d(3, embedding_dim, patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, 101, embedding_dim))
        nn.init.normal_(self.cls_token, std=.02)
        nn.init.normal_(self.position_embedding, std=.02)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=4, dim_feedforward=512,
            dropout=0., activation='gelu', batch_first=True, norm_first=True)
        self.cls_norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(2 * embedding_dim, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1))

    def encode_images(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 100, 100):
            raise ValueError('Expected batch x3RGB x100 x100 normalized weighted sums')
        tokens = self.patch_embedding(images - .5).flatten(2).transpose(1, 2)
        tokens = torch.cat([self.cls_token.expand(len(images), -1, -1), tokens], dim=1)
        encoded = self.transformer(tokens + self.position_embedding)
        return self.cls_norm(encoded[:, 0])

    def encode_clips(self, clips):
        return self.encode_images(weighted_sum(clips))

    def forward(self, before_clips, after_clips):
        if len(before_clips) != len(after_clips):
            raise ValueError('Before and after clips must have matching batch sizes')
        n = len(before_clips)
        # Batch concatenation shares parameters, not attention along token axis.
        embeddings = self.encode_clips(torch.cat([before_clips, after_clips], dim=0))
        return self.classifier(torch.cat([embeddings[:n], embeddings[n:]], dim=1)).squeeze(-1)
