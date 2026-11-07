"""Trained deterministic VAE spatial means driving one fresh shared RViT block."""
from __future__ import annotations
import hashlib
from pathlib import Path
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from SecondPass.ThreeFrameConvVAE.model import TripletEncoder
from SecondPass.TwoFrameRViT.model import RecurrentVisualBlock


class VAERViT(nn.Module):
    def __init__(self, checkpoint_encoder: bool = True, checkpoint_recurrent: bool = True):
        super().__init__()
        self.checkpoint_encoder = checkpoint_encoder
        self.checkpoint_recurrent = checkpoint_recurrent
        self.encoder = TripletEncoder()
        self.mu_head = nn.Conv2d(256, 256, 1)
        self.row_embedding = nn.Parameter(torch.empty(13, 256))
        self.column_embedding = nn.Parameter(torch.empty(13, 256))
        nn.init.normal_(self.row_embedding, std=.02)
        nn.init.normal_(self.column_embedding, std=.02)
        self.token_norm = nn.LayerNorm(256)
        self.recurrent_block = RecurrentVisualBlock()
        self.readout_norm = nn.LayerNorm(256)
        self.token_readout = nn.Sequential(nn.Linear(256, 16), nn.GELU())
        self.classifier = nn.Sequential(nn.Linear(2704, 512), nn.GELU(),
                                        nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 2))

    @staticmethod
    def _validate(images: Tensor, movie: bool = True) -> None:
        if images.ndim != (5 if movie else 4) or tuple(images.shape[-3:]) != (3, 100, 100):
            raise ValueError('expected native [B,T,3,100,100] movies or [B,3,100,100] frames')
        if images.dtype != torch.float32 or torch.is_autocast_enabled(images.device.type):
            raise ValueError('VAERViT requires FP32 and disabled autocast')
        if movie and images.shape[1] < 1:
            raise ValueError('empty movie')

    def transfer_encoder(self, vae_state: dict) -> dict:
        """Copy only the VAE encoder and posterior-mean head, never other state."""
        state = vae_state.get('model', vae_state)
        copied = []
        for prefix, module in (('encoder.', self.encoder), ('mu_head.', self.mu_head)):
            weights = {name[len(prefix):]: value for name, value in state.items() if name.startswith(prefix)}
            if not weights or any(value.dtype != torch.float32 for value in weights.values()):
                raise ValueError('missing or non-FP32 VAE encoding weights: ' + prefix)
            module.load_state_dict(weights, strict=True)
            copied.extend(prefix + name for name in weights)
        assert all(parameter.requires_grad for parameter in self.parameters())
        return dict(copied_tensors=len(copied), copied_names=copied,
                    copied_parameters=sum(p.numel() for module in (self.encoder, self.mu_head) for p in module.parameters()),
                    excluded_modules=['logvar_head', 'decoder'], all_learned_parameters_trainable=True)

    def load_pretrained_encoder(self, path: str | Path, expected_sha256: str | None = None) -> dict:
        path = Path(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError('VAE source checkpoint digest mismatch')
        payload = torch.load(path, map_location='cpu', weights_only=False)
        receipt = self.transfer_encoder(payload)
        receipt.update(path=str(path.resolve()), sha256=digest, source_step=payload.get('state', {}).get('step'),
                       inherited_optimizer=False, inherited_rng=False, inherited_stream=False)
        return receipt

    def encoder_mu(self, triplets: Tensor) -> Tensor:
        self._validate(triplets)
        if triplets.shape[1] != 3:
            raise ValueError('VAE encoding requires exactly three ordered frames')
        ordered = triplets.reshape(triplets.shape[0], 9, 100, 100)
        if self.checkpoint_encoder and self.training and torch.is_grad_enabled():
            field = checkpoint(self.encoder, ordered, use_reentrant=False)
        else:
            field = self.encoder(ordered)
        return self.mu_head(field)

    def encode(self, triplets: Tensor) -> Tensor:
        return self.encoder_mu(triplets)

    def encode_tokens(self, triplets: Tensor) -> Tensor:
        # Row-major spatial addresses become169tokens, each with256features.
        tokens = self.encoder_mu(triplets).flatten(2).transpose(1, 2)
        positions = (self.row_embedding[:, None] + self.column_embedding[None, :]).reshape(169, 256)
        return self.token_norm(tokens + positions)

    def update_memory(self, current: Tensor, memory: Tensor | None = None) -> Tensor:
        if memory is None:
            memory = current.new_zeros(current.shape)
        if memory.shape != current.shape or memory.device != current.device or memory.dtype != current.dtype:
            raise ValueError('memory must match current [B,169,256] tokens')
        if self.checkpoint_recurrent and self.training and torch.is_grad_enabled():
            return checkpoint(self.recurrent_block, current, memory, use_reentrant=False)
        return self.recurrent_block(current, memory)

    def decode(self, memory: Tensor) -> Tensor:
        return self.classifier(self.token_readout(self.readout_norm(memory)).flatten(1))

    def forward(self, images: Tensor, task: str = 'krauzlis_cued_motion') -> Tensor:
        if task not in ('krauzlis_cued_motion', 'krauzlis'):
            raise ValueError('VAERViT is the single-stimulus Krauzlis candidate')
        self._validate(images)
        memory = None
        for step in range(images.shape[1]):
            triplets = torch.stack([images[:, max(step-lag, 0)] for lag in (2, 1, 0)], dim=1)
            memory = self.update_memory(self.encode_tokens(triplets), memory)
        return self.decode(memory)

    def stream_step(self, current: Tensor, previous_frames: Tensor | None = None,
                    memory: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """Stateless causal update: return logits, next two raw frames, next memory."""
        self._validate(current, movie=False)
        if previous_frames is None:
            previous_frames = current.unsqueeze(1).expand(-1, 2, -1, -1, -1)
        self._validate(previous_frames)
        if previous_frames.shape[:2] != (current.shape[0], 2) or previous_frames.device != current.device:
            raise ValueError('history must contain two prior raw frames [B,2,3,100,100]')
        triplets = torch.cat((previous_frames, current.unsqueeze(1)), dim=1)
        updated = self.update_memory(self.encode_tokens(triplets), memory)
        return self.decode(updated), triplets[:, 1:], updated
