"""VAE9900 encoder → normalized spatial tokens ×10 → original RViT."""
from SecondPass.VAERViT.model import VAERViT as BaseVAERViT

class VAERViT(BaseVAERViT):
    input_scale = 10.0

    def encode_tokens(self, triplets):
        # Scale after token LayerNorm so normalization cannot cancel this change.
        # The RViT itself retains its original internal norms and residual route.
        return super().encode_tokens(triplets) * self.input_scale
