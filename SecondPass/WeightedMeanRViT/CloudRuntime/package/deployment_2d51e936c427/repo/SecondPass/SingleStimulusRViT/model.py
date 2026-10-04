"""Architecture alias: identical fresh original conv-encoder RViT, full BPTT."""
from SecondPass.TwoFrameRViT.model import TwoFrameRViT


class SingleStimulusRViT(TwoFrameRViT):
    """No architecture or decoding changes relative to the older cloud RViT."""

