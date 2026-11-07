"""Relevant author classes extracted verbatim from the NeurIPS 2024 implementation.

Source: mtangemann/motion_energy_segmentation commit
997ec55adf6062d8f92d4adcd555fdaccc5a1200, motion_energy_segmentation/model.py.
The unchanged source is preserved under upstream/. Our wrapper corrects the
H,W,T -> T,H,W filter storage layout and converts analytic constants to buffers.
"""
import logging
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Literal, Sequence
import torch
import torch.nn.functional as F
from torch import nn
from .upstream.simoncelli_heeger import SimoncelliHeegerModel
LOGGER = logging.getLogger(__name__)

class ImagePyramid(nn.Module):
    """Image pyramid based on downsampling by a factor of 2."""

    def __init__(self, levels: Sequence[int]) -> None:
        """Return a list of images downsampled to the specified levels."""
        super().__init__()

        levels = sorted(set(levels))
        if levels[0] < 0:
            raise ValueError(
                f"Expected all levels to be non-negative, but got {levels=}"
            )
        self.levels = levels

        self.blur = nn.Sequential(
            nn.Conv3d(
                1, 1, (1, 1, 5), bias=False, padding="same", padding_mode="replicate"),
            nn.Conv3d(
                1, 1, (1, 5, 1), bias=False, padding="same", padding_mode="replicate"),
        )
        weight = torch.Tensor([0.0884, 0.3536, 0.5303, 0.3536, 0.0884])
        weight_x = weight.view(1, 1, 1, 1, -1).expand_as(self.blur[0].weight)
        self.blur[0].weight.data = weight_x
        weight_y = weight.view(1, 1, 1, -1, 1).expand_as(self.blur[1].weight)
        self.blur[1].weight.data = weight_y
        for parameter in self.blur.parameters():
            parameter.requires_grad = False

        self.downsample = nn.AvgPool3d((1, 2, 2))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Return a list of images downsampled to the specified levels."""
        pyramid = []

        for level in range(self.levels[-1] + 1):
            if level in self.levels:
                pyramid.append(input)

            input = self.downsample(self.blur(input))

        return pyramid


class SimoncelliHeegerCNN(nn.Sequential):
    """CNN based on the Simoncelli & Heeger (1998) model.
    
    This keeps the linear filters, nonlinearities, and normalization from the original
    model. Scale factors are dropped everywhere.
    """

    def __init__(
        self,
        padding: Literal["same", "valid"] = "same",
        padding_time: Literal["same", "valid"] | None = None,
    ) -> None:
        if padding_time is None:
            padding_time = padding

        layers = OrderedDict()
        layers["v1_linear"] = nn.Sequential(OrderedDict([
            ("conv_t", nn.Conv3d(
                1, 10, (9, 1, 1),
                bias=False,
                padding=padding_time,
                padding_mode="replicate",
            )),
            ("conv_y", nn.Conv3d(
                10, 10, (1, 9, 1),
                groups=10,
                bias=False,
                padding=padding,
                padding_mode="replicate",
            )),
            ("conv_x", nn.Conv3d(
                10, 10, (1, 1, 9),
                groups=10,
                bias=False,
                padding=padding,
                padding_mode="replicate",
            )),
            ("readout", nn.Conv3d(
                10, 28, (1, 1, 1),
                bias=False,
                padding=padding,
                padding_mode="replicate",
            )),
        ]))
        layers["v1_nonlinear"] = Square()
        layers["v1_blur"] = nn.Sequential(
            nn.Conv3d(
                28, 28, (1, 11, 1),
                groups=28,
                bias=False,
                padding=padding,
                padding_mode="replicate",
            ),
            nn.Conv3d(
                28, 28, (1, 1, 11),
                groups=28,
                bias=False,
                padding=padding,
                padding_mode="replicate",
            ),
        )
        layers["v1_normalize"] = ChannelNorm()
        layers["mt_linear"] = nn.Conv3d(28, 19, 1, bias=False)
        layers["mt_blur"] = nn.Sequential(
            nn.Conv3d(
                19, 19, (1, 19, 1),
                groups=19,
                bias=False,
                padding=padding,
                padding_mode="replicate",
            ),
            nn.Conv3d(
                19, 19, (1, 1, 19),
                groups=19,
                bias=False,
                padding=padding,
                padding_mode="replicate",
            ),
        )
        layers["mt_nonlinear"] = RectifiedSquare()
        layers["mt_normalize"] = ChannelNorm()

        super().__init__(layers)
    
    @staticmethod
    def from_original_model(
        padding: Literal["same", "valid"] = "same",
        padding_time: Literal["same", "valid"] | None = None,
        layers: Sequence[str] | None = None,
    ) -> "SimoncelliHeegerCNN":
        LOGGER.info("Initializing SimoncelliHeegerCNN from original model")

        if layers is None:
            layers = ["v1_linear", "v1_blur", "mt_linear", "mt_blur"]

        model = SimoncelliHeegerCNN(padding, padding_time)
        original_model = SimoncelliHeegerModel()

        if "v1_linear" in layers:
            model.v1_linear.conv_t.weight.data = \
                original_model.v1_linear.conv_t.weight.data
            model.v1_linear.conv_y.weight.data = \
                original_model.v1_linear.conv_y.weight.data
            model.v1_linear.conv_x.weight.data = \
                original_model.v1_linear.conv_x.weight.data
            model.v1_linear.readout.weight.data = \
                original_model.v1_linear.readout.weight.data

        if "v1_blur" in layers:
            blur_weight = original_model.v1_blur.complex_filters
            model.v1_blur[0].weight.data = \
                blur_weight.view(1, 1, 1, -1, 1).expand_as(model.v1_blur[0].weight)
            model.v1_blur[1].weight.data = \
                blur_weight.view(1, 1, 1, 1, -1).expand_as(model.v1_blur[1].weight)

        if "mt_linear" in layers:
            model.mt_linear.weight.data = original_model.mt_linear.conv.weight.data

        if "mt_blur" in layers:
            blur_weight = original_model.mt_blur.spatial_pooling_filter
            model.mt_blur[0].weight.data = \
                blur_weight.view(1, 1, 1, -1, 1).expand_as(model.mt_blur[0].weight)
            model.mt_blur[1].weight.data = \
                blur_weight.view(1, 1, 1, 1, -1).expand_as(model.mt_blur[1].weight)

        return model

    @staticmethod
    def from_config(config: dict[str, Any]) -> "SimoncelliHeegerCNN":
        """Creates a SimoncelliHeegerCNN from a config dict."""
        config = deepcopy(config)

        _type = config.pop("_type", "SimoncelliHeegerCNN")
        if _type != "SimoncelliHeegerCNN":
            raise ValueError(
                f"Expected _type to be 'SimoncelliHeegerCNN', but got '{_type}'."
            )

        from_original_model = config.get("from_original_model", False)
        freeze_original_model = config.get("freeze_original_model", False)

        padding = config.get("padding", "same")
        padding_time = config.get("padding_time", "same")

        if from_original_model is not False:
            if isinstance(from_original_model, bool):
                layers = None
            else:
                layers = from_original_model
                
            model = SimoncelliHeegerCNN.from_original_model(
                padding=padding, padding_time=padding_time,layers=layers,
            )

        else:
            model = SimoncelliHeegerCNN(padding=padding, padding_time=padding_time)
        
        if isinstance(freeze_original_model, bool):
            for parameter in model.parameters():
                parameter.requires_grad = not freeze_original_model

        else:
            for name, module in model.named_children():
                if name in freeze_original_model:
                    for parameter in module.parameters():
                        parameter.requires_grad = False

        return model


class Square(nn.Module):
    def forward(self, x):
        return x ** 2


class ChannelNorm(nn.Module):
    def forward(self, x, epsilon: float = 1e-6):
        return x / (torch.sum(x, dim=1, keepdim=True) + epsilon)


class RectifiedSquare(nn.Module):
    def forward(self, x):
        return torch.relu(x) ** 2


