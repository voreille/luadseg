# TODO: check if input size not multiple of 256 or 128 generalizes

from typing import List, Tuple, Union

from dotenv import load_dotenv
import timm
import torch
from torch import nn
import torch.nn.functional as F

load_dotenv()


class BaseEncoder(nn.Module):
    """Base class for encoders with a common interface."""

    def __init__(self, extract_layers: Union[Tuple[int], List[int]] = (1, 2, 3, 4)) -> None:
        super().__init__()
        self.extract_layers = tuple(extract_layers)

    def forward(self, x):
        raise NotImplementedError("Subclasses should implement this method.")

    def forward_intermediates(
        self, x, indices=None, norm=True, stop_early=False, intermediates_only=False
    ):
        raise NotImplementedError("Subclasses should implement this method.")


class UNI2Encoder(BaseEncoder):
    def __init__(
        self,
        extract_layers: Union[Tuple[int], List[int]] = (6, 12, 18, 24),
    ) -> None:
        super().__init__()
        timm_kwargs = {
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }
        self.encoder = timm.create_model(
            "hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs
        )
        self.embed_dim = 1536
        self.autocast_dtype = torch.bfloat16
        self.extract_layers = tuple(extract_layers)
        self.depth = 24
        self.patch_size = 14
        self.input_channels = 3

        # Define training crop → resize ratio
        crop_size = 256
        resize_size = 224
        self.resize_ratio = resize_size / crop_size

    def _resize_input(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape

        # Compute expected resize size
        resized_H = int(round(H * self.resize_ratio))
        resized_W = int(round(W * self.resize_ratio))

        # Round to nearest multiple of patch_size (optional but clean)
        resized_H = (resized_H // self.patch_size) * self.patch_size
        resized_W = (resized_W // self.patch_size) * self.patch_size

        return F.interpolate(x, size=(resized_H, resized_W), mode="bilinear", align_corners=False)

    def forward(self, x):
        x = self._resize_input(x) # UNI2 was trained like this, and for the UNet I need it to make the shape works
        return self.encoder.forward_intermediates(
            x,
            indices=[i - 1 for i in self.extract_layers],
            norm=True,
            stop_early=True,
            intermediates_only=True,
        )
