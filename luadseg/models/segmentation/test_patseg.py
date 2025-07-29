import pytest
import torch

from luadseg.models.segmentation.encoder_wrapper import UNI2Encoder
from luadseg.models.segmentation.patseg import PatSeg


@pytest.mark.parametrize("batch_size", [1, 2])
def test_patseg_forward(batch_size):
    """Test if PatSeg can be instantiated and performs a forward pass."""
    encoder = UNI2Encoder()
    model = PatSeg(encoder=encoder, num_classes=6)

    x = torch.randn(batch_size, 3, 224, 224)  # input divisible by patch_size
    out = model(x)

    # Check output shape (B, num_classes, H, W)
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == batch_size
    assert out.shape[1] == model.num_classes
    assert out.shape[2] == x.shape[2]
    assert out.shape[3] == x.shape[3]
