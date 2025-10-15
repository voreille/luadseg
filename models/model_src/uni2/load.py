import os
from typing import Union

from dotenv import find_dotenv, load_dotenv
from huggingface_hub import login
import timm
from timm.data import resolve_data_config  # type: ignore
from timm.data.transforms_factory import create_transform
import torch


# pretrained=True needed to load UNI2-h weights (and download weights for the first time)
def load(*, cfg, device: str = "cuda", img_size: Union[int, None] = None, **_):
    load_dotenv(find_dotenv(usecwd=True))

    # Accept either env var name
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        login(token=token)  # explicit; avoids interactive prompt
    # else: you might already be logged in (cached); otherwise HF will 401
    timm_kwargs = {
        'img_size': 224,
        'patch_size': 14,
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5,
        'embed_dim': 1536,
        'mlp_ratio': 2.66667 * 2,
        'num_classes': 0,
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked, # type: ignore
        'act_layer': torch.nn.SiLU,
        'reg_tokens': 8,
        'dynamic_img_size': True
    }
    model = timm.create_model("hf-hub:MahmoodLab/UNI2-h",
                              pretrained=True,
                              **timm_kwargs)
    preprocess = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model))
    model.eval()

    emb_dim = 1536
    dtype = torch.bfloat16
    meta = {}
    return model, preprocess, emb_dim, dtype, meta
