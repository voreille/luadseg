from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

WEIGHTS = "/mnt/nas7/data/Personal/Darya/Checkpoints/superpixel_cluster_clean/no_cluster_round2_queue32768_alpha0.01_beta0.005_epochs100_moco_superpixel/best_model_epoch96.pth"
ARCH = "resnet50"
SD_KEY = "encoder_q"
STRIP_PREFIX = "module."


def _imagenet_preprocess():
    return T.Compose([
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


def _extract_substate(sd: Dict[str, torch.Tensor], sd_key: Optional[str],
                      strip_prefix: Optional[str]):

    if "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if "model_state_dict" in sd and isinstance(sd["model_state_dict"], dict):
        sd = sd["model_state_dict"]
    if sd_key:
        pref = f"{sd_key}."
        sd = {k[len(pref):]: v for k, v in sd.items() if k.startswith(pref)}
    if strip_prefix:
        pref = str(strip_prefix)
        sd = {
            (k[len(pref):] if k.startswith(pref) else k): v
            for k, v in sd.items()
        }
    for k in list(sd.keys()):
        if k.startswith(("fc.", "classifier.", "head.")):
            sd.pop(k, None)
    return sd


def load(*, cfg, device: str = "cuda", img_size: Union[int, None] = None, **_):
    model = models.__dict__.get(ARCH)(128)
    # before = copy.deepcopy(model.state_dict())

    state = torch.load(Path(WEIGHTS), map_location="cpu")
    state = _extract_substate(state, SD_KEY, STRIP_PREFIX)
    model.load_state_dict(state, strict=False)

    # for k, v in model.state_dict().items():
    #     if not torch.equal(before[k], v):
    #         print(f"Parameter {k} changed.")

    model.fc = nn.Identity()
    model.eval().to(device)
    emb_dim = int(getattr(model, "num_features", 0))
    preprocess = _imagenet_preprocess()

    emb_dim = 2048
    dtype = None
    meta = {}
    return model, preprocess, emb_dim, dtype, meta
