# import copy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from luadseg.models.foundation_models import load_model as _foundation_load


def foundation_loader(*,
                      name: str,
                      apply_torch_scripting: bool = True,
                      device: str = "cuda",
                      **_) -> Tuple[nn.Module, Any, int, torch.dtype]:
    return _foundation_load(name,
                            device=device,
                            apply_torch_scripting=apply_torch_scripting)


def _imagenet_preprocess():
    import torchvision.transforms as T
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


def tv_models_state_dict_loader(
        *,
        arch: str,
        weights: str,
        sd_key: Optional[str] = None,
        strip_prefix: Optional[str] = None,
        device: str = "cuda",
        **_) -> Tuple[nn.Module, Any, int, torch.dtype]:
    import torchvision.models as models
    model = models.__dict__.get(arch)(128)
    # before = copy.deepcopy(model.state_dict())

    state = torch.load(Path(weights), map_location="cpu")
    state = _extract_substate(state, sd_key, strip_prefix)
    model.load_state_dict(state, strict=False)

    # for k, v in model.state_dict().items():
    #     if not torch.equal(before[k], v):
    #         print(f"Parameter {k} changed.")

    model.fc = nn.Identity()
    model.eval().to(device)
    emb_dim = int(getattr(model, "num_features", 0))
    preprocess = _imagenet_preprocess()
    autocast_dtype = torch.float16 if "cuda" in device else torch.float32

    return model, preprocess, emb_dim, autocast_dtype
