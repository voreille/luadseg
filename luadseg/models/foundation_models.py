import os

from dotenv import load_dotenv
from huggingface_hub import login
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch
from torchvision import transforms

load_dotenv()


def load_model(model_name, device, apply_torch_scripting=True):
    """Load the model dynamically based on the model name."""

    if model_name == "H-optimus-0":
        login(token=os.getenv("HUGGING_FACE_TOKEN"))
        model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-0",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False,
        )
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.707223, 0.578729, 0.703617),
                    std=(0.211883, 0.230117, 0.177517),
                ),
            ]
        )
        embedding_dim = 1536
        autocast_dtype = torch.float16
    elif model_name == "H-optimus-1":
        model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-1",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False,
        )
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.707223, 0.578729, 0.703617),
                    std=(0.211883, 0.230117, 0.177517),
                ),
            ]
        )
        embedding_dim = 1536
        autocast_dtype = torch.float16

    elif model_name == "UNI2":
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
        model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
        preprocess = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        embedding_dim = 1536
        autocast_dtype = torch.bfloat16
    elif "prov-gigapath" == model_name:
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        preprocess = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        embedding_dim = 1536
        autocast_dtype = torch.float16
    else:
        raise ValueError(f"Model {model_name} is not supported.")

    model.to(device)
    model.eval()

    if apply_torch_scripting:
        # Apply torch tracing if specified
        print("Applying torch scripting...")
        scripted_model = torch.jit.script(model)
        scripted_model.to(device)
        return scripted_model, preprocess, embedding_dim, autocast_dtype

    return model, preprocess, embedding_dim, autocast_dtype
