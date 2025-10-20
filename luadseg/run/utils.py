import logging
from pathlib import Path

from luadseg.embed.store import get_embeddings_metadata, parquet_fingerprint

log = logging.getLogger(__name__)


def prep_outputs(cfg):
    # tie path to mpp + tile size so 10x/20x don’t collide
    return Path(
        f"data/tiles/ANORAK/tiles_{cfg.prep.tile_size}_{cfg.mag_label}_pad")


def embed_out(cfg, enc_name):
    return Path(
        f"embeds/{enc_name}_anorak_{cfg.prep.tile_size}_{cfg.mag_label}.h5")


def should_skip_prep(cfg) -> bool:
    out_dir = Path(cfg.prep.out_dir)
    has_index = (out_dir / "index.parquet").exists()
    if has_index:
        log.info(f"[prep] skip: {out_dir}/index.parquet already exists")
    return has_index


def should_skip_embed(cfg, enc_name) -> bool:
    pt_path = Path(cfg.embeddings.pt_path)
    if not pt_path.exists():
        return False
    try:
        import torch
        data = torch.load(pt_path, map_location="cpu") 
        enc = data["encoder"]
        ok = enc.get("name") == enc_name
        if ok:
            log.info(f"[embed] skip: {pt_path} already present for {enc_name}")
        return ok
    except Exception:
        return False
