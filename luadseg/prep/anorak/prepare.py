#!/usr/bin/env python3
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

from jsonargparse import ActionConfigFile, ArgumentParser
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

from luadseg.data.constants import ANORAK_CLASS_MAPPING, PATTERN_COLORS

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("prepare_anorak")

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
PADDING_LABEL = 255


def compute_resize_factor(
    source_mpp: Optional[float],
    target_mpp: Optional[float],
    explicit_factor: Optional[float],
) -> float:
    """If target_mpp is given, use factor = source_mpp / target_mpp.
    Else fall back to explicit_factor or 1.0."""
    if target_mpp is not None:
        if source_mpp is None:
            raise ValueError("target_mpp provided but source_mpp is None.")
        return float(source_mpp / target_mpp)
    return float(explicit_factor if explicit_factor is not None else 1.0)


def iter_images(images_dir: Path) -> Iterable[Path]:
    for ext in IMAGE_EXTS:
        yield from images_dir.rglob(f"*{ext}")


def match_mask_for_image(image_path: Path, masks_dir: Path) -> Path:
    """Match mask by stem prefix; raise if not unique."""
    candidates = list(masks_dir.rglob(f"{image_path.stem}*"))
    candidates = [p for p in candidates if p.suffix.lower() in IMAGE_EXTS]
    if len(candidates) == 0:
        raise FileNotFoundError(f"No mask found for {image_path.name}")
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple masks found for {image_path.name}: {candidates}")
    return candidates[0]


def ensure_same_size(img: Image.Image, msk: Image.Image, name: str):
    if img.size != msk.size:
        raise ValueError(
            f"Image/Mask size mismatch for {name}: {img.size} vs {msk.size}")


def maybe_resize(img: Image.Image, factor: float,
                 is_mask: bool) -> Image.Image:
    if abs(factor - 1.0) < 1e-9:
        return img
    w, h = img.size
    new_w = max(1, int(round(w * factor)))
    new_h = max(1, int(round(h * factor)))
    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.LANCZOS
    return img.resize((new_w, new_h), resample=resample)


def pad_to_multiple(img: Image.Image, tile: int,
                    is_mask: bool) -> Tuple[Image.Image, Tuple[int, int]]:
    """Pad right/bottom so width/height are multiples of tile (no border loss). Returns (padded_img, (pad_right, pad_bottom))."""
    w, h = img.size
    pad_r = (tile - (w % tile)) % tile
    pad_b = (tile - (h % tile)) % tile
    if pad_r == 0 and pad_b == 0:
        return img, (0, 0)
    fill = PADDING_LABEL if is_mask else (255, 255, 255)
    padded = ImageOps.expand(img, border=(0, 0, pad_r, pad_b), fill=fill)
    return padded, (pad_r, pad_b)


def center_crop_to_multiple(
        img: Image.Image,
        tile: int) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """Center-crop to nearest multiple of tile. Returns (cropped_img, (left, top, right, bottom) crop box)."""
    w, h = img.size
    target_w = w - (w % tile)
    target_h = h - (h % tile)
    left = (w - target_w) // 2
    top = (h - target_h) // 2
    right = left + target_w
    bottom = top + target_h
    return img.crop((left, top, right, bottom)), (left, top, right, bottom)


def tile_grid(width: int, height: int, tile: int,
              stride: Optional[int]) -> List[Tuple[int, int, int, int]]:
    """Return list of crop boxes (left, top, right, bottom)."""
    s = stride if stride is not None else tile
    xs = list(range(0, width - tile + 1, s))
    ys = list(range(0, height - tile + 1, s))
    if not xs:
        xs = [0]
    if not ys:
        ys = [0]
    boxes = []
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            boxes.append((x, y, x + tile, y + tile))
    return boxes


def bincount_ratios(mask_arr_flat: np.ndarray, num_classes: int,
                    ignore_label: Optional[int]) -> Tuple[np.ndarray, int]:

    if mask_arr_flat.max() > num_classes - 1 or mask_arr_flat.min() < 0:
        raise ValueError(
            f"Mask contains labels outside of valid range [0, {num_classes - 1}]: {mask_arr_flat.min()} - {mask_arr_flat.max()}"
        )
    counts = np.bincount(mask_arr_flat, minlength=num_classes)

    area = float(mask_arr_flat.size)
    ratios = counts.astype(np.float64) / area
    dominant = int(np.argmax(ratios))
    if ignore_label is not None and dominant == ignore_label:
        # pick next best non-ignore if possible
        order = np.argsort(-ratios)
        for c in order:
            if c != ignore_label:
                dominant = int(c)
                break
    return ratios, dominant


def get_color_map() -> dict:
    """Return color map indexed by class ID using the above mappings."""

    color_map = {
        cls_id: PATTERN_COLORS[name]
        for cls_id, name in ANORAK_CLASS_MAPPING.items()
    }
    color_map.update({255: (255, 255, 255)})
    return color_map


def colorize_mask(mask: Image.Image) -> Image.Image:
    """Convert class ID mask (H, W) → RGB PIL Image."""
    np_mask = np.array(mask)
    color_map = get_color_map()
    lut = np.zeros((256, 3), dtype=np.uint8)
    for k, rgb in color_map.items():
        lut[int(k)] = rgb
    rgb = lut[np_mask.astype(np.uint8)]
    return Image.fromarray(rgb, mode="RGB")


def overlay_tile_with_mask(tile: Image.Image,
                           mask: Image.Image,
                           alpha: float = 0.4) -> Image.Image:
    """
    Overlay a colorized mask on top of the original tile for debugging.
    `alpha` controls transparency of the mask (0 = invisible, 1 = opaque).
    """
    colored_mask = colorize_mask(mask).convert("RGBA")
    tile_rgba = tile.convert("RGBA")

    # blend the two images
    overlay = Image.blend(tile_rgba, colored_mask, alpha)
    return overlay


def main(
    images_dir: str,
    masks_dir: str,
    out_dir: str,
    dataset_name: str = "anorak",
    source_mpp: Optional[
        float] = 0.5,  # typical 20x ~ 0.5 µm/px; adjust if you know exact
    target_mpp: Optional[float] = None,  # set this OR resize_factor
    resize_factor: Optional[float] = 1.0,
    tile_size: int = 256,
    stride: Optional[
        int] = None,  # default None => non-overlapping: stride = tile_size
    pad_mode: str = "pad",  # "pad" (recommended) or "crop"
    save_tiles: bool = True,
    save_overlay: bool = True,
    num_classes: int = 7,  # adjust to your mask schema
    ignore_label: Optional[int] = 0,  # e.g., 0 = background/tissue
    min_foreground_ratio: float = 0.0,
    min_valid_pixel_ratio: float = 0.0,
    index_path: Optional[Union[str, None]] = None,
):
    images_dir_path = Path(images_dir).resolve()
    masks_dir_path = Path(masks_dir).resolve()
    out_dir_path = Path(out_dir).resolve()
    img_out = out_dir_path / "image"
    msk_out = out_dir_path / "mask"
    overlay_out = out_dir_path / "overlay"
    overlay_thumbnail_out = out_dir_path / "overlay_thumbnail"

    if save_tiles:
        img_out.mkdir(parents=True, exist_ok=True)
        msk_out.mkdir(parents=True, exist_ok=True)

    if save_overlay:
        overlay_out.mkdir(parents=True, exist_ok=True)
        overlay_thumbnail_out.mkdir(parents=True, exist_ok=True)

    resize_factor = compute_resize_factor(source_mpp, target_mpp,
                                          resize_factor)
    log.info(
        f"Resize factor: {resize_factor:.6f}  (source_mpp={source_mpp}, target_mpp={target_mpp})"
    )

    rows = []
    roi_ratio_accum: Dict[str, List[np.ndarray]] = {}

    image_files = [p for p in iter_images(images_dir_path)]
    logging.info(f"Found {len(image_files)} images under {images_dir_path}")
    if len(image_files) == 0:
        raise FileNotFoundError(f"No images found under {images_dir_path}")

    list_no_valid_mask = []
    tile_count = 0
    for img_path in tqdm(image_files, desc="Tiling"):
        try:
            msk_path = match_mask_for_image(img_path, masks_dir_path)
            img = Image.open(img_path).convert("RGB")
            msk = Image.open(msk_path)  # keep as labeled integers
            ensure_same_size(img, msk, img_path.name)

            # resize
            img = maybe_resize(img, resize_factor, is_mask=False)
            msk = maybe_resize(msk, resize_factor, is_mask=True)

            # pad or crop to multiple of tile_size
            if pad_mode == "pad":
                img, (pad_r, pad_b) = pad_to_multiple(img,
                                                      tile_size,
                                                      is_mask=False)
                msk, _ = pad_to_multiple(msk, tile_size, is_mask=True)
                x0 = y0 = 0  # top-left offset after padding is zero
            elif pad_mode == "crop":
                img, (left, top, right,
                      bottom) = center_crop_to_multiple(img, tile_size)
                msk = msk.crop((left, top, right, bottom))
                x0, y0 = left, top
            else:
                raise ValueError("pad_mode must be 'pad' or 'crop'.")

            W, H = img.size
            boxes = tile_grid(W, H, tile_size, stride)

            # identify ROI / WSI
            roi_id = img_path.stem
            wsi_id = roi_id  # adjust if you have true WSI IDs

            # loop tiles
            have_valid_tiles = False
            for idx, (x, y, r, b) in enumerate(boxes):
                tile = img.crop((x, y, r, b))
                msk_tile = msk.crop((x, y, r, b))
                m_arr = np.array(msk_tile, dtype=np.int32)

                valid_ratio = np.mean(m_arr != PADDING_LABEL)

                if valid_ratio <= float(min_valid_pixel_ratio):
                    continue

                ratios, dominant = bincount_ratios(
                    m_arr[m_arr != PADDING_LABEL],
                    num_classes=num_classes,
                    ignore_label=ignore_label)
                foreground_ratio = 1.0 - (ratios[ignore_label]
                                          if ignore_label is not None else 0.0)
                if foreground_ratio <= float(min_foreground_ratio):
                    continue  # skip mostly-background tiles

                have_valid_tiles = True

                tile_row = (y // (stride if stride else tile_size))
                tile_col = (x // (stride if stride else tile_size))
                # tile_idx = tile_row * (
                #     W // (stride if stride else tile_size)) + tile_col

                tile_id = f"{roi_id}_r{tile_row}_c{tile_col}"

                if save_tiles:
                    tile_name = f"{tile_id}.png"
                    mask_name = f"{tile_id}.png"
                    tile.save(img_out / tile_name, "PNG")
                    msk_tile.save(msk_out / mask_name, "PNG")
                    if save_overlay:
                        tile_overlay = overlay_tile_with_mask(tile, msk_tile)
                        tile_overlay_name = f"{tile_id}__{ANORAK_CLASS_MAPPING[dominant]}.png"
                        tile_overlay.save(overlay_out / tile_overlay_name,
                                          "PNG")
                        tile_overlay_thumbnail = tile_overlay.resize(
                            (112, 112), Image.Resampling.LANCZOS)
                        tile_overlay_thumbnail.save(
                            overlay_thumbnail_out / tile_name, "PNG")

                    tile_path = str(
                        (img_out /
                         tile_name).resolve().relative_to(out_dir_path))
                    mask_tile_path = str(
                        (msk_out /
                         mask_name).resolve().relative_to(out_dir_path))
                else:
                    tile_path = ""
                    mask_tile_path = ""

                row = {
                    "dataset":
                    dataset_name,
                    "wsi_id":
                    wsi_id,
                    "roi_id":
                    roi_id,
                    "tile_row":
                    int(tile_row),
                    "tile_col":
                    int(tile_col),
                    "tile_id":
                    tile_id,
                    "tile_idx":
                    int(tile_count),
                    "x":
                    int(x + x0),
                    "y":
                    int(y + y0),
                    "tile_size_px":
                    int(tile_size),
                    "stride_px":
                    int(stride if stride else tile_size),
                    "resize_factor":
                    float(resize_factor),
                    "target_mpp":
                    float(target_mpp) if target_mpp is not None else np.nan,
                    "tile_path":
                    tile_path,
                    "mask_tile_path":
                    mask_tile_path,
                    "dominant_label":
                    int(dominant)
                }
                # add per-class ratios
                for k in range(num_classes):
                    row[f"ratio_{k}"] = float(ratios[k])
                rows.append(row)
                tile_count += 1

                # accumulate for ROI-level GT ratios
                roi_ratio_accum.setdefault(roi_id, []).append(ratios)

            if not have_valid_tiles:
                list_no_valid_mask.append(msk_path)

        except Exception as e:
            log.error(f"[{img_path.name}] {e}")
            continue

    if not rows:
        log.warning("No tiles produced (all filtered?)")
        return

    df = pd.DataFrame(rows)

    # ROI-level ground-truth ratios (mean over tiles)
    roi_rows = []
    for roi_id, rlist in roi_ratio_accum.items():
        R = np.stack(rlist, axis=0)  # [T, C]
        mean_ratios = R.mean(axis=0)
        entry: dict[str, Union[str, float]] = {"roi_id": roi_id}
        for k, v in enumerate(mean_ratios.tolist()):
            entry[f"ratio_{k}"] = float(v)
        roi_rows.append(entry)
    df_roi = pd.DataFrame(roi_rows)

    out_dir_path.mkdir(parents=True, exist_ok=True)
    index_path_path = Path(index_path) if index_path else (out_dir_path /
                                                           "index.parquet")
    gt_ratios_path = out_dir_path / "roi_ratios.csv"

    # Write Parquet (requires pyarrow); fall back to CSV if needed
    try:
        df.to_parquet(index_path_path, index=False)
        log.info(f"Wrote index: {index_path_path}")
    except Exception as e:
        csv_fallback = index_path_path.with_suffix(".csv")
        df.to_csv(csv_fallback, index=False)
        log.warning(f"Parquet failed ({e}); wrote CSV instead: {csv_fallback}")

    df_roi.to_csv(gt_ratios_path, index=False)
    log.info(f"Wrote ROI-level GT ratios: {gt_ratios_path}")

    # Save a tiny meta.json for provenance
    meta = {
        "dataset": dataset_name,
        "tile_size_px": tile_size,
        "stride_px": stride if stride else tile_size,
        "resize_factor": resize_factor,
        "target_mpp": target_mpp,
        "num_classes": num_classes,
        "ignore_label": ignore_label,
        "images_dir": str(images_dir_path),
        "masks_dir": str(masks_dir_path),
        "pad_mode": pad_mode,
        "save_tiles": save_tiles,
    }
    (out_dir_path / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info(f"Wrote meta: {out_dir_path / 'meta.json'}")
    log.info(f"Done. Total tiles: {len(df)}")
    log.info(f"Total ROIs with non valid tiles: {len(list_no_valid_mask)}")
    with open(out_dir_path / "roi_with_no_valid_tiles.txt", "w") as f:
        for p in list_no_valid_mask:
            f.write(f"{p}\n")


def run_prep(cfg):
    main(**cfg)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="ANORAK tiling → tiles + Parquet index")
    parser.add_argument("--config",
                        action=ActionConfigFile,
                        help="Optional YAML config file")
    parser.add_argument("--images_dir",
                        required=True,
                        help="Directory with ROI images")
    parser.add_argument("--masks_dir",
                        required=True,
                        help="Directory with ROI masks (same names)")
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory (will contain image/, mask/, index.parquet)")
    parser.add_argument("--dataset_name", default="anorak")
    parser.add_argument("--source_mpp", type=float, default=0.5)
    parser.add_argument("--target_mpp", type=float, default=None)
    parser.add_argument("--resize_factor", type=float, default=1.0)
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--pad_mode", choices=["pad", "crop"], default="pad")
    parser.add_argument("--save_tiles", type=bool, default=True)
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--ignore_label", type=int, default=0)
    parser.add_argument("--min_foreground_ratio", type=float, default=0.0)
    parser.add_argument("--min_valid_pixel_ratio", type=float, default=0.0)
    parser.add_argument("--index_path", type=str, default=None)

    ns = parser.parse_args()
    kwargs = vars(ns).copy()
    kwargs.pop("config", None)
    main(**kwargs)
