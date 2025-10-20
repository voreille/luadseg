"""
Batch feature-map/heatmap export for all slides in an EmbeddingStore, across multiple model heads.

Edit the CONFIG section at the bottom:
    TASKS = [
        {
            "weights_path": "/path/to/fold_0.pt",
            "embedding_dir": "/path/to/embeds/DHMC/224_10x/uni2",
            "output_folder": "out/fold0",
        },
        # add more tasks here...
    ]

It will loop over each task, load the linear head, iterate over all slides found by
`build_embedding_store_from_dir(slides_root, embedding_dir)`, compute class heatmaps, and
save a figure per slide into the task's output folder.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from histoseg_plugin.storage.factory import build_embedding_store_from_dir
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
from scipy.ndimage import affine_transform
import torch
from tqdm import tqdm

from luadseg.data.constants import ANORAK_CLASS_MAPPING
from luadseg.eval.anorak import LinearSoftmaxHead

# ---------------------------
# Global defaults (edit as needed)
# ---------------------------
SLIDES_DIR = "/home/valentin/workspaces/luadseg/data/raw/DHMC/DHMC_LUAD_corrected"
METADATA_CSV = "/home/valentin/workspaces/luadseg/data/raw/DHMC/MetaData_Release_1.0.csv"

# Model head defaults (UNI2)
NUM_CLASSES = 7

# Heatmap rendering defaults
HEATMAP_LEVEL = 4  # change based on your pyramid
THUMBNAIL_MAX = 500  # px max side for the thumbnail panel

# ---------------------------
# Utilities
# ---------------------------


def load_metadata(metadata_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(metadata_csv)
    if "slide_id" not in df.columns and "File Name" in df.columns:
        df = df.copy()
        df["slide_id"] = df["File Name"].str.replace(".tif", "", regex=False)
    return df


def class_from_slide_id(df: pd.DataFrame, slide_id: str) -> Optional[str]:
    if "slide_id" not in df.columns or "Class" not in df.columns:
        return None
    row = df[df["slide_id"] == slide_id]
    if len(row) != 1:
        return None
    return str(row["Class"].values[0])


def build_linear_head(weights_path: str | Path) -> LinearSoftmaxHead:
    state = torch.load(str(weights_path), map_location="cpu")
    head = LinearSoftmaxHead(
        in_dim=state["in_dim"],
        n_classes=state["n_classes"],
        temperature=state["temperature"],
    )
    # allow either the raw state_dict or a wrapper {"state_dict": ...}
    state_dict = state.get("state_dict", state)
    head.load_state_dict(state_dict)
    head.eval()
    return head


def compute_heatmaps(
    wsi: openslide.OpenSlide,
    coords: np.ndarray,  # (N,2) top-left tile coords at patch_level
    predictions: np.ndarray,  # (N, n_classes)
    patch_size: int,
    patch_level: int,
    n_classes: int,
    heatmap_level: int,
) -> np.ndarray:
    """Return (H, W, n_classes) heatmaps at heatmap_level using bilinear resampling."""
    coords = np.asarray(coords, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float32)
    if predictions.ndim == 1:
        predictions = predictions[:, None]

    # level-0 sizing
    level_downsample_patch = float(wsi.level_downsamples[patch_level])
    patch_size_level0 = patch_size * level_downsample_patch

    # 1) origin/max in level-0 coords (tight bbox around tiles, using top-left)
    min_x0 = float(coords[:, 0].min())
    min_y0 = float(coords[:, 1].min())
    max_x0 = float(coords[:, 0].max() + patch_size_level0)
    max_y0 = float(coords[:, 1].max() + patch_size_level0)

    origin_x0, origin_y0 = min_x0, min_y0

    # 2) grid size in “tile cells”
    grid_w = int(np.ceil((max_x0 - origin_x0) / patch_size_level0))
    grid_h = int(np.ceil((max_y0 - origin_y0) / patch_size_level0))

    # 3) instantiate heatmap grid (one cell per tile) + fill
    heatmaps = np.zeros((grid_h, grid_w, n_classes), dtype=np.float32)

    # map each tile to integer cell indices (use floor)
    x_idx = np.floor(
        (coords[:, 0] - origin_x0) / patch_size_level0).astype(np.int64)
    y_idx = np.floor(
        (coords[:, 1] - origin_y0) / patch_size_level0).astype(np.int64)

    x_idx = np.clip(x_idx, 0, grid_w - 1)
    y_idx = np.clip(y_idx, 0, grid_h - 1)

    for i in range(coords.shape[0]):
        heatmaps[y_idx[i], x_idx[i], :] = predictions[i, :]

    # 4) resample to the requested WSI level
    output_W, output_H = wsi.level_dimensions[heatmap_level]  # (W, H)
    d_out = float(
        wsi.level_downsamples[heatmap_level])  # level-0 px per output px

    # Build affine: output (row,col) -> input (row_in,col_in) on the tile grid
    scale = d_out / patch_size_level0  # how many tile-cells per output pixel
    A = np.array([[scale, 0.0], [0.0, scale]], dtype=np.float64)

    # offset is in input-index units (tile cells), note row=y, col=x ordering
    offset = np.array(
        [
            -origin_y0 / patch_size_level0,  # row offset
            -origin_x0 / patch_size_level0,  # col offset
        ],
        dtype=np.float64)

    out = np.zeros((output_H, output_W, n_classes), dtype=np.float32)
    for c in range(n_classes):
        out[:, :, c] = affine_transform(
            heatmaps[:, :, c],
            matrix=A,
            offset=offset,
            output_shape=(output_H, output_W),
            order=1,  # bilinear
            mode='constant',
            cval=0.0,
            prefilter=True,
        )

    return out


def figure_for_slide(wsi: openslide.OpenSlide,
                     heatmaps: np.ndarray,
                     class_names: List[str],
                     out_png: Path,
                     thumbnail_max: int = THUMBNAIL_MAX) -> None:
    """Save a 2x4 grid: WSI thumbnail + 7 class heatmaps."""
    n_classes = heatmaps.shape[-1]
    assert n_classes == len(
        class_names), "class_names length must match heatmaps channels"

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    # Thumbnail
    dims0 = wsi.level_dimensions[wsi.get_best_level_for_downsample(1)]
    # compute a reasonable thumbnail size keeping aspect
    w0, h0 = dims0
    if w0 >= h0:
        size = (thumbnail_max, int(thumbnail_max * h0 / max(w0, 1)))
    else:
        size = (int(thumbnail_max * w0 / max(h0, 1)), thumbnail_max)
    thumbnail = wsi.get_thumbnail(size)
    axes[0].imshow(thumbnail)
    axes[0].set_title('WSI Thumbnail')
    axes[0].axis('off')

    cmaps = [
        'Blues', 'Reds', 'Greens', 'Purples', 'Oranges', 'viridis', 'plasma'
    ]

    for class_idx in range(n_classes):
        im = axes[class_idx + 1].imshow(heatmaps[..., class_idx],
                                        cmap=cmaps[class_idx % len(cmaps)],
                                        vmin=0,
                                        vmax=1)
        axes[class_idx + 1].set_title(f'{class_idx}: {class_names[class_idx]}')
        axes[class_idx + 1].axis('off')
        plt.colorbar(im, ax=axes[class_idx + 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)


# ---------------------------
# Core pipeline
# ---------------------------


def process_task(
    slides_dir: str | Path,
    embedding_dir: str | Path,
    weights_path: str | Path,
    output_folder: str | Path,
    metadata_df: Optional[pd.DataFrame] = None,
    heatmap_level: int = HEATMAP_LEVEL,
    num_classes: int = NUM_CLASSES,
) -> None:
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build embedding store
    embedding_store = build_embedding_store_from_dir(
        slides_root=str(slides_dir), root_dir=str(embedding_dir))
    slide_ids = embedding_store.slide_ids()

    # Model head
    head = build_linear_head(weights_path)

    # Class names
    class_names = [ANORAK_CLASS_MAPPING[c] for c in range(num_classes)]

    pbar = tqdm(slide_ids, desc=f"Exporting heatmaps -> {out_dir}")
    for slide_id in pbar:
        try:
            feats, coords, attrs = embedding_store.load(slide_id)
            with torch.inference_mode():
                probs = head(torch.tensor(
                    feats, dtype=torch.float32)).softmax(dim=-1).cpu().numpy()

            wsi_path = Path(slides_dir) / attrs['relative_wsi_path']
            wsi = openslide.OpenSlide(str(wsi_path))
            heatmaps = compute_heatmaps(
                wsi,
                coords,
                probs,
                patch_size=int(attrs["patch_size"]),
                patch_level=int(attrs["patch_level"]),
                n_classes=num_classes,
                heatmap_level=heatmap_level,
            )

            # decorate filename with class label if available
            label_suffix = ""
            if metadata_df is not None:
                slide_cls = class_from_slide_id(metadata_df, slide_id)
                if slide_cls is not None:
                    # safe filename part
                    safe = ''.join(ch for ch in slide_cls
                                   if ch.isalnum() or ch in ('-', '_'))
                    label_suffix = f"__{safe}"

            out_png = out_dir / f"{slide_id}{label_suffix}.png"
            figure_for_slide(wsi, heatmaps, class_names, out_png)
            wsi.close()
        except Exception as e:
            pbar.write(f"[WARN] {slide_id}: {e}")
            continue


# ---------------------------
# CONFIG: define your tasks here
# ---------------------------
TASKS: List[Dict[str, str]] = [
    # Example (edit or replace):
    # {
    #     "weights_path":
    #     "/home/valentin/workspaces/luadseg/mlflow/873983245792450036/43c1778b62cb4b1ca1ef0401d2204460/artifacts/heads/fold_0.pt",
    #     "embedding_dir":
    #     "/home/valentin/workspaces/luadseg/data/embeds/DHMC/224_20x/uni2",
    #     "output_folder":
    #     "/home/valentin/workspaces/luadseg/outputs/heatmaps/uni2/224_20x/fold_0",
    # },
    {
        "weights_path":
        '/home/valentin/workspaces/luadseg/mlflow/873983245792450036/c45e9f870f42409889af88a61e029f1e/artifacts/heads/fold_0.pt',
        "embedding_dir":
        "/home/valentin/workspaces/luadseg/data/embeds/DHMC/224_10x/uni2",
        "output_folder":
        "/home/valentin/workspaces/luadseg/outputs/heatmaps/uni2/224_10x/fold_0",
    },
]


def main() -> None:
    if not TASKS:
        print("No TASKS defined. Edit TASKS at the bottom of the script.")
        return

    metadata_df = None
    if METADATA_CSV and Path(METADATA_CSV).exists():
        metadata_df = load_metadata(METADATA_CSV)

    for task in TASKS:
        weights_path = task["weights_path"]
        embedding_dir = task["embedding_dir"]
        output_folder = task["output_folder"]

        process_task(
            slides_dir=SLIDES_DIR,
            embedding_dir=embedding_dir,
            weights_path=weights_path,
            output_folder=output_folder,
            metadata_df=metadata_df,
            heatmap_level=HEATMAP_LEVEL,
            num_classes=NUM_CLASSES,
        )


if __name__ == "__main__":
    main()
