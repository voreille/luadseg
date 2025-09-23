import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_image(image_id, dir):
    image_path_match = list(dir.glob(f"{image_id}*"))
    if len(image_path_match) > 1:
        raise ValueError(f"Multiple images found for {image_id} in {dir}")
    if len(image_path_match) == 0:
        raise FileNotFoundError(f"Image {image_id} not found in {dir}")

    image_path = image_path_match[0]

    return Image.open(image_path)


def center_crop_image(image, target_width, target_height):
    image_width, image_height = image.size
    left = (image_width - target_width) // 2
    top = (image_height - target_height) // 2
    right = (image_width + target_width) // 2
    bottom = (image_height + target_height) // 2
    return image.crop((left, top, right, bottom))


def tile_image(image_id,
               image_dir,
               mask_dir,
               output_dir,
               tile_size=256,
               resize_factor=None):
    """Tile the image and mask into smaller patches."""
    # Generate tiles
    image = load_image(image_id, image_dir)
    mask = load_image(image_id, mask_dir)

    output_mask_dir = output_dir / "mask"
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    output_image_dir = output_dir / "image"
    output_image_dir.mkdir(parents=True, exist_ok=True)

    image_width, image_height = image.size
    mask_width, mask_height = mask.size

    if image_width != mask_width or image_height != mask_height:
        raise ValueError(
            f"Image {image_id} and mask {image_id} must have the same dimensions"
        )

    if resize_factor:
        new_width = np.round(resize_factor * image_width).astype(int)
        new_height = np.round(resize_factor * image_height).astype(int)
        image = image.resize((new_width, new_height), resample=Image.LANCZOS)
        mask = mask.resize((new_width, new_height), resample=Image.NEAREST)

    image_width, image_height = image.size

    if image_width < tile_size or image_height < tile_size:
        raise ValueError(
            f"Image {image_id} ({image_width}x{image_height}) is smaller than tile size {tile_size}x{tile_size}"
        )
    # Make the image exactly croppable
    target_width = image_width - (image_width % tile_size)
    target_height = image_height - (image_height % tile_size)

    image = center_crop_image(image, target_width, target_height)
    mask = center_crop_image(mask, target_width, target_height)

    stride_x = tile_size
    stride_y = tile_size

    x_positions = list(
        range(0, image_width - tile_size + 1, stride_x or tile_size))
    y_positions = list(
        range(0, image_height - tile_size + 1, stride_y or tile_size))

    ratio_overlap_x = (1 - stride_x / tile_size)
    ratio_overlap_y = (1 - stride_y / tile_size)

    # If image is smaller than tile_size in any dimension, use full image
    if not x_positions:
        x_positions = [0]
    if not y_positions:
        y_positions = [0]

    for i, left in enumerate(x_positions):
        for j, top in enumerate(y_positions):
            right = min(left + tile_size, image_width)
            bottom = min(top + tile_size, image_height)

            # Crop the tile
            tile = image.crop((left, top, right, bottom))
            mask_tile = mask.crop((left, top, right, bottom))
            # Save the tile
            tile_id = f"tile_{i}_{j}"
            tile_output_path = output_image_dir / f"{image_id}_{tile_id}.png"
            tile.save(tile_output_path, "PNG")

            mask_tile_output_path = output_mask_dir / f"{image_id}_{tile_id}.png"
            mask_tile.save(mask_tile_output_path, "PNG")

    return {
        "image_id": image_id,
        "tile_size": tile_size,
        "stride_x": stride_x,
        "stride_y": stride_y,
        "ratio_overlap_x": ratio_overlap_x,
        "ratio_overlap_y": ratio_overlap_y,
        "tiles_created": len(x_positions) * len(y_positions),
    }


def calculate_stride(image_dim, tile_size):
    """Calculate stride to align tiles symmetrically with borders."""
    if image_dim <= tile_size:
        return 0  # Single tile, no stride needed

    n_tiles = np.ceil(image_dim / tile_size)
    if n_tiles == 1:
        return 0  # Single tile, no stride needed
    total_stride_space = image_dim - tile_size * n_tiles
    stride = tile_size + total_stride_space // (n_tiles - 1)
    return int(stride)


@click.command()
@click.option("--images-dir",
              default="data/images",
              help="Directory containing images.")
@click.option("--masks-dir",
              default="data/masks",
              help="Directory containing masks.")
@click.option("--tile-size", default=256, help="Size of the tiles to create.")
@click.option("--resize-factor",
              default=1.0,
              help="Factor by which to resize images.")
@click.option("--output-dir",
              default="data/processed",
              help="Directory to save the processed data.")
def main(images_dir, masks_dir, output_dir, tile_size, resize_factor):
    """Simple CLI program to greet someone"""
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    output_dir = Path(output_dir)

    image_ids = [f.stem for f in images_dir.glob("*.png")]

    error_df = pd.DataFrame(columns=["image_id", "error"])
    tiles_stats = []
    for image_id in tqdm(image_ids, desc="Processing images"):
        try:
            stats = tile_image(image_id,
                               images_dir,
                               masks_dir,
                               output_dir,
                               tile_size=tile_size,
                               resize_factor=resize_factor)
            tiles_stats.append(stats)
        except Exception as e:
            logger.error(f"Error processing {image_id}: {e}")
            error_df = pd.concat(
                [
                    error_df,
                    pd.DataFrame({
                        "image_id": image_id,
                        "error": str(e)
                    },
                                 index=[0]),
                ],
                ignore_index=True,
            )

            continue

    if tiles_stats:
        stats_df = pd.DataFrame(tiles_stats)
        stats_df.to_csv(output_dir / "tile_statistics.csv", index=False)
        logger.info(
            f"Tile statistics saved to {output_dir / 'tile_statistics.csv'}")

    if len(error_df) > 0:
        error_df.to_csv(output_dir / "tile_errors.csv", index=False)
        logger.info(f"Errors saved to {output_dir / 'tile_errors.csv'}")


if __name__ == "__main__":
    main()
