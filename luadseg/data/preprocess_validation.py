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


def tile_image(image_id, image_dir, mask_dir, output_dir, tile_size=256):
    """Tile the image and mask into smaller patches."""
    # Generate tiles
    image = load_image(image_id, image_dir)
    mask = load_image(image_id, mask_dir)

    output_mask_dir = output_dir / "masks"
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    output_image_dir = output_dir / "images"
    output_image_dir.mkdir(parents=True, exist_ok=True)

    image_width, image_height = image.size

    if image_width < tile_size or image_height < tile_size:
        raise ValueError(f"Image {image_id} is smaller than tile size {tile_size}x{tile_size}")

    stride_x = calculate_stride(image_width, tile_size)
    stride_y = calculate_stride(image_height, tile_size)

    if stride_x / tile_size < 0.5 or stride_y / tile_size < 0.5:
        logger.warning(
            f"Stride for image {image_id} is too small compared to tile size {tile_size}x{tile_size}"
        )


    x_positions = list(range(0, image_width - tile_size + 1, stride_x or tile_size))
    y_positions = list(range(0, image_height - tile_size + 1, stride_y or tile_size))


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
@click.option("--images-dir", default="data/images", help="Directory containing images.")
@click.option("--masks-dir", default="data/masks", help="Directory containing masks.")
@click.option("--split-csv", default="data/split_df.csv", help="Path to the split CSV file.")
@click.option("--tile-size", default=256, help="Size of the tiles to create.")
@click.option(
    "--output-dir", default="data/processed", help="Directory to save the processed data."
)
def main(images_dir, masks_dir, split_csv, output_dir, tile_size):
    """Simple CLI program to greet someone"""
    split_df = pd.read_csv(split_csv)
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    output_dir = Path(output_dir)

    # Create output directory if it doesn't exist
    val_ids = split_df[split_df["is_val"]]["image_id"].unique().tolist()

    tiles_stats = []
    for image_id in tqdm(val_ids, desc="Processing images"):
        try:
            stats = tile_image(image_id, images_dir, masks_dir, output_dir, tile_size=tile_size)
            tiles_stats.append(stats)
        except Exception as e:
            logger.error(f"Error processing {image_id}: {e}")
            continue

    if tiles_stats:
        stats_df = pd.DataFrame(tiles_stats)
        stats_df.to_csv(output_dir / "tile_statistics.csv", index=False)
        logger.info(f"Tile statistics saved to {output_dir / 'tile_statistics.csv'}")


if __name__ == "__main__":
    main()
