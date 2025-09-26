# sslbench/embed/store.py
"""HDF5/Parquet API (read/write/index)."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import h5py
import numpy as np
import pandas as pd


def parquet_fingerprint(index_path: Path,
                        cols: Optional[Iterable[str]] = None) -> str:
    """Deterministic sha256 hash of the Parquet index (optionally selecting columns)."""
    df = pd.read_parquet(index_path)
    if cols:
        df = df[list(cols)]
    df = df.sort_values(list(df.columns)).reset_index(drop=True)
    payload = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def create_embeddings_file(
    path: Path,
    encoder_metadata: Dict[str, Any],
    dataset_name: Optional[str] = None,
    emb_dim: int = 1536,
) -> None:
    with h5py.File(path, 'w') as f:
        meta = f.create_group('meta')
        meta.create_dataset('encoder.json',
                            data=json.dumps(encoder_metadata, indent=2),
                            dtype=h5py.string_dtype(encoding='utf-8'))
        if dataset_name:
            meta.attrs['dataset_name'] = dataset_name
        meta.attrs['emb_dim'] = emb_dim
        meta.attrs['created_with'] = 'sslbench'
        f.create_group('embeddings')


def write_group_embeddings(
    path: Path,
    split: str,
    roi_id: str,
    emb: np.ndarray,  # [N, D] float32
    tile_idx: np.ndarray,  # [N] int32
    valid_fraction: Optional[np.ndarray] = None,  # [N] float16/float32
    index_fingerprint: Optional[str] = None,
    compression: str = 'lzf',
) -> None:
    """Create /embeddings/<split>/<roi_id> with E, tile_idx, (optional) valid_fraction."""
    emb = np.asarray(emb, dtype=np.float32)
    tile_idx = np.asarray(tile_idx, dtype=np.int32)
    if valid_fraction is not None:
        valid_fraction = np.asarray(valid_fraction, dtype=np.float16)

    # Defensive: sort by tile_idx to guarantee order
    order = np.argsort(tile_idx, kind="stable")
    emb = emb[order]
    tile_idx = tile_idx[order]
    if valid_fraction is not None:
        valid_fraction = valid_fraction[order]

    with h5py.File(path, 'a') as f:
        gsplit = f.require_group(f"embeddings/{split}")
        if roi_id in gsplit:
            del gsplit[roi_id]
        g = gsplit.create_group(roi_id)

        g.create_dataset("E",
                         data=emb,
                         chunks=(min(len(emb), 1024), emb.shape[1]),
                         compression=compression)
        g.create_dataset("tile_idx",
                         data=tile_idx,
                         chunks=True,
                         compression=compression)
        if valid_fraction is not None:
            g.create_dataset("valid_fraction",
                             data=valid_fraction,
                             chunks=True,
                             compression=compression)

        g.attrs["order"] = "tile_idx_ascending"
        if index_fingerprint:
            g.attrs["index_fingerprint"] = index_fingerprint


def read_group_embeddings(path: Path, split: str,
                          roi_id: str) -> Dict[str, np.ndarray]:
    with h5py.File(path, 'r') as f:
        g = f[f"embeddings/{split}/{roi_id}"]
        out = {
            "E": g["E"][:],
            "tile_idx": g["tile_idx"][:],
        }
        if "valid_fraction" in g:
            out["valid_fraction"] = g["valid_fraction"][:]
        return out


def get_embeddings_metadata(path: Path) -> Dict[str, Any]:
    with h5py.File(path, 'r') as f:
        encoder_json = f['meta/encoder.json'][()]
        if isinstance(encoder_json, bytes):
            encoder_json = encoder_json.decode('utf-8')
        encoder_metadata = json.loads(encoder_json)
        meta = dict(f['meta'].attrs)
        splits = list(f['embeddings'].keys()) if 'embeddings' in f else []
        return {
            'encoder': encoder_metadata,
            'file_metadata': meta,
            'splits': splits
        }


def write_embeddings_batch(
    path: Path,
    split: str,
    embeddings: np.ndarray,  # [N, D] float32/float64
    compression: str = "lzf",
) -> None:
    """
    Write a single batch of embeddings to /embeddings/<split>/E.

    Overwrites any existing data for the split.

    Args:
        path: HDF5 file path (created beforehand via create_embeddings_file).
        split: Split name under /embeddings.
        embeddings: Array of shape [N, D].
        compression: h5py compression algorithm (default: 'lzf').
    """
    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError(
            f"`embeddings` must be 2D [N, D], got shape {emb.shape!r}")

    with h5py.File(path, "a") as f:
        # Ensure /embeddings exists (create_embeddings_file should have created it)
        gemb = f.require_group("embeddings")

        # If the split group already exists, delete and recreate it to overwrite cleanly
        if split in gemb:
            del gemb[split]
        gsplit = gemb.create_group(split)

        # Create dataset with reasonable chunking along N
        chunk_rows = min(len(emb), 1024) if len(emb) > 0 else 1
        gsplit.create_dataset(
            "E",
            data=emb,
            chunks=(chunk_rows, emb.shape[1]),
            compression=compression,
        )

        # Some helpful attributes
        gsplit.attrs["order"] = "as_written"
        gsplit.attrs["num_rows"] = emb.shape[0]
        gsplit.attrs["emb_dim"] = emb.shape[1]
