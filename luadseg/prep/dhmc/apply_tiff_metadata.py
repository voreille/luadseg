#!/usr/bin/env python3
"""
apply_tiff_metadata.py
----------------------
Recursively find TIFF files under ROOT_DIR and set the ImageDescription tag using `tifftools`,
with per-file values taken from a CSV.

CSV requirements (simple & flexible):
- Must contain at least two columns for the values:
    - Magnification (default column name: "Magnification")
    - Microns Per Pixel (default column name: "Microns Per Pixel")
- Must include a key column to match files (default: "filename"), which is compared
  to each TIFF's stem (name without extension). You can change this with --csv-key-column.

Examples:
    # In-place update of all *.tif/*.tiff under ./slides using defaults and CSV with columns:
    # filename, Magnification, Microns Per Pixel
    python apply_tiff_metadata.py ./slides ./MetaData.csv --inplace

    # Write outputs to a separate folder, mirroring the input tree
    python apply_tiff_metadata.py ./slides ./MetaData.csv --no-inplace --output-dir ./out

    # If your matching key column is different
    python apply_tiff_metadata.py ./slides ./MetaData.csv --csv-key-column id
"""
import csv
from pathlib import Path
import shutil
import subprocess
from typing import Dict, Iterable, List, Tuple

import click


def load_mapping(
    csv_path: Path,
    key_col: str,
    mag_col: str,
    mpp_col: str,
) -> Dict[str, Tuple[str, str]]:
    """
    Load CSV into a mapping: key -> (magnification, mpp). Values are kept as strings.
    """
    mapping: Dict[str, Tuple[str, str]] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Normalize fieldnames for robust lookups / allow exact key names
        fieldnames = {name: name for name in (reader.fieldnames or [])}
        missing = [
            c for c in (key_col, mag_col, mpp_col) if c not in fieldnames
        ]
        if missing:
            raise click.BadParameter(
                f"CSV is missing required columns: {missing}. Present columns: {list(fieldnames)}"
            )
        for row in reader:
            key = str(row[key_col]).strip()
            if not key:
                continue
            mag = str(row[mag_col]).strip()
            mpp = str(row[mpp_col]).strip()
            mapping[key] = (mag, mpp)
    return mapping


def iter_files(root_dir: Path, patterns: Iterable[str]) -> Iterable[Path]:
    """
    Recursively yield files matching the provided glob patterns (comma-separated, e.g., '*.tif,*.tiff').
    """
    seen = set()
    for pat in patterns:
        for p in root_dir.rglob(pat):
            if p.is_file() and p not in seen:
                seen.add(p)
                yield p


def ensure_destination(src: Path, root_dir: Path, output_dir: Path) -> Path:
    """
    Compute mirrored destination path in output_dir, creating parent dirs.
    """
    rel = src.relative_to(root_dir)
    dst = output_dir.joinpath(rel)
    dst.parent.mkdir(parents=True, exist_ok=True)
    return dst


def run_tifftools_set(tifftools_cmd: str, target_path: Path, description: str,
                      verbose: bool) -> int:
    """
    Run: tifftools set -y -s ImageDescription "<description>" <target_path>
    Returns the subprocess return code.
    """
    cmd = [
        tifftools_cmd,
        "set",
        "-y",
        "-s",
        "ImageDescription",
        description,
        str(target_path),
    ]
    if verbose:
        click.echo(f"[tifftools] {' '.join(cmd)}")
    try:
        res = subprocess.run(cmd,
                             capture_output=not verbose,
                             text=True,
                             check=False)
        if res.returncode != 0 and not verbose:
            click.echo(
                f"tifftools failed for {target_path} (code {res.returncode}): {res.stderr}",
                err=True)
        return res.returncode
    except FileNotFoundError:
        raise click.ClickException(
            f"`{tifftools_cmd}` not found. Install it or point to it with --tifftools-cmd."
        )


@click.command(context_settings=dict(help_option_names=["-h", "--help"],
                                     show_default=True))
@click.argument("root_dir",
                type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("csv_path",
                type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--inplace/--no-inplace",
    default=True,
    help="Edit files in place. If --no-inplace, you must pass --output-dir.")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help=
    "Output directory when --no-inplace is used; input tree will be mirrored.")
@click.option(
    "--pattern",
    default="*.tif,*.tiff",
    help="Comma-separated patterns to search recursively under ROOT_DIR.")
@click.option("--csv-key-column",
              default="filename",
              help="CSV column to match files (compared to each TIFF's stem).")
@click.option("--magnification-column",
              default="Magnification",
              help="CSV column with Magnification value.")
@click.option("--mpp-column",
              default="Microns Per Pixel",
              help="CSV column with Microns Per Pixel value.")
@click.option("--tifftools-cmd",
              default="tifftools",
              help="Path to the `tifftools` executable.")
@click.option("--verbose", is_flag=True, help="Print commands and progress.")
def main(
    root_dir: Path,
    csv_path: Path,
    inplace: bool,
    output_dir: Path,
    pattern: str,
    csv_key_column: str,
    magnification_column: str,
    mpp_column: str,
    tifftools_cmd: str,
    verbose: bool,
):
    """
    Apply ImageDescription to TIFF files found under ROOT_DIR using data from CSV_PATH.
    """
    if not inplace and output_dir is None:
        raise click.BadOptionUsage(
            "--output-dir",
            "You must provide --output-dir when using --no-inplace.")

    if verbose:
        click.echo(f"Loading CSV: {csv_path}")
    mapping = load_mapping(csv_path, csv_key_column, magnification_column,
                           mpp_column)

    patterns = [p.strip() for p in pattern.split(",") if p.strip()]
    files = list(iter_files(root_dir, patterns))
    if verbose:
        click.echo(
            f"Found {len(files)} file(s) under {root_dir} matching {patterns}")

    processed = 0
    skipped_no_match: List[Path] = []
    failed: List[Path] = []

    for src in files:
        key = src.stem  # default key is the filename without extension
        vals = mapping.get(key)
        if not vals:
            skipped_no_match.append(src)
            if verbose:
                click.echo(f"[skip] No CSV row for key '{key}'")
            continue
        mag, mpp = vals

        # Compose ImageDescription string
        # Note: The user provided example had a newline after Magnification; we omit that for robustness.
        description = f"Aperio Fake |AppMag = {mag}|MPP = {mpp}"

        # Decide target path
        target = src
        if not inplace:
            target = ensure_destination(src, root_dir, output_dir)
            # Copy source to target first, then modify
            if verbose:
                click.echo(f"[copy] {src} -> {target}")
            shutil.copy2(src, target)

        rc = run_tifftools_set(tifftools_cmd, target, description, verbose)
        if rc == 0:
            processed += 1
            if verbose:
                click.echo(f"[ok] {target.name}: AppMag={mag}, MPP={mpp}")
        else:
            failed.append(target)

    # Summary
    click.echo(f"\nDone. Updated {processed} file(s).")
    if skipped_no_match:
        click.echo(f"Skipped (no CSV match): {len(skipped_no_match)}")
        if verbose:
            for p in skipped_no_match:
                click.echo(f"  - {p}")
    if failed:
        click.echo(f"Failed: {len(failed)}")
        for p in failed:
            click.echo(f"  - {p}", err=True)


if __name__ == "__main__":
    main()
