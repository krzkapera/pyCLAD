"""Precomputes the 14x14 label maps exactly as the reference derives them.

Usage: visa_reference_grid_masks.py <reference-mask-dir> <target-dir>

The reference reads its 224x224 masks with cv2.imread() and resizes them straight to the feature grid
with the default bilinear interpolation, whereas pyCLAD's provider samples the full-resolution map
with nearest-neighbour: the two agree on only 82% of patch pairs, so the contrastive loss sees
different supervision. Writing the reference's grid out at 14x14 removes the difference, because the
provider's resize from 14x14 to 14x14 is an identity.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

GRID = (14, 14)


def convert(source_dir: Path, target_dir: Path) -> None:
    sources = sorted(source_dir.rglob("*.JPG"))
    if not sources:
        raise FileNotFoundError(f"No reference masks found under {source_dir}")

    written = 0
    for source in sources:
        target = (target_dir / source.relative_to(source_dir)).with_suffix(".png")
        if target.exists():
            continue

        mask = cv2.imread(str(source))
        if mask is None:
            raise OSError(f"Could not read {source}")

        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), cv2.resize(mask, GRID)[:, :, 0].astype(np.uint8))
        written += 1

    print(f"masks: {len(sources)} source, {written} written")


if __name__ == "__main__":
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
