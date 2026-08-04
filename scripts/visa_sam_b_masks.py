"""Rewrites the VisA SAM2 label maps into the format the reference UCAD implementation reads.

Usage: visa_sam_b_masks.py <source-masks-dir> <target-masks-dir> [target-suffix]

The reference derives a mask path by replacing 'visa' with 'visa-sam-b' in the image path, so the
files keep the image's .JPG name while holding PNG bytes (cv2 dispatches on content, and a JPEG
encoder would quantize the label ids). It then reads them with cv2.imread() without flags, which
converts 16-bit input by dropping the low byte - every label of a 16-bit map collapses to 0 - so the
labels are renumbered into uint8 at the 224x224 size of the authors' released MVTec masks.

pyCLAD's own provider instead resolves masks by suffix, so pass '.png' to build a copy it can read.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

MASK_SIZE = (224, 224)
MAX_LABELS = 256


def to_reference_format(label_map: np.ndarray) -> np.ndarray:
    resized = cv2.resize(label_map, MASK_SIZE, interpolation=cv2.INTER_NEAREST)
    labels = np.unique(resized)
    if len(labels) > MAX_LABELS:
        raise ValueError(f"{len(labels)} labels do not fit in uint8")

    renumbered = np.zeros(resized.shape, dtype=np.uint8)
    for new_label, old_label in enumerate(labels):
        renumbered[resized == old_label] = new_label
    return renumbered


def convert(source_dir: Path, target_dir: Path, suffix: str = ".JPG") -> None:
    sources = sorted(source_dir.rglob("*.png"))
    if not sources:
        raise FileNotFoundError(f"No masks found under {source_dir}")

    written = 0
    for source in sources:
        target = (target_dir / source.relative_to(source_dir)).with_suffix(suffix)
        if target.exists():
            continue

        label_map = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if label_map is None:
            raise OSError(f"Could not read {source}")

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(cv2.imencode(".png", to_reference_format(label_map))[1].tobytes())
        written += 1

    print(f"masks: {len(sources)} source, {written} written, {len(sources) - written} already present")


if __name__ == "__main__":
    convert(Path(sys.argv[1]), Path(sys.argv[2]), *sys.argv[3:])
