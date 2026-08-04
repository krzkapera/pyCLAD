"""Characterizes what the structure-based contrastive loss actually sees, per mask source.

Usage: mask_supervision_stats.py <masks-root> [<masks-root> ...]

The loss compares every pair of the 196 patch embeddings and treats a pair as positive when both
patches carry the same SAM label, so what matters is not the mask image but the label map downsampled
to the 14x14 feature grid. This reports, per mask root: labels surviving the downsample, the fraction
of positive pairs, the share of the grid left unsegmented, and the share taken by the largest region.

With two or more roots holding masks for the same images, it also reports how often they agree on
"same region", which is the fraction of the loss's supervision they share.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

GRID = (14, 14)


def grid_labels(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise OSError(f"Could not read {path}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return cv2.resize(mask, GRID, interpolation=cv2.INTER_NEAREST).astype(np.int64).reshape(-1)


def describe(root: Path) -> dict:
    masks = sorted(root.rglob("*"))
    masks = [path for path in masks if path.is_file()]
    if not masks:
        raise FileNotFoundError(f"No masks under {root}")

    labels, positives, background, largest = [], [], [], []
    for path in masks:
        grid = grid_labels(path)
        labels.append(len(np.unique(grid)))
        positives.append(float((grid[:, None] == grid[None, :]).mean()))
        background.append(float((grid == 0).mean()))
        largest.append(float(np.bincount(grid - grid.min()).max() / grid.size))

    return {
        "root": root.name,
        "masks": len(masks),
        "labels": float(np.mean(labels)),
        "positive_pairs": float(np.mean(positives)),
        "background": float(np.mean(background)),
        "largest_region": float(np.mean(largest)),
    }


def agreement(first: Path, second: Path) -> tuple[int, float]:
    """Fraction of patch pairs both roots agree on, over the images they share."""
    shared, agreements = 0, []
    for path in sorted(first.rglob("*")):
        if not path.is_file():
            continue
        counterpart = next(second.glob(f"{path.relative_to(first).with_suffix('')}.*"), None)
        if counterpart is None:
            continue

        one, other = grid_labels(path), grid_labels(counterpart)
        same_one = one[:, None] == one[None, :]
        same_other = other[:, None] == other[None, :]
        agreements.append(float((same_one == same_other).mean()))
        shared += 1

    return shared, float(np.mean(agreements)) if agreements else float("nan")


if __name__ == "__main__":
    roots = [Path(argument) for argument in sys.argv[1:]]
    summaries = [describe(root) for root in roots]

    print(f"{'mask root':<24} {'masks':>7} {'labels':>7} {'pos pairs':>10} {'background':>11} {'largest':>8}")
    for summary in summaries:
        print(
            f"{summary['root']:<24} {summary['masks']:7d} {summary['labels']:7.2f} "
            f"{summary['positive_pairs']:10.4f} {summary['background']:11.4f} {summary['largest_region']:8.4f}"
        )

    for other in roots[1:]:
        shared, score = agreement(roots[0], other)
        print(f"\nsupervision shared by {roots[0].name} and {other.name}: {score:.4f} over {shared} images")
