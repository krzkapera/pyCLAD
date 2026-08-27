"""Generates VisA structure masks with SAM ViT-B the way the UCAD authors did.

Usage: visa_sam_vit_b_masks.py <images-root> <masks-root> <checkpoint> [<category> ...]

Follows segment_anything/dataset_sam.py from the reference repository: the image is resized to
224x224, SamAutomaticMaskGenerator runs with its default thresholds, and regions are numbered from 1
in order of decreasing area, leaving unsegmented pixels at 0. Masks land next to their image's
relative path with a .png suffix, which is where pyCLAD's offline provider looks for them.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

MASK_SIZE = (224, 224)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def label_map(masks: list[dict], shape: tuple[int, int]) -> np.ndarray:
    labels = np.zeros(shape, dtype=np.uint8)
    for label, mask in enumerate(sorted(masks, key=lambda mask: mask["area"], reverse=True), start=1):
        labels[mask["segmentation"]] = label
    return labels


def generate(images_root: Path, masks_root: Path, checkpoint: Path, categories: list[str]) -> None:
    sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
    sam.to(device="cuda")
    mask_generator = SamAutomaticMaskGenerator(sam)

    for category in categories:
        sources = sorted(
            path
            for path in (images_root / category / "train" / "good").iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not sources:
            raise FileNotFoundError(f"No training images for {category} under {images_root}")

        written = 0
        for source in sources:
            target = (masks_root / source.relative_to(images_root)).with_suffix(".png")
            if target.exists():
                continue

            image = cv2.imread(str(source))
            if image is None:
                raise OSError(f"Could not read {source}")

            resized = cv2.cvtColor(cv2.resize(image, MASK_SIZE), cv2.COLOR_BGR2RGB)
            with torch.inference_mode():
                masks = mask_generator.generate(resized)

            target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target), label_map(masks, MASK_SIZE))
            written += 1

        print(f"{category}: {len(sources)} images, {written} masks written", flush=True)


if __name__ == "__main__":
    generate(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:])
