"""Compares how pyCLAD and the reference turn a ground-truth mask into pixel labels.

Usage: mask_convention_effect.py   (configured through the environment)

    UCAD_DATASET    visa | mvtec              (default visa)
    UCAD_CATEGORIES ';'-separated categories  (default candle;capsules;cashew)

pyCLAD resizes a mask with nearest-neighbour interpolation and keeps every non-zero pixel, so the
positive set includes the boundary. The reference resizes bilinearly and then casts the tensor to
int32, which truncates every fractional pixel to zero and keeps only the mask's interior. Pixel AUPR
depends on the size of that positive set, so this reports both conventions side by side: the positive
rate each produces and, on identical score maps, the AUPR each yields. Both use the geometry the run
itself used - comparing a stretched mask against cropped maps measures misalignment, not thresholding.
"""

import logging
import os

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score

from visa_layout import VISA_FOLDER_LAYOUT
from pyclad.vision.data.benchmarks.readers import index_vision_benchmark
from pyclad.vision.data.geometry import resize_image
from pyclad.vision.models.ucad import UCADConfig, UCADModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

BENCHMARKS = {
    "visa": ("VISA_ROOT", "VISA_MASKS_ROOT", VISA_FOLDER_LAYOUT),
    "mvtec": ("MVTEC_ROOT", "MVTEC_MASKS_ROOT", "mvtec"),
}
DATASET = os.environ.get("UCAD_DATASET", "visa")
ROOT_VAR, MASKS_VAR, BENCHMARK = BENCHMARKS[DATASET]
CATEGORIES = [c for c in os.environ.get("UCAD_CATEGORIES", "candle;capsules;cashew").split(";") if c]
INPUT_SIZE = (224, 224)
# Both conventions must share the run's geometry, or the comparison measures misalignment instead.
RESIZE_MODE = os.environ.get("UCAD_RESIZE_MODE", "short_side_crop")


def pyclad_labels(mask_path) -> np.ndarray:
    """Nearest-neighbour resize under the run's geometry, every non-zero pixel positive."""
    with Image.open(mask_path) as mask:
        resized = resize_image(mask.convert("L"), INPUT_SIZE, RESIZE_MODE, Image.Resampling.NEAREST)
    return (np.asarray(resized) > 0).astype(np.uint8)


def reference_labels(mask_path) -> np.ndarray:
    """Bilinear resize of the short side, centre crop, then the int cast that drops fractions."""
    with Image.open(mask_path) as mask:
        grey = mask.convert("L")
        scale = INPUT_SIZE[0] / min(grey.size)
        resized = grey.resize((round(grey.width * scale), round(grey.height * scale)), Image.Resampling.BILINEAR)

    left = (resized.width - INPUT_SIZE[1]) // 2
    top = (resized.height - INPUT_SIZE[0]) // 2
    cropped = resized.crop((left, top, left + INPUT_SIZE[1], top + INPUT_SIZE[0]))
    return (np.asarray(cropped).astype(np.float32) / 255.0).astype(np.int32).astype(np.uint8)


def main():
    config = UCADConfig(
        max_tasks=1,
        input_size=INPUT_SIZE,
        resize_mode=RESIZE_MODE,
        training_epochs=1,
        sam_masks_dir=os.environ.get(MASKS_VAR),
        sam_images_root=os.environ[ROOT_VAR],
    )

    for category in CATEGORIES:
        samples = index_vision_benchmark(root=os.environ[ROOT_VAR], benchmark=BENCHMARK, categories=[category])
        train = [s for s in samples if s.split == "train"]
        test = [s for s in samples if s.split == "test"]

        model = UCADModel(config)
        model.fit(np.asarray([str(s.image_path) for s in train], dtype=object))
        results = model.predict(np.asarray([str(s.image_path) for s in test], dtype=object))

        ours = np.stack([pyclad_labels(s.mask_path) if s.mask_path else np.zeros(INPUT_SIZE, np.uint8) for s in test])
        theirs = np.stack(
            [reference_labels(s.mask_path) if s.mask_path else np.zeros(INPUT_SIZE, np.uint8) for s in test]
        )
        maps = results.score_maps.reshape(-1)

        logger.info(
            "CONVENTION %s positives pyclad=%.5f reference=%.5f ratio=%.2f | aupr pyclad=%.4f reference=%.4f",
            category, float(ours.mean()), float(theirs.mean()),
            float(ours.sum() / max(theirs.sum(), 1)),
            average_precision_score(ours.reshape(-1), maps),
            average_precision_score(theirs.reshape(-1), maps),
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
