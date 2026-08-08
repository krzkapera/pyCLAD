"""Scores one run's anomaly maps against both ground-truth mask conventions.

Usage: pixel_convention_effect.py     (configured through the environment, see ucad_probe.py)

Two conventions separate pyCLAD's pixel AUPR from the reference's, and neither is the model:

- the ground truth. pyCLAD resamples the mask with nearest-neighbour and counts every nonzero pixel;
  the reference resamples it bilinearly and truncates to int, keeping only pixels that survive the
  interpolation at full weight.
- the anomaly map. pyCLAD scores patches by Euclidean distance; the reference reads squared
  distances out of faiss and never takes their root, so its map is squared before it is upsampled
  and smoothed.

This trains once per concept and reports the full 2x2, which makes every comparison paired.
"""

import logging
import os

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score

from pyclad.vision.data.benchmarks.readers import index_vision_benchmark, read_vision_benchmark_dataset
from pyclad.vision.data.geometry import resize_image
from pyclad.vision.models.ucad import UCADConfig
from pyclad.vision.models.ucad.reference_ensemble import ReferenceEnsembleUCAD

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

BENCHMARKS = {
    "visa": ("VISA_ROOT", "VISA_MASKS_ROOT", "visa_folder"),
    "mvtec": ("MVTEC_ROOT", "MVTEC_MASKS_ROOT", "mvtec"),
}
DATASET = os.environ.get("UCAD_DATASET", "visa")
ROOT_VAR, MASKS_VAR, BENCHMARK = BENCHMARKS[DATASET]
ROOT = os.environ[ROOT_VAR]
MASKS_DIR = os.environ.get("UCAD_MASKS_DIR") or os.environ[MASKS_VAR]
EPOCHS = int(os.environ.get("UCAD_EPOCHS", "25"))
BATCH_SIZE = int(os.environ.get("UCAD_BATCH_SIZE", "8"))
CORESET_MODE = os.environ.get("UCAD_CORESET", "approximate")
RESIZE_MODE = os.environ.get("UCAD_RESIZE_MODE", "short_side_crop")
BLUR_SIGMA = float(os.environ.get("UCAD_BLUR", "4.0"))
SEED = int(os.environ.get("UCAD_SEED", "0"))
CATEGORIES = [c for c in os.environ.get("UCAD_CATEGORIES", "").split(";") if c]
INPUT_SIZE = (224, 224)


def reference_mask(sample) -> np.ndarray:
    """The reference's ground truth: bilinear resize, ToTensor, then truncation to int.

    `astype(np.int32)` on a 0..1 float keeps only the pixels the interpolation left at full weight,
    so a bilinear edge pixel counts as background where pyCLAD's nearest-neighbour path keeps it.
    """
    if sample.mask_path is None:
        return np.zeros(INPUT_SIZE, dtype=np.uint8)

    with Image.open(sample.mask_path) as mask:
        resized = resize_image(mask.convert("L"), INPUT_SIZE, RESIZE_MODE, Image.Resampling.BILINEAR)
    return (np.asarray(resized, dtype=np.float32) / 255.0).astype(np.int32).astype(np.uint8)


def reference_masks_for(category: str, samples) -> np.ndarray:
    """Same samples, in the same order and with the same drops, as the dataset reader keeps."""
    kept = [
        sample
        for sample in samples
        if sample.category == category
        and sample.split == "test"
        and not (sample.image_label == 1 and sample.mask_path is None)
    ]
    return np.stack([reference_mask(sample) for sample in kept], axis=0)


def main():
    logger.info(
        "CONVENTION_CONFIG dataset=%s epochs=%d batch_size=%d resize=%s blur=%.1f seed=%d masks=%s",
        DATASET, EPOCHS, BATCH_SIZE, RESIZE_MODE, BLUR_SIGMA, SEED, MASKS_DIR,
    )

    dataset = read_vision_benchmark_dataset(
        root=ROOT,
        benchmark=BENCHMARK,
        categories=CATEGORIES or None,
        data_mode="paths",
        resize_to=INPUT_SIZE,
        resize_mode=RESIZE_MODE,
    )
    samples = index_vision_benchmark(root=ROOT, benchmark=BENCHMARK, categories=CATEGORIES or None)

    for train_concept, test_concept in zip(dataset.train_concepts(), dataset.test_concepts()):
        config = UCADConfig(
            max_tasks=1,
            input_size=INPUT_SIZE,
            resize_mode=RESIZE_MODE,
            training_epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            coreset_mode=CORESET_MODE,
            blur_sigma=BLUR_SIGMA,
            seed=SEED,
            sam_masks_dir=MASKS_DIR,
            sam_images_root=ROOT,
        )
        model = ReferenceEnsembleUCAD(config, members=EPOCHS)
        model.fit(train_concept.data)

        truths = {"ours": test_concept.masks, "reference": reference_masks_for(test_concept.name, samples)}
        if truths["reference"].shape != truths["ours"].shape:
            raise ValueError(
                f"{test_concept.name}: {truths['reference'].shape} reference masks against "
                f"{truths['ours'].shape} ours"
            )

        scores = {}
        for map_convention in ("ours", "reference"):
            model.scorer.squared_distances = map_convention == "reference"
            maps = model.predict(test_concept.data).score_maps
            for truth_convention, truth in truths.items():
                scores[f"map_{map_convention}_truth_{truth_convention}"] = average_precision_score(
                    truth.reshape(-1), maps.reshape(-1)
                )

        logger.info(
            "CONVENTION %s %s positive_pixels_ours=%d positive_pixels_reference=%d",
            test_concept.name,
            " ".join(f"{name}={value:.4f}" for name, value in scores.items()),
            int(truths["ours"].sum()),
            int(truths["reference"].sum()),
        )


if __name__ == "__main__":
    main()
