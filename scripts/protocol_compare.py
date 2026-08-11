"""Scores one configuration under both evaluation protocols.

Usage: protocol_compare.py      (configured through the environment, see ucad_probe.py)

The reference evaluates the test set after every epoch, averages the min-max normalized scores of
every epoch so far, and keeps the epoch whose image AUROC on that cumulative ensemble is highest. Two
numbers therefore live inside one training run: the cumulative ensemble after the last epoch, which
uses no test labels, and its maximum over epochs, which does. This reports both per concept.

Each epoch's state scores the test set once; the cumulative averages are then formed on the stored
maps, so the cost is one evaluation per epoch rather than one per (epoch, prefix) pair.
"""

import logging
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from visa_layout import VISA_FOLDER_LAYOUT
from pyclad.vision.data.benchmarks.readers import read_vision_benchmark_dataset
from pyclad.vision.models.ucad import UCADConfig
from reference_ensemble import ReferenceEnsembleUCAD

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

BENCHMARKS = {
    "visa": ("VISA_ROOT", "VISA_MASKS_ROOT", VISA_FOLDER_LAYOUT),
    "mvtec": ("MVTEC_ROOT", "MVTEC_MASKS_ROOT", "mvtec"),
}
DATASET = os.environ.get("UCAD_DATASET", "visa")
ROOT_VAR, MASKS_VAR, BENCHMARK = BENCHMARKS[DATASET]
ROOT = os.environ[ROOT_VAR]
MASKS_DIR = os.environ.get("UCAD_MASKS_DIR") or os.environ[MASKS_VAR]
EPOCHS = int(os.environ.get("UCAD_EPOCHS", "25"))
BATCH_SIZE = int(os.environ.get("UCAD_BATCH_SIZE", "24"))
CORESET_MODE = os.environ.get("UCAD_CORESET", "approximate")
RESIZE_MODE = os.environ.get("UCAD_RESIZE_MODE", "short_side_crop")
BLUR_SIGMA = float(os.environ.get("UCAD_BLUR", "4.0"))
SEED = int(os.environ.get("UCAD_SEED", "0"))
CATEGORIES = [c for c in os.environ.get("UCAD_CATEGORIES", "").split(";") if c]
INPUT_SIZE = (224, 224)


def normalized(values: np.ndarray) -> np.ndarray:
    """Min-max over the whole test set, as the reference normalizes each epoch's output."""
    flat = values.reshape(len(values), -1)
    low = flat.min(axis=1).reshape(-1, *([1] * (values.ndim - 1)))
    high = flat.max(axis=1).reshape(-1, *([1] * (values.ndim - 1)))
    return (values - low) / np.maximum(high - low, 1e-12)


def member_outputs(model: ReferenceEnsembleUCAD, concept) -> tuple[np.ndarray, np.ndarray]:
    outputs = model.member_predictions(concept.data)
    return normalized(np.stack([s for s, _ in outputs])), normalized(np.stack([m for _, m in outputs]))


def main():
    logger.info(
        "PROTOCOL_CONFIG dataset=%s epochs=%d batch_size=%d coreset=%s resize=%s blur=%.1f seed=%d "
        "masks=%s categories=%s",
        DATASET, EPOCHS, BATCH_SIZE, CORESET_MODE, RESIZE_MODE, BLUR_SIGMA, SEED, MASKS_DIR,
        CATEGORIES or "all",
    )

    dataset = read_vision_benchmark_dataset(
        root=ROOT,
        benchmark=BENCHMARK,
        categories=CATEGORIES or None,
        data_mode="paths",
        resize_to=INPUT_SIZE,
        resize_mode=RESIZE_MODE,
    )

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

        scores, maps = member_outputs(model, test_concept)
        truth = test_concept.masks.reshape(-1)

        images, pixels = [], []
        for epochs in range(1, EPOCHS + 1):
            images.append(float(roc_auc_score(test_concept.labels, scores[:epochs].mean(axis=0))))
            pixels.append(float(average_precision_score(truth, maps[:epochs].mean(axis=0).reshape(-1))))

        singles = [float(roc_auc_score(test_concept.labels, scores[epoch])) for epoch in range(EPOCHS)]
        single_pixels = [
            float(average_precision_score(truth, maps[epoch].reshape(-1))) for epoch in range(EPOCHS)
        ]
        logger.info("TRAJECTORY %s single=%s", test_concept.name, np.round(singles, 4).tolist())
        logger.info("TRAJECTORY %s single_pixel=%s", test_concept.name, np.round(single_pixels, 4).tolist())
        logger.info("TRAJECTORY %s cumulative=%s", test_concept.name, np.round(images, 4).tolist())

        best = int(np.argmax(images))
        logger.info(
            "PROTOCOL %s honest_image=%.4f honest_pixel=%.4f | selected_epoch=%d selected_image=%.4f "
            "selected_pixel=%.4f | selection_gain=%.4f",
            test_concept.name, images[-1], pixels[-1], best + 1, images[best], pixels[best],
            images[best] - images[-1],
        )


if __name__ == "__main__":
    main()
