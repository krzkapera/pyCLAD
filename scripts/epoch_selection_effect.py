"""Quantifies what picking the best training epoch on the test set is worth.

Usage: epoch_selection_effect.py    (configured through the environment)

    UCAD_DATASET    visa | mvtec                 (default visa)
    UCAD_CATEGORIES ';'-separated categories     (default candle;capsules;cashew)
    UCAD_EPOCHS     epochs, each one a candidate (default 10)

The reference keeps, per concept, the epoch whose test-set image AUROC is highest, so its reported
number is a maximum over epochs rather than the value of a fixed one. This trains a concept, scores
the test set separately with every epoch's prompt and knowledge, and reports each epoch's metrics
next to their mean and maximum: the gap between the last epoch and the maximum is what the selection
buys, and it cannot be had without test labels.
"""

import logging
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from pyclad.vision.data.benchmarks.readers import read_vision_benchmark_dataset
from pyclad.vision.models.ucad import UCADConfig, UCADModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

BENCHMARKS = {
    "visa": ("VISA_ROOT", "VISA_MASKS_ROOT", "visa_folder"),
    "mvtec": ("MVTEC_ROOT", "MVTEC_MASKS_ROOT", "mvtec"),
}
DATASET = os.environ.get("UCAD_DATASET", "visa")
ROOT_VAR, MASKS_VAR, BENCHMARK = BENCHMARKS[DATASET]
CATEGORIES = [c for c in os.environ.get("UCAD_CATEGORIES", "candle;capsules;cashew").split(";") if c]
EPOCHS = int(os.environ.get("UCAD_EPOCHS", "10"))
INPUT_SIZE = (224, 224)


def member_metrics(model: UCADModel, concept, member: int) -> tuple[float, float]:
    """Image AUROC and pixel AUPR when only one epoch's state scores the concept."""
    states = model.memory.get_states(model.memory.num_tasks - 1)
    model.memory.tasks[-1].states = [states[member]]
    try:
        results = model.predict(concept.data)
    finally:
        model.memory.tasks[-1].states = states

    return (
        float(roc_auc_score(concept.labels, results.anomaly_scores)),
        float(average_precision_score(concept.masks.reshape(-1), results.score_maps.reshape(-1))),
    )


def main():
    dataset = read_vision_benchmark_dataset(
        root=os.environ[ROOT_VAR],
        benchmark=BENCHMARK,
        categories=CATEGORIES,
        data_mode="paths",
        resize_to=INPUT_SIZE,
    )

    for train_concept, test_concept in zip(dataset.train_concepts(), dataset.test_concepts()):
        config = UCADConfig(
            max_tasks=1,
            input_size=INPUT_SIZE,
            training_epochs=EPOCHS,
            score_ensemble_epochs=EPOCHS,
            sam_masks_dir=os.environ.get(MASKS_VAR),
            sam_images_root=os.environ[ROOT_VAR],
        )
        model = UCADModel(config)
        model.fit(train_concept.data)

        per_epoch = [member_metrics(model, test_concept, member) for member in range(EPOCHS)]
        images = [image for image, _ in per_epoch]
        pixels = [pixel for _, pixel in per_epoch]

        for epoch, (image, pixel) in enumerate(per_epoch, start=1):
            logger.info("SELECTION %s epoch=%d image_auroc=%.4f pixel_aupr=%.4f", test_concept.name, epoch, image, pixel)
        logger.info(
            "SELECTION %s last=%.4f mean=%.4f best=%.4f selection_gain=%.4f | pixel last=%.4f best=%.4f gain=%.4f",
            test_concept.name, images[-1], float(np.mean(images)), max(images), max(images) - images[-1],
            pixels[-1], max(pixels), max(pixels) - pixels[-1],
        )


if __name__ == "__main__":
    main()
