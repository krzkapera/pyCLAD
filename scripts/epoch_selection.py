"""Asks whether an epoch can be chosen without reading the labels it is scored on.

Usage: epoch_selection.py        (configured through the environment, see ucad_probe.py)

    UCAD_VAL_FRACTION   share of the test concept held out for selection   (default 0.5)

The reference picks the epoch whose image AUROC on the test set is highest, which is worth about
+0.044 to it and is a leak. Two honest replacements are measured here against that upper bound and
against taking the last epoch:

labelled validation   the test concept is split in two, stratified by label; the epoch is chosen on
                      one half and reported on the other. Legitimate, but it needs labelled
                      anomalies, which an unsupervised method is not supposed to have.
label-free criteria   a slice of the training normals is held out of the bank and scored by the same
                      model, and the stored bank is inspected directly. Nothing here sees an anomaly
                      label, so any of these could run in a real deployment:

                      calibration_mean/_max/_p95  where the held-out normals land - a model that
                                                  scores its own normals low is well calibrated
                      calibration_fpr             the held-out normals are split in two, the
                                                  threshold is the 95th percentile of one half and
                                                  the false positive rate is read off the other
                      bank_cosine, bank_rank      geometry of the 196 stored vectors, mean pairwise
                                                  cosine and effective rank, both of which 25 epochs
                                                  of the authors' loss drive down
                      test_bimodality             a two-means split of the unlabelled test scores,
                                                  between-cluster over total variance. This one
                                                  touches the test images, though never their
                                                  labels, so it is transductive and stands apart.

Each criterion picks one epoch per concept; all of them are then scored on the same held-out half, so
the columns are comparable.
"""

import json
import logging
import os
import pathlib

import numpy as np
from sklearn.metrics import roc_auc_score

from visa_layout import VISA_FOLDER_LAYOUT
from pyclad.vision.data.benchmarks.readers import read_vision_benchmark_dataset
from pyclad.vision.models.ucad import UCADConfig, UCADModel
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
MASKS_DIR = os.environ.get("UCAD_MASKS_DIR") or os.environ.get(MASKS_VAR)
EPOCHS = int(os.environ.get("UCAD_EPOCHS", "25"))
BATCH_SIZE = int(os.environ.get("UCAD_BATCH_SIZE", "8"))
VAL_FRACTION = float(os.environ.get("UCAD_VAL_FRACTION", "0.5"))
CALIBRATION_FRACTION = float(os.environ.get("UCAD_CALIBRATION_FRACTION", "0.1"))
RESIZE_MODE = os.environ.get("UCAD_RESIZE_MODE", "short_side_crop")
BLUR_SIGMA = float(os.environ.get("UCAD_BLUR", "4.0"))
SEED = int(os.environ.get("UCAD_SEED", "0"))
OUTPUT_PATH = pathlib.Path(os.environ.get("UCAD_OUTPUT", "epoch_selection.json"))
INPUT_SIZE = (224, 224)


def effective_rank(vectors: np.ndarray) -> float:
    spectrum = np.linalg.svd(vectors - vectors.mean(axis=0), compute_uv=False)
    spectrum = spectrum / max(spectrum.sum(), 1e-12)
    spectrum = spectrum[spectrum > 0]
    return float(np.exp(-(spectrum * np.log(spectrum)).sum()))


def mean_pairwise_cosine(vectors: np.ndarray) -> float:
    normalised = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    similarity = normalised @ normalised.T
    return float(similarity[~np.eye(len(vectors), dtype=bool)].mean())


def bimodality(scores: np.ndarray) -> float:
    """Between-cluster over total variance for the best two-means split of one-dimensional scores."""
    ordered = np.sort(scores)
    total = float(((ordered - ordered.mean()) ** 2).sum())
    if total <= 0:
        return 0.0

    best = 0.0
    for cut in range(1, len(ordered)):
        low, high = ordered[:cut], ordered[cut:]
        between = len(low) * (low.mean() - ordered.mean()) ** 2 + len(high) * (high.mean() - ordered.mean()) ** 2
        best = max(best, float(between))
    return best / total


def stratified_split(labels: np.ndarray, fraction: float, rng: np.random.Generator):
    validation = np.zeros(len(labels), dtype=bool)
    for value in np.unique(labels):
        indices = np.flatnonzero(labels == value)
        rng.shuffle(indices)
        validation[indices[: int(round(fraction * len(indices)))]] = True
    return validation, ~validation


def main():
    logger.info(
        "SELECTION dataset=%s epochs=%d val_fraction=%.2f calibration=%.2f seed=%d",
        DATASET, EPOCHS, VAL_FRACTION, CALIBRATION_FRACTION, SEED,
    )

    dataset = read_vision_benchmark_dataset(
        root=ROOT, benchmark=BENCHMARK, data_mode="paths", resize_to=INPUT_SIZE, resize_mode=RESIZE_MODE
    )
    rng = np.random.default_rng(SEED)
    summary: dict[str, list[float]] = {}

    for train_concept, test_concept in zip(dataset.train_concepts(), dataset.test_concepts()):
        train_paths = np.asarray(train_concept.data)
        held_out = rng.permutation(len(train_paths))[: max(1, int(CALIBRATION_FRACTION * len(train_paths)))]
        calibration = train_paths[held_out]
        fitting = np.delete(train_paths, held_out)

        config = UCADConfig(
            max_tasks=1, input_size=INPUT_SIZE, resize_mode=RESIZE_MODE, training_epochs=EPOCHS,
            batch_size=BATCH_SIZE, blur_sigma=BLUR_SIGMA, seed=SEED,
            sam_masks_dir=MASKS_DIR, sam_images_root=ROOT if MASKS_DIR else None,
        )
        model = ReferenceEnsembleUCAD(config, members=EPOCHS)
        model.fit(fitting)

        # The same concept with the prefix left untrained, so "do not train at all" is a candidate on
        # exactly the data every criterion is scored on.
        untrained = UCADModel(config.model_copy(update={"training_epochs": 0}))
        untrained.fit(fitting)

        labels = np.asarray(test_concept.labels)
        val_mask, test_mask = stratified_split(labels, VAL_FRACTION, rng)
        test_paths = np.asarray(test_concept.data)

        per_epoch_test = [s for s, _ in model.member_predictions(test_paths)]
        per_epoch_calibration = [s for s, _ in model.member_predictions(calibration)]

        val_auroc = [float(roc_auc_score(labels[val_mask], s[val_mask])) for s in per_epoch_test]
        held_auroc = [float(roc_auc_score(labels[test_mask], s[test_mask])) for s in per_epoch_test]
        full_auroc = [float(roc_auc_score(labels, s)) for s in per_epoch_test]

        half = max(1, len(calibration) // 2)
        banks = [bank.detach().cpu().numpy() for _, bank in model._task_members[-1]]

        picks = {
            "last_epoch": len(held_auroc) - 1,
            "oracle_full_test": int(np.argmax(full_auroc)),
            "labelled_validation": int(np.argmax(val_auroc)),
            "calibration_mean": int(np.argmin([float(np.mean(s)) for s in per_epoch_calibration])),
            "calibration_max": int(np.argmin([float(np.max(s)) for s in per_epoch_calibration])),
            "calibration_p95": int(np.argmin([float(np.percentile(s, 95)) for s in per_epoch_calibration])),
            "calibration_fpr": int(np.argmin([
                float(np.mean(s[half:] > np.percentile(s[:half], 95))) for s in per_epoch_calibration
            ])),
            "bank_cosine": int(np.argmax([mean_pairwise_cosine(b) for b in banks])),
            "bank_rank": int(np.argmax([effective_rank(b) for b in banks])),
            "test_bimodality": int(np.argmax([bimodality(s) for s in per_epoch_test])),
        }

        for criterion, epoch in picks.items():
            summary.setdefault(criterion, []).append(held_auroc[epoch])
        summary.setdefault("oracle_held_out", []).append(max(held_auroc))
        untrained_scores = untrained.predict(test_paths).anomaly_scores
        summary.setdefault("no_training", []).append(
            float(roc_auc_score(labels[test_mask], untrained_scores[test_mask]))
        )

        logger.info(
            "SELECTION %s picks=%s scores=%s", test_concept.name,
            {k: v + 1 for k, v in picks.items()},
            {k: round(summary[k][-1], 4) for k in picks},
        )

    logger.info("SELECTION_AVERAGE dataset=%s seed=%d %s", DATASET, SEED,
                {k: round(float(np.mean(v)), 4) for k, v in summary.items()})
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
