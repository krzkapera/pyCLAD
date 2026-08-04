"""Runs one UCAD configuration over a benchmark and reports every concept-level metric.

Everything the probe varies comes from the environment, so a batch of configurations is a batch of
sbatch submissions differing only in their --export list:

    UCAD_DATASET      visa | mvtec                     (default visa)
    UCAD_EPOCHS       prompt-tuning epochs per concept (required)
    UCAD_BATCH_SIZE   training batch size              (required)
    UCAD_CATEGORIES   comma-separated, empty = all     (default all)
    UCAD_MASKS_DIR    SAM masks root                   (default the dataset's masks env var)
    UCAD_REWEIGHTING  reweighting_num_nn, 0 = max      (default 0)
    UCAD_SEED         prompt and data-loader seed      (default 0)
    UCAD_OUTPUT       JSON destination                 (default ./ucad_probe.json)
"""

import logging
import os
import pathlib

from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.scenarios.concept_incremental import ConceptIncrementalScenario
from pyclad.strategies.baselines.naive import NaiveStrategy
from pyclad.vision.callbacks.vision_pixel_concept_metric_callback import (
    VisionPixelConceptMetricCallback,
)
from pyclad.vision.data.benchmarks.readers import read_vision_benchmark_dataset
from pyclad.vision.metrics.pixel_average_precision import PixelAveragePrecision
from pyclad.vision.models.ucad import UCADConfig, UCADModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

BENCHMARKS = {
    "visa": ("VISA_ROOT", "VISA_MASKS_ROOT", "visa_folder"),
    "mvtec": ("MVTEC_ROOT", "MVTEC_MASKS_ROOT", "mvtec"),
}

DATASET = os.environ.get("UCAD_DATASET", "visa")
ROOT_VAR, MASKS_VAR, BENCHMARK = BENCHMARKS[DATASET]
ROOT = os.environ[ROOT_VAR]
MASKS_DIR = os.environ.get("UCAD_MASKS_DIR") or os.environ.get(MASKS_VAR)
EPOCHS = int(os.environ["UCAD_EPOCHS"])
BATCH_SIZE = int(os.environ["UCAD_BATCH_SIZE"])
REWEIGHTING = int(os.environ.get("UCAD_REWEIGHTING", "0"))
SEED = int(os.environ.get("UCAD_SEED", "0"))
CATEGORIES = [category for category in os.environ.get("UCAD_CATEGORIES", "").split(",") if category]
OUTPUT_PATH = pathlib.Path(os.environ.get("UCAD_OUTPUT", "ucad_probe.json"))
INPUT_SIZE = (224, 224)


def main():
    logger.info(
        "PROBE dataset=%s epochs=%d batch_size=%d reweighting=%d seed=%d masks=%s categories=%s",
        DATASET, EPOCHS, BATCH_SIZE, REWEIGHTING, SEED, MASKS_DIR, CATEGORIES or "all",
    )

    dataset = read_vision_benchmark_dataset(
        root=ROOT,
        benchmark=BENCHMARK,
        dataset_name=f"{DATASET}-probe",
        categories=CATEGORIES or None,
        data_mode="paths",
        resize_to=INPUT_SIZE,
    )

    config = UCADConfig(
        max_tasks=len(dataset.train_concepts()),
        input_size=INPUT_SIZE,
        training_epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        reweighting_num_nn=REWEIGHTING,
        seed=SEED,
        sam_masks_dir=MASKS_DIR,
        sam_images_root=ROOT if MASKS_DIR else None,
    )
    model = UCADModel(config)
    strategy = NaiveStrategy(model)

    callbacks = [
        ConceptMetricCallback(base_metric=RocAuc(), summarized_metrics=[ContinualAverage()]),
        VisionPixelConceptMetricCallback(
            base_metric=PixelAveragePrecision(), summarized_metrics=[ContinualAverage()]
        ),
    ]

    ConceptIncrementalScenario(dataset, strategy=strategy, callbacks=callbacks).run()

    JsonOutputWriter(OUTPUT_PATH).write([model, dataset, strategy, *callbacks])


if __name__ == "__main__":
    main()
