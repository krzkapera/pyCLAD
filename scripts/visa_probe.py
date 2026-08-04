"""Probes how VisA results depend on the amount of prompt tuning.

The paper's numbers come from a run that evaluates the test set after every epoch and
reports the best one, on batches of 24 (scripts/../ucad-ref-run/args_dict.npy); the
framework evaluates once, after the last epoch, on batches of 8. This sweeps epoch count
and batch size over two categories to separate "too much prompt tuning" from "wrong batch".
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

VISA_ROOT = os.environ["VISA_ROOT"]
VISA_MASKS_ROOT = os.environ.get("VISA_MASKS_ROOT")
EPOCHS = int(os.environ["UCAD_EPOCHS"])
BATCH_SIZE = int(os.environ["UCAD_BATCH_SIZE"])
OUTPUT_PATH = pathlib.Path(os.environ.get("UCAD_OUTPUT", "ucad_visa_probe.json"))
INPUT_SIZE = (224, 224)
# An empty UCAD_CATEGORIES probes the whole benchmark.
CATEGORIES = [category for category in os.environ.get("UCAD_CATEGORIES", "candle,capsules").split(",") if category]


def main():
    logger.info("PROBE epochs=%d batch_size=%d categories=%s", EPOCHS, BATCH_SIZE, CATEGORIES or "all")

    dataset = read_vision_benchmark_dataset(
        root=VISA_ROOT,
        benchmark="visa_folder",
        dataset_name="VisA-probe",
        categories=CATEGORIES or None,
        data_mode="paths",
        resize_to=INPUT_SIZE,
    )

    config = UCADConfig(
        max_tasks=len(dataset.train_concepts()),
        input_size=INPUT_SIZE,
        training_epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        sam_masks_dir=VISA_MASKS_ROOT,
        sam_images_root=VISA_ROOT if VISA_MASKS_ROOT else None,
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
