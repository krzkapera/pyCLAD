"""Wiring check for the VisA run: two categories, few samples, one epoch.

Exercises the same reader, model, scenario and metrics as examples/ucad_visa_example.py,
so a green run means paths, SAM masks and pixel ground truth line up.
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

VISA_ROOT = os.environ["VISA_ROOT"]
VISA_MASKS_ROOT = os.environ.get("VISA_MASKS_ROOT")
OUTPUT_PATH = pathlib.Path(os.environ.get("UCAD_OUTPUT", "ucad_visa_smoke.json"))
INPUT_SIZE = (224, 224)
CATEGORIES = ["candle", "capsules"]


def main():
    dataset = read_vision_benchmark_dataset(
        root=VISA_ROOT,
        benchmark="visa_folder",
        dataset_name="VisA-smoke",
        categories=CATEGORIES,
        data_mode="paths",
        resize_to=INPUT_SIZE,
        max_train_samples_per_category=16,
    )

    config = UCADConfig(
        max_tasks=len(CATEGORIES),
        input_size=INPUT_SIZE,
        training_epochs=1,
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
