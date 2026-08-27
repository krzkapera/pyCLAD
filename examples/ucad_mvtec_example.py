import logging
import os
import pathlib

from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.metrics.continual.backward_transfer import BackwardTransfer
from pyclad.metrics.continual.forward_transfer import ForwardTransfer
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.scenarios.concept_incremental import ConceptIncrementalScenario
from pyclad.strategies.baselines.naive import NaiveStrategy
from pyclad.vision.callbacks.vision_pixel_concept_metric_callback import (
    VisionPixelConceptMetricCallback,
)
from pyclad.vision.data.benchmarks.readers import read_vision_benchmark_dataset
from pyclad.vision.metrics.pixel_average_precision import PixelAveragePrecision
from pyclad.vision.metrics.pixel_roc_auc import PixelRocAuc
from pyclad.vision.models.ucad import UCADConfig, UCADModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

MVTEC_ROOT = os.environ["MVTEC_ROOT"]
MVTEC_MASKS_ROOT = os.environ.get("MVTEC_MASKS_ROOT")
OUTPUT_PATH = pathlib.Path(os.environ.get("UCAD_OUTPUT", "ucad_mvtec.json"))
INPUT_SIZE = (224, 224)


def main():
    dataset = read_vision_benchmark_dataset(
        root=MVTEC_ROOT,
        benchmark="mvtec",
        dataset_name="MVTec-AD",
        data_mode="paths",
        resize_to=INPUT_SIZE,
        resize_mode="short_side_crop",
    )

    config = UCADConfig(
        max_tasks=len(dataset.train_concepts()),
        input_size=INPUT_SIZE,
        sam_masks_dir=MVTEC_MASKS_ROOT,
        sam_images_root=MVTEC_ROOT if MVTEC_MASKS_ROOT else None,
    )
    model = UCADModel(config)
    strategy = NaiveStrategy(model)

    callbacks = [
        ConceptMetricCallback(
            base_metric=RocAuc(),
            summarized_metrics=[ContinualAverage(), BackwardTransfer(), ForwardTransfer()],
        ),
        VisionPixelConceptMetricCallback(
            base_metric=PixelAveragePrecision(),
            summarized_metrics=[ContinualAverage(), BackwardTransfer()],
        ),
        VisionPixelConceptMetricCallback(
            base_metric=PixelRocAuc(),
            summarized_metrics=[ContinualAverage()],
        ),
    ]

    ConceptIncrementalScenario(dataset, strategy=strategy, callbacks=callbacks).run()

    JsonOutputWriter(OUTPUT_PATH).write([model, dataset, strategy, *callbacks])


if __name__ == "__main__":
    main()
