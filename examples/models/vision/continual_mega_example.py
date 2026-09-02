import logging
import pathlib

from pyclad.callbacks.evaluation.grouped_concept_metric_evaluation import (
    GroupedConceptMetricCallback,
)
from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_accuracy import AverageAccuracy
from pyclad.metrics.continual.forgetting_measure_strict import ForgettingMeasureStrict
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.scenarios.concept_incremental import ConceptIncrementalScenario
from pyclad.strategies.baselines.naive import NaiveStrategy
from pyclad.vision.callbacks.grouped_vision_pixel_concept_metric_callback import (
    GroupedVisionPixelConceptMetricCallback,
)
from pyclad.vision.data.benchmarks.continual_mega import ContinualMegaBenchmarkReader
from pyclad.vision.metrics.pixel_average_precision import PixelAveragePrecision
from pyclad.vision.models.fastflow.config import FastFlowConfig
from pyclad.vision.models.fastflow.fastflow import FastFlow

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    reader = ContinualMegaBenchmarkReader(
        data_root=pathlib.Path("../../resources/vision/continual_mega"),
        meta_dir=pathlib.Path("../../resources/vision/continual_mega/meta_files"),
        scenario=2,
        task_size=30,
        zero_shot=True,
        train_samples="normal",
    )
    dataset = reader.read_dataset()

    model = FastFlow(
        FastFlowConfig(
            input_size=(336, 336),
            backbone_name="resnet18",
            batch_size=16,
            epochs=20,
            learning_rate=1e-4,
        )
    )
    strategy = NaiveStrategy(model)

    groups = dataset.group_by_concept()
    summarized_metrics = [AverageAccuracy(), ForgettingMeasureStrict()]
    callbacks = [
        GroupedConceptMetricCallback(RocAuc(), groups, summarized_metrics),
        GroupedVisionPixelConceptMetricCallback(PixelAveragePrecision(), groups, summarized_metrics),
        TimeEvaluationCallback(),
    ]

    ConceptIncrementalScenario(dataset=dataset, strategy=strategy, callbacks=callbacks).run()

    JsonOutputWriter(pathlib.Path("output.json")).write([model, dataset, strategy, *callbacks])
