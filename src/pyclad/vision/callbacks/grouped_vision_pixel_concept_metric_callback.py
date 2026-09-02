from pyclad.callbacks.evaluation.grouped_concept_metric_evaluation import (
    GroupedConceptMetricCallback,
)
from pyclad.vision.callbacks.vision_pixel_concept_metric_callback import (
    VisionPixelConceptMetricCallback,
)


class GroupedVisionPixelConceptMetricCallback(GroupedConceptMetricCallback, VisionPixelConceptMetricCallback):
    pass
