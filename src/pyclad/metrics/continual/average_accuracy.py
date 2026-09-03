import numpy as np

from pyclad.metrics.continual.concepts_metric import (
    ConceptLevelMatrix,
    SummarizedMetric,
)


class AverageAccuracy(SummarizedMetric):
    def compute(self, metric_matrix: ConceptLevelMatrix) -> float:
        if len(metric_matrix) == 0 or len(metric_matrix[-1]) == 0:
            return 0
        return float(np.nanmean(metric_matrix[-1]))

    def name(self) -> str:
        return "AverageAccuracy"
