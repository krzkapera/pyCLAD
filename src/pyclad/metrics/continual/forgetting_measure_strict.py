import numpy as np

from pyclad.metrics.continual.concepts_metric import (
    ConceptLevelMatrix,
    SummarizedMetric,
)


class ForgettingMeasureStrict(SummarizedMetric):
    def compute(self, metric_matrix: ConceptLevelMatrix) -> float:
        if len(metric_matrix) < 2:
            return 0
        values = np.asarray(metric_matrix, dtype=float)
        best_before_last = np.nanmax(values[:-1, :-1], axis=0)
        return float(np.nanmean(best_before_last - values[-1, :-1]))

    def name(self) -> str:
        return "ForgettingMeasureStrict"
