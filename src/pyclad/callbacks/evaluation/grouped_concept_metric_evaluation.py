from typing import Iterable, Mapping

from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.data.concept import Concept
from pyclad.metrics.base.base_metric import BaseMetric
from pyclad.metrics.continual.concepts_metric import (
    StepwiseConceptMetric,
    SummarizedMetric,
)


class GroupedConceptMetricCallback(ConceptMetricCallback):
    def __init__(
        self,
        base_metric: BaseMetric,
        group_by_concept: Mapping[str, str],
        summarized_metrics: Iterable[SummarizedMetric] = (),
        stepwise_metrics: Iterable[StepwiseConceptMetric] = None,
    ):
        super().__init__(base_metric, summarized_metrics, stepwise_metrics)
        self._group_by_concept = dict(group_by_concept)

    def _column(self, evaluated_concept: Concept) -> str:
        return self._group_by_concept[evaluated_concept.name]

    def _info_key(self) -> str:
        return f"grouped_{super()._info_key()}"
