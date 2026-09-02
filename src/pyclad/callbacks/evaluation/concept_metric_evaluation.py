from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from pyclad.callbacks.callback import Callback
from pyclad.data.concept import Concept
from pyclad.metrics.base.base_metric import BaseMetric
from pyclad.metrics.continual.concepts_metric import (
    ConceptLevelMatrix,
    StepwiseConceptMetric,
    SummarizedMetric,
)
from pyclad.output.output_writer import InfoProvider


class ConceptMetricCallback(Callback, InfoProvider):
    def __init__(
        self,
        base_metric: BaseMetric,
        summarized_metrics: Iterable[SummarizedMetric] = (),
        stepwise_metrics: Iterable[StepwiseConceptMetric] = None,
    ):
        self._base_metric: BaseMetric = base_metric
        self._summarized_metrics = summarized_metrics
        self._stepwise_metrics = stepwise_metrics if stepwise_metrics is not None else []
        self._learned_concepts: List[str] = []
        self._columns: List[str] = []
        self._values: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self._evaluated_per_row: Dict[str, List[str]] = defaultdict(list)

    def after_training(self, learned_concept: Concept, *args, **kwargs):
        self._learned_concepts.append(learned_concept.name)

    def after_evaluation(
        self,
        evaluated_concept: Concept,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        anomaly_scores: np.ndarray,
        score_maps: Optional[np.ndarray] = None,
        *args,
        **kwargs,
    ):
        value = self._concept_value(evaluated_concept, y_true, y_pred, anomaly_scores, score_maps)
        if value is None:
            return

        row = self._learned_concepts[-1]
        assert (
            evaluated_concept.name not in self._evaluated_per_row[row]
        ), "The same concept should not be evaluated twice after the same learned concept"
        self._evaluated_per_row[row].append(evaluated_concept.name)

        column = self._column(evaluated_concept)
        if column not in self._columns:
            self._columns.append(column)
        self._values[row][column].append(value)

    def info(self) -> Dict[str, Any]:
        if self._learned_concepts and not self._values:
            return {}

        metric_matrix = {
            row: {column: float(np.nanmean(values)) for column, values in columns.items()}
            for row, columns in self._values.items()
        }
        square_matrix = self._square_matrix(metric_matrix)
        held_out_columns = [column for column in self._columns if column not in self._learned_concepts]

        payload = {
            "base_metric_name": self._base_metric.name(),
            "metrics": {m.name(): m.compute(square_matrix) for m in self._summarized_metrics},
            "stepwise_metrics": {m.name(): m.compute(square_matrix) for m in self._stepwise_metrics},
            "concepts_order": list(self._learned_concepts),
            "metric_matrix": metric_matrix,
            **self._extra_info(),
        }
        if held_out_columns:
            payload["held_out_columns"] = {
                column: {row: metric_matrix[row][column] for row in self._learned_concepts}
                for column in held_out_columns
            }

        return {f"{self._info_key()}_{self._base_metric.name()}": payload}

    def _concept_value(
        self,
        evaluated_concept: Concept,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        anomaly_scores: np.ndarray,
        score_maps: Optional[np.ndarray],
    ) -> Optional[float]:
        return self._base_metric.compute(anomaly_scores=anomaly_scores, y_true=y_true, y_pred=y_pred)

    def _column(self, evaluated_concept: Concept) -> str:
        return evaluated_concept.name

    def _info_key(self) -> str:
        return "concept_metric_callback"

    def _extra_info(self) -> Dict[str, Any]:
        return {}

    def _square_matrix(self, metric_matrix: Dict[str, Dict[str, float]]) -> ConceptLevelMatrix:
        if not self._learned_concepts:
            return [[]]
        return [
            [metric_matrix.get(row, {}).get(column, float("nan")) for column in self._learned_concepts]
            for row in self._learned_concepts
        ]
