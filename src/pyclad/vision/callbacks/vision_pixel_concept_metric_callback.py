from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.data.concept import Concept
from pyclad.vision.data.vision_concept import VisionConcept

EMPTY_PREDICTIONS = np.asarray([], dtype=np.uint8)


class VisionPixelConceptMetricCallback(ConceptMetricCallback):
    def _concept_value(
        self,
        evaluated_concept: Concept,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        anomaly_scores: np.ndarray,
        score_maps: Optional[np.ndarray],
    ) -> Optional[float]:
        if score_maps is None or not isinstance(evaluated_concept, VisionConcept) or evaluated_concept.masks is None:
            return None
        return self._base_metric.compute(
            anomaly_scores=score_maps,
            y_true=evaluated_concept.masks,
            y_pred=EMPTY_PREDICTIONS,
        )

    def _info_key(self) -> str:
        return "pixel_concept_metric_callback"

    def _extra_info(self) -> Dict[str, Any]:
        return {"evaluation_level": "pixel"}
