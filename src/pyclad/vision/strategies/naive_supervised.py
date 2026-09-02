from typing import Dict

import numpy as np

from pyclad.data.concept import Concept
from pyclad.strategies.exceptions import SupervisionRequiredError
from pyclad.strategies.strategy import ConceptIncrementalStrategy
from pyclad.vision.data.vision_concept import VisionConcept
from pyclad.vision.models.supervised_vision_model import SupervisedVisionModel
from pyclad.vision.prediction_results import VisionPredictionResults


class NaiveSupervisedStrategy(ConceptIncrementalStrategy):
    def __init__(self, model: SupervisedVisionModel):
        self._model = model

    def learn(self, data: np.ndarray) -> None:
        raise SupervisionRequiredError(f"{self.name()} learns from labelled concepts, not from raw data")

    def learn_concept(self, concept: Concept) -> None:
        if concept.labels is None:
            raise SupervisionRequiredError(f"{self.name()} requires labels, concept '{concept.name}' has none")
        masks = concept.masks if isinstance(concept, VisionConcept) else None
        self._model.fit(data=concept.data, labels=concept.labels, masks=masks)

    def predict(self, data: np.ndarray) -> VisionPredictionResults:
        return self._model.predict(data)

    def name(self) -> str:
        return "NaiveSupervised"

    def additional_info(self) -> Dict:
        return {"model": self._model.name()}
