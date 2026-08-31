from typing import Dict

import numpy as np

from pyclad.data.concept import Concept
from pyclad.strategies.strategy import SupervisedStrategy
from pyclad.vision.data.vision_concept import VisionConcept
from pyclad.vision.models.supervised_vision_model import SupervisedVisionModel
from pyclad.vision.prediction_results import VisionPredictionResults


class NaiveSupervisedStrategy(SupervisedStrategy):
    def __init__(self, model: SupervisedVisionModel):
        self._model = model

    def learn_concept(self, concept: Concept) -> None:
        if concept.labels is None:
            raise ValueError(f"Supervised training requires labelled concepts, got none for '{concept.name}'")
        masks = concept.masks if isinstance(concept, VisionConcept) else None
        self._model.fit_supervised(data=concept.data, labels=concept.labels, masks=masks)

    def predict(self, data: np.ndarray) -> VisionPredictionResults:
        return self._model.predict(data)

    def name(self) -> str:
        return "NaiveSupervised"

    def additional_info(self) -> Dict:
        return {"model": self._model.name()}
