from abc import abstractmethod
from typing import Optional

import numpy as np

from pyclad.vision.models.vision_model import VisionModel


class SupervisedVisionModel(VisionModel):
    @abstractmethod
    def fit_supervised(self, data: np.ndarray, labels: np.ndarray, masks: Optional[np.ndarray] = None) -> None: ...
