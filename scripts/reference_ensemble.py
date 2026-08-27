# TEMPORARY - REFERENCE REPRODUCTION ONLY. NOT PART OF UCAD AS PUBLISHED. DO NOT MERGE.
#
# The paper stores one prompt and one knowledge bank per concept, and its own memory accounting
# says so. The authors' released code does something the paper never describes: after every epoch
# it scores the whole test set, rescales that epoch's scores to 0..1 across the test set, and
# averages every epoch so far - so the number it reports is the mean opinion of 25 different models
# rather than the score of one. Reproducing their published figures requires scoring the same way.
#
# Nothing of this is in the library. The model is subclassed here, the per-epoch states are kept
# here, and the base memory bank still holds the single prompt and bank the method defines - the
# last one - so task routing and the continual scenario are unaffected.

from typing import List, Sequence

import numpy as np

from pyclad.vision.prediction_results import VisionPredictionResults

from pyclad.vision.models.ucad.ucad_model import PromptedBank, TaskTraining, UCADModel


class ReferenceEnsembleUCAD(UCADModel):
    def __init__(self, config, members: int, mask_provider=None):
        super().__init__(config, mask_provider=mask_provider)
        self.members = members
        self._task_members: List[List[PromptedBank]] = []

    def fit(self, training_data):
        task = self._begin_task(training_data)
        wanted = snapshot_epochs(self.config.training_epochs, self.members)
        states: List[PromptedBank] = []

        for epoch in range(self.config.training_epochs):
            self._train_epoch(task, epoch)
            if epoch + 1 in wanted:
                states.append(self._snapshot(task))

        if not states:
            states.append(self._snapshot(task))

        self._task_members.append(states)
        self._end_task(task, *states[-1])

    def member_predictions(self, data) -> List[tuple[np.ndarray, np.ndarray]]:
        members = max(len(states) for states in self._task_members)
        return [
            self._score_dataset(data, [task[min(member, len(task) - 1)] for task in self._task_members])
            for member in range(members)
        ]

    def predict(self, data) -> VisionPredictionResults:
        outputs = self.member_predictions(data)
        scores, maps = combine_members([s for s, _ in outputs], [m for _, m in outputs])
        return VisionPredictionResults(y_pred=np.zeros_like(scores, dtype=int), anomaly_scores=scores, score_maps=maps)


def snapshot_epochs(training_epochs: int, members: int) -> set[int]:
    if members >= training_epochs:
        return set(range(1, training_epochs + 1))
    return {round(training_epochs / members * (index + 1)) for index in range(members)}


def combine_members(
    member_scores: Sequence[np.ndarray], member_maps: Sequence[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    if len(member_scores) == 1:
        return member_scores[0], member_maps[0]

    def normalized(values: np.ndarray) -> np.ndarray:
        low, high = values.min(), values.max()
        return np.zeros_like(values) if high == low else (values - low) / (high - low)

    scores = np.mean([normalized(scores) for scores in member_scores], axis=0)
    maps = np.mean([normalized(maps) for maps in member_maps], axis=0)
    return scores, maps
