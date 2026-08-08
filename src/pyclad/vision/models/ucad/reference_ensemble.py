# TEMPORARY - REFERENCE REPRODUCTION ONLY, NOT PART OF UCAD AS PUBLISHED. DO NOT MERGE.
#
# The paper stores one prompt and one knowledge bank per concept. The authors' released code does
# something its paper never describes: after every epoch it scores the whole test set, rescales each
# epoch's scores to 0..1 and averages every epoch so far, so the number it reports is the mean
# opinion of 25 models rather than the score of one. Reproducing their published numbers requires
# scoring the same way, which is why this exists; it is evaluation machinery of theirs, not method.
#
# Everything ensemble-specific lives here. What remains outside is marked `reference-protocol:`
# and is only: the score_ensemble_epochs config field, the list in TaskMemory.states, and the member
# loop in UCADModel.predict. Removing this file and those four sites drops UCAD back to the paper.

import numpy as np


def snapshot_epochs(training_epochs: int, members: int) -> set[int]:
    if members >= training_epochs:
        return set(range(1, training_epochs + 1))
    return {round(training_epochs / members * (index + 1)) for index in range(members)}


def combine_members(
    member_scores: list[np.ndarray], member_maps: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    if len(member_scores) == 1:
        return member_scores[0], member_maps[0]

    def normalized(values: np.ndarray) -> np.ndarray:
        low, high = values.min(), values.max()
        return np.zeros_like(values) if high == low else (values - low) / (high - low)

    scores = np.mean([normalized(scores) for scores in member_scores], axis=0)
    maps = np.mean([normalized(maps) for maps in member_maps], axis=0)
    return scores, maps
