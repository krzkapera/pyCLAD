import math
from typing import Literal, Optional

import torch

CoresetMode = Literal["exact", "approximate"]


def _project(features: torch.Tensor, target_dim: int, generator: torch.Generator) -> torch.Tensor:
    if features.shape[1] == target_dim:
        return features

    # Same distribution as the reference's unbiased nn.Linear, drawn from `generator` instead of
    # the global RNG so that a run is reproducible from UCADConfig.seed alone.
    bound = 1.0 / math.sqrt(features.shape[1])
    projection = torch.empty(features.shape[1], target_dim).uniform_(-bound, bound, generator=generator)
    return features @ projection.to(features.device)


def _greedy_select(
    selection_space: torch.Tensor, anchor_distances: torch.Tensor, target_size: int, seed_indices: list[int]
) -> list[int]:
    selected = list(seed_indices)

    while len(selected) < target_size:
        current_idx = int(torch.argmax(anchor_distances))
        selected.append(current_idx)
        distances_to_new = torch.norm(selection_space - selection_space[current_idx], dim=1)
        anchor_distances = torch.minimum(anchor_distances, distances_to_new)

    return selected


def greedy_coreset_sampling(
    features: torch.Tensor,
    target_size: int,
    generator: torch.Generator,
    device: Optional[torch.device] = None,
    mode: CoresetMode = "exact",
    num_starting_points: int = 10,
    projection_dim: int = 128,
) -> torch.Tensor:
    """Selects target_size representative vectors out of features.

    'exact' seeds with the first vector and measures distances in the original feature space.
    'approximate' seeds with the mean distance to random starting points and measures distances
    in a randomly projected space; both of those draws come from `generator`.
    """
    if features.shape[0] <= target_size:
        return features

    if device is None:
        device = features.device
    features = features.to(device)

    if mode == "exact":
        anchor_distances = torch.norm(features - features[0], dim=1)
        indices = _greedy_select(features, anchor_distances, target_size, seed_indices=[0])
    else:
        selection_space = _project(features, projection_dim, generator)
        starting_points = torch.randperm(len(selection_space), generator=generator)[
            : min(num_starting_points, len(selection_space))
        ].to(device)
        anchor_distances = torch.cdist(selection_space, selection_space[starting_points]).mean(dim=1)
        indices = _greedy_select(selection_space, anchor_distances, target_size, seed_indices=[])

    return features[indices]
