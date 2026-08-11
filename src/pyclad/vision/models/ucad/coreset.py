from __future__ import annotations

import math
from typing import Literal, Optional

import torch

CoresetMode = Literal["greedy", "approximate"]


def _project(features: torch.Tensor, target_dim: int, generator: torch.Generator) -> torch.Tensor:
    if features.shape[1] == target_dim:
        return features

    bound = 1.0 / math.sqrt(features.shape[1])
    projection = torch.empty(features.shape[1], target_dim).uniform_(-bound, bound, generator=generator)
    return features @ projection.to(features.device)


def _select_from_matrix(distances: torch.Tensor, anchor_distances: torch.Tensor, target_size: int) -> list[int]:
    selected: list[int] = []

    while len(selected) < target_size:
        current_idx = int(torch.argmax(anchor_distances))
        selected.append(current_idx)
        anchor_distances = torch.minimum(anchor_distances, distances[:, current_idx])

    return selected


def _select_iteratively(
    selection_space: torch.Tensor, anchor_distances: torch.Tensor, target_size: int
) -> list[int]:
    selected: list[int] = []

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
    mode: CoresetMode = "approximate",
    num_starting_points: int = 10,
    projection_dim: int = 128,
) -> torch.Tensor:
    if features.shape[0] <= target_size:
        return features

    if device is None:
        device = features.device
    features = features.to(device)
    selection_space = _project(features, projection_dim, generator)

    if mode == "greedy":
        distances = torch.cdist(selection_space, selection_space)
        indices = _select_from_matrix(distances, torch.norm(distances, dim=1), target_size)
    else:
        starting_points = torch.randperm(len(selection_space), generator=generator)[
            : min(num_starting_points, len(selection_space))
        ].to(device)
        anchor_distances = torch.cdist(selection_space, selection_space[starting_points]).mean(dim=1)
        indices = _select_iteratively(selection_space, anchor_distances, target_size)

    return features[indices]
