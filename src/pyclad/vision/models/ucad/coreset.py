import torch
from typing import Optional


def greedy_coreset_sampling(
    features: torch.Tensor,
    target_size: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if features.shape[0] <= target_size:
        return features

    if device is None:
        device = features.device

    features = features.to(device)

    selected_indices = [0]
    min_distances = torch.norm(features - features[0], dim=1)

    for _ in range(1, target_size):
        current_idx = torch.argmax(min_distances).item()
        selected_indices.append(current_idx)

        distances_to_new = torch.norm(features - features[current_idx], dim=1)
        min_distances = torch.minimum(min_distances, distances_to_new)

    return features[selected_indices]
