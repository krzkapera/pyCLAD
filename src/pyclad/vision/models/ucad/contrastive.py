from typing import Literal

import torch
import torch.nn.functional as F

LossMode = Literal["linear", "exp_negatives"]


def structure_contrastive_loss(
    features: torch.Tensor,
    mask_labels: torch.Tensor,
    mode: LossMode = "exp_negatives",
    temperature: float = 0.5,
) -> torch.Tensor:
    features_normalized = F.normalize(features, dim=2)
    similarity_matrix = torch.bmm(features_normalized, features_normalized.transpose(1, 2))
    mask = (mask_labels.unsqueeze(1) == mask_labels.unsqueeze(2)).float()

    if mode == "linear":
        return ((1 - mask) * similarity_matrix - mask * similarity_matrix).mean()

    similarity_matrix = similarity_matrix / temperature
    return (-similarity_matrix * mask + (1 - mask) * similarity_matrix.exp()).mean()
