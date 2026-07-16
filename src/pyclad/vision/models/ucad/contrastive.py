import torch
import torch.nn.functional as F


def structure_contrastive_loss(
    features: torch.Tensor, mask_labels: torch.Tensor, temperature: float = 0.5
) -> torch.Tensor:
    features_normalized = F.normalize(features, dim=2)
    similarity_matrix = torch.bmm(features_normalized, features_normalized.transpose(1, 2)) / temperature

    mask = (mask_labels.unsqueeze(1) == mask_labels.unsqueeze(2)).float()
    loss = (-similarity_matrix * mask + (1 - mask) * similarity_matrix.exp()).mean()

    return loss
