import torch
import torch.nn.functional as F


def structure_contrastive_loss(features: torch.Tensor, mask_labels: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    Structure-based Contrastive Loss (SCL) as described in the UCAD paper.
    
    Features belonging to the same segment (mask) are pulled together,
    while features from different segments are pushed apart.
    
    Args:
        features: Tensor of shape (B, N, C) containing patch embeddings.
        mask_labels: Tensor of shape (B, N) containing integer labels for each patch,
                     where patches with the same label belong to the same SAM mask.
        temperature: Temperature scaling factor for the softmax.
        
    Returns:
        A scalar tensor representing the contrastive loss.
    """
    # Normalize features along channel dimension
    features_normalized = F.normalize(features, dim=2)
    
    # Calculate cosine similarity matrix: (B, N, N)
    # Divided by temperature to scale the logits
    similarity_matrix = torch.bmm(features_normalized, features_normalized.transpose(1, 2)) / temperature
    
    # Create mask matrix where True means same segment (positive pair)
    # mask_labels.unsqueeze(1) shape: (B, 1, N)
    # mask_labels.unsqueeze(2) shape: (B, N, 1)
    # mask shape: (B, N, N)
    mask = (mask_labels.unsqueeze(1) == mask_labels.unsqueeze(2)).float()
    
    # Loss calculation (Eq 3 in paper):
    # For positive pairs (mask=1), we want to maximize similarity, so we minimize -similarity
    # For negative pairs (mask=0), we want to minimize similarity, so we minimize exp(similarity)
    # This matches the implementation in the original UCAD code.
    loss = (-similarity_matrix * mask + (1 - mask) * similarity_matrix.exp()).mean()
    
    return loss
