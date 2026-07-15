import torch


def greedy_coreset_sampling(features: torch.Tensor, target_size: int, device: torch.device = None) -> torch.Tensor:
    """
    Greedy coreset subsampling (Farthest Point Sampling) algorithm.
    
    This function selects a subset of features such that the selected subset
    covers the original set as well as possible. It iteratively picks the point
    that is farthest from the already selected points.
    
    Args:
        features: Tensor of shape (N, C) containing the features to subsample.
        target_size: The number of features to select.
        device: Device to perform computations on. If None, uses features' device.
        
    Returns:
        Tensor of shape (target_size, C) containing the subsampled features.
    """
    if features.shape[0] <= target_size:
        return features
        
    if device is None:
        device = features.device
        
    features = features.to(device)
    
    # Initialize the selected indices list
    selected_indices = []
    
    # Start with a random point (or the first point for determinism)
    # Using the first point here for reproducibility, although a random point
    # is often used in practice.
    current_idx = 0
    selected_indices.append(current_idx)
    
    # Keep track of the minimum distance from each point to the selected set
    # Initialize with the distance to the first selected point
    min_distances = torch.cdist(
        features.unsqueeze(0), 
        features[current_idx].unsqueeze(0).unsqueeze(0)
    ).squeeze()
    
    for _ in range(1, target_size):
        # Find the point that is farthest from the selected set
        current_idx = torch.argmax(min_distances).item()
        selected_indices.append(current_idx)
        
        # Calculate distances from all points to the newly selected point
        distances_to_new = torch.cdist(
            features.unsqueeze(0), 
            features[current_idx].unsqueeze(0).unsqueeze(0)
        ).squeeze()
        
        # Update the minimum distances
        min_distances = torch.minimum(min_distances, distances_to_new)
        
    return features[selected_indices]
