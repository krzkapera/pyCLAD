import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter


class NearestNeighborScorer:
    """
    Nearest-neighbor anomaly scorer based on PatchCore, with distance re-weighting
    as used in UCAD.
    """
    def __init__(self, n_neighbors: int = 1, blur_sigma: float = 3.0):
        self.n_neighbors = n_neighbors
        self.blur_sigma = blur_sigma
        
    def _compute_distance(self, query_features: torch.Tensor, support_features: torch.Tensor) -> torch.Tensor:
        """
        Computes the nearest neighbor distance from query features to support features.
        
        Args:
            query_features: Tensor of shape (B, Np, C)
            support_features: Tensor of shape (M, C) containing the knowledge bank
            
        Returns:
            distances: Tensor of shape (B, Np) containing the distance to nearest neighbor
            indices: Tensor of shape (B, Np) containing the index of the nearest neighbor
        """
        B, Np, C = query_features.shape
        M = support_features.shape[0]
        
        # Flatten query features to (B * Np, C)
        query_flat = query_features.reshape(B * Np, C)
        
        # Compute pairwise distances
        # Using cdist for efficiency (can be memory intensive for very large M)
        distances = torch.cdist(query_flat, support_features)
        
        if self.n_neighbors == 1:
            # Simple min distance
            min_dist, min_idx = torch.min(distances, dim=1)
            
            return min_dist.reshape(B, Np), min_idx.reshape(B, Np)
        else:
            # Top-k nearest neighbors
            # Note: torch.topk returns largest, so we negate distances
            topk_vals, topk_idx = torch.topk(-distances, k=self.n_neighbors, dim=1)
            topk_vals = -topk_vals
            
            # The distance is typically the mean of the top-k distances,
            # or the distance to the 1st neighbor re-weighted by the others.
            # In PatchCore/UCAD, they use a specific re-weighting scheme.
            
            # Distance to the nearest neighbor
            d_1 = topk_vals[:, 0]
            
            # Equation 5-6 in UCAD/PatchCore: re-weighting
            # Weight = 1 - exp(-max(d_k) / sum(d_k))
            max_d = topk_vals[:, -1]
            sum_d = torch.sum(topk_vals, dim=1)
            weight = 1 - torch.exp(-max_d / (sum_d + 1e-8))
            
            # Final distance: w * d_1
            final_dist = weight * d_1
            
            return final_dist.reshape(B, Np), topk_idx[:, 0].reshape(B, Np)

    def predict(
        self, 
        test_features: torch.Tensor, 
        knowledge_bank: torch.Tensor,
        input_size: tuple[int, int] = (224, 224)
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predicts anomaly scores and segmentation maps for test features.
        
        Args:
            test_features: Tensor of shape (B, Np, C)
            knowledge_bank: Tensor of shape (M, C)
            input_size: The original image size (H, W) to upsample the map to.
            
        Returns:
            image_scores: Array of shape (B,) containing image-level anomaly scores.
            anomaly_maps: Array of shape (B, H, W) containing pixel-level anomaly scores.
        """
        B, Np, C = test_features.shape
        
        # Assume square feature map
        grid_size = int(np.sqrt(Np))
        assert grid_size * grid_size == Np, f"Feature map must be square, got Np={Np}"
        
        # 1. Compute nearest neighbor distances
        patch_scores, _ = self._compute_distance(test_features, knowledge_bank)
        
        # Reshape to spatial grid: (B, 1, grid_size, grid_size)
        patch_scores_spatial = patch_scores.reshape(B, 1, grid_size, grid_size)
        
        # 2. Upsample to original image size
        anomaly_maps = F.interpolate(
            patch_scores_spatial,
            size=input_size,
            mode="bilinear",
            align_corners=False
        )
        
        anomaly_maps = anomaly_maps.squeeze(1).cpu().numpy()
        
        # 3. Apply Gaussian blur to the anomaly maps
        for i in range(B):
            anomaly_maps[i] = gaussian_filter(anomaly_maps[i], sigma=self.blur_sigma)
            
        # 4. Compute image-level scores (max patch score)
        # PatchCore typically uses the max of the upsampled & smoothed map
        image_scores = np.max(anomaly_maps, axis=(1, 2))
        
        return image_scores, anomaly_maps
