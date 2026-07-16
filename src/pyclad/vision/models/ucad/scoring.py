import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter


class NearestNeighborScorer:
    def __init__(self, n_neighbors: int = 5, blur_sigma: float = 3.0):
        self.n_neighbors = n_neighbors
        self.blur_sigma = blur_sigma

    def _compute_distance(
        self, query_features: torch.Tensor, support_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, Np, C = query_features.shape

        query_flat = query_features.reshape(B * Np, C)
        distances = torch.cdist(query_flat, support_features)

        if self.n_neighbors == 1:
            min_dist, min_idx = torch.min(distances, dim=1)

            return min_dist.reshape(B, Np), min_idx.reshape(B, Np)
        else:
            topk_vals, topk_idx = torch.topk(-distances, k=self.n_neighbors, dim=1)
            topk_vals = -topk_vals
            d_1 = topk_vals[:, 0]

            max_d = topk_vals[:, -1]
            sum_d = torch.sum(topk_vals, dim=1)

            weight = 1 - torch.exp(-max_d / (sum_d + 1e-8))
            final_dist = weight * d_1

            return final_dist.reshape(B, Np), topk_idx[:, 0].reshape(B, Np)

    def predict(
        self,
        test_features: torch.Tensor,
        knowledge_bank: torch.Tensor,
        input_size: tuple[int, int] = (224, 224),
    ) -> tuple[np.ndarray, np.ndarray]:
        B, Np, C = test_features.shape
        grid_size = int(np.sqrt(Np))
        assert grid_size * grid_size == Np, f"Feature map must be square, got Np={Np}"

        patch_scores, _ = self._compute_distance(test_features, knowledge_bank)
        patch_scores_spatial = patch_scores.reshape(B, 1, grid_size, grid_size)

        anomaly_maps = F.interpolate(
            patch_scores_spatial,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        anomaly_maps = anomaly_maps.squeeze(1).cpu().numpy()
        for i in range(B):
            anomaly_maps[i] = gaussian_filter(anomaly_maps[i], sigma=self.blur_sigma)

        image_scores = np.max(anomaly_maps, axis=(1, 2))

        return image_scores, anomaly_maps
