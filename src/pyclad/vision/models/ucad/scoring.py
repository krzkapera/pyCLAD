import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


class NearestNeighborScorer:
    def __init__(self, num_nn: int = 1, reweighting_num_nn: int = 0, blur_sigma: float = 3.0):
        self.num_nn = num_nn
        self.reweighting_num_nn = reweighting_num_nn
        self.blur_sigma = blur_sigma

    def predict(
        self,
        test_features: torch.Tensor,
        knowledge_bank: torch.Tensor,
        input_size: tuple[int, int] = (224, 224),
    ) -> tuple[np.ndarray, np.ndarray]:
        B, Np, C = test_features.shape
        grid_size = int(np.sqrt(Np))
        assert grid_size * grid_size == Np, f"Feature map must be square, got Np={Np}"

        distances = torch.cdist(test_features.reshape(B * Np, C), knowledge_bank).reshape(B, Np, -1)

        if self.reweighting_num_nn > 1:
            patch_scores = distances.min(dim=2).values
            image_scores = self._reweighted_image_scores(distances, patch_scores, knowledge_bank)
        else:
            knn_distances, _ = torch.topk(distances, k=self.num_nn, dim=2, largest=False)
            patch_scores = knn_distances.mean(dim=2)
            image_scores = patch_scores.max(dim=1).values

        anomaly_maps = self._build_anomaly_maps(patch_scores.reshape(B, 1, grid_size, grid_size), input_size)
        return image_scores.cpu().numpy(), anomaly_maps

    def _reweighted_image_scores(
        self, distances: torch.Tensor, patch_scores: torch.Tensor, knowledge_bank: torch.Tensor
    ) -> torch.Tensor:
        batch_idx = torch.arange(distances.shape[0], device=distances.device)

        star_patch = patch_scores.argmax(dim=1)
        star_distances = distances[batch_idx, star_patch]
        s_star = patch_scores[batch_idx, star_patch]

        nearest_bank_idx = star_distances.argmin(dim=1)
        bank_neighbors = torch.cdist(knowledge_bank, knowledge_bank).topk(
            self.reweighting_num_nn, dim=1, largest=False
        ).indices
        neighbor_distances = torch.gather(star_distances, 1, bank_neighbors[nearest_bank_idx])

        weight = 1 - torch.exp(s_star - torch.logsumexp(neighbor_distances, dim=1))
        return weight * s_star

    def _build_anomaly_maps(self, patch_scores_spatial: torch.Tensor, input_size: tuple[int, int]) -> np.ndarray:
        anomaly_maps = F.interpolate(patch_scores_spatial, size=input_size, mode="bilinear", align_corners=False)
        anomaly_maps = anomaly_maps.squeeze(1).cpu().numpy()

        for i in range(len(anomaly_maps)):
            anomaly_maps[i] = gaussian_filter(anomaly_maps[i], sigma=self.blur_sigma)

        return anomaly_maps
