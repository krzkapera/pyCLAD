import torch
import torch.nn.functional as F


def patchcore_aggregate(
    features: torch.Tensor, grid_size: tuple[int, int], patchsize: int = 3, output_dim: int = 1024
) -> torch.Tensor:
    B, Np, C = features.shape
    h, w = grid_size
    k2 = patchsize * patchsize

    x = features.transpose(1, 2).reshape(B, C, h, w)
    x = F.unfold(x, kernel_size=patchsize, padding=(patchsize - 1) // 2)
    x = x.reshape(B, C, k2, Np).permute(0, 3, 1, 2).reshape(B * Np, 1, C * k2)
    x = F.adaptive_avg_pool1d(x, output_dim)

    return x.reshape(B, Np, output_dim)
