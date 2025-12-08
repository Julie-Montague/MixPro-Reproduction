from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def sample_beta_tau(batch_size: int, beta: float, device: torch.device) -> torch.Tensor:
    dist = torch.distributions.Beta(beta, beta)
    return dist.sample((batch_size,)).to(device)


def make_random_mask_patch(
    batch_size: int,
    grid_h: int,
    grid_w: int,
    scale: int = 1,
    beta: float = 1.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    if device is None:
        device = torch.device("cpu")

    tau = sample_beta_tau(batch_size, beta=beta, device=device)  # [B]

    coarse_h = max(1, int(math.ceil(grid_h / scale)))
    coarse_w = max(1, int(math.ceil(grid_w / scale)))
    S = coarse_h * coarse_w

    r = torch.rand(batch_size, S, device=device)
    order = r.argsort(dim=1)

    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(S, device=device).expand(batch_size, S))

    n = torch.floor(tau * S).long().clamp(min=0, max=S)
    mask_flat = (ranks < n.unsqueeze(1)).float()

    mask_coarse = mask_flat.view(batch_size, 1, coarse_h, coarse_w)
    mask_patch = F.interpolate(mask_coarse, size=(grid_h, grid_w), mode="nearest").squeeze(1)
    return mask_patch


def apply_maskmix(
    x_i: torch.Tensor,
    x_j: torch.Tensor,
    mask_patch: torch.Tensor,
    patch_size_px: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, C, H, W = x_i.shape
    device = x_i.device

    mask_pix = mask_patch.repeat_interleave(patch_size_px, dim=1).repeat_interleave(patch_size_px, dim=2)
    mask_pix = mask_pix[:, :H, :W].to(device)
    mask_pix = mask_pix.unsqueeze(1)  # [B,1,H,W]

    x_mix = mask_pix * x_i + (1.0 - mask_pix) * x_j
    lambda_area = mask_patch.mean(dim=(1, 2))
    return x_mix, lambda_area
