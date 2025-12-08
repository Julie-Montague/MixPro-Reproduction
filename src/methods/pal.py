from __future__ import annotations

import torch
import torch.nn.functional as F


def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(labels, num_classes=num_classes).float()


@torch.no_grad()
def compute_lambda_attn(A: torch.Tensor, mask_patch: torch.Tensor) -> torch.Tensor:
    """
    A: [B, num_patches] cls->patch attention, ideally normalized to sum=1 over patches
    mask_patch: [B, H, W] or [B, num_patches] where 1 means "from image i"
    """
    M = mask_patch.flatten(1).float()
    return (A * M).sum(dim=1).clamp(0.0, 1.0)


@torch.no_grad()
def progressive_alpha(p: torch.Tensor, y_area: torch.Tensor) -> torch.Tensor:
    """
    Eq.(8): alpha = cosine_similarity(p, y_e)
    where y_e is the *area-mixed* label (Algorithm 1).
    """
    a = F.cosine_similarity(p, y_area, dim=1, eps=1e-8)
    return a.clamp(0.0, 1.0)


def build_mixpro_labels(
    y_i: torch.Tensor,
    y_j: torch.Tensor,
    num_classes: int,
    lambda_attn: torch.Tensor,
    lambda_area: torch.Tensor,
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      y_final: [B, C] soft label using lambda_final
      lambda_final: [B]
      alpha: [B]
    """
    yi = one_hot(y_i, num_classes)
    yj = one_hot(y_j, num_classes)

    # Algorithm 1: y_e based on lambda_area (BEFORE attention reweighting)
    y_area = lambda_area.unsqueeze(1) * yi + (1.0 - lambda_area).unsqueeze(1) * yj

    # Eq.(8): alpha = cos_similarity(softmax(logits), y_area)
    with torch.no_grad():
        p = torch.softmax(logits, dim=1)
        alpha = progressive_alpha(p, y_area)

    # Eq.(9): lambda = alpha*lambda_attn + (1-alpha)*lambda_area
    lambda_final = alpha * lambda_attn + (1.0 - alpha) * lambda_area
    lambda_final = lambda_final.clamp(0.0, 1.0)

    # Final mixed label
    y_final = lambda_final.unsqueeze(1) * yi + (1.0 - lambda_final).unsqueeze(1) * yj
    return y_final, lambda_final, alpha
