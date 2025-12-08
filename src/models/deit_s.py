# src/models/deit_s.py
from __future__ import annotations

import types
from typing import Tuple

import torch
import timm


def build_deit_s(
    num_classes: int = 1000,
    img_size: int = 224,
    drop_path_rate: float = 0.1,
    pretrained: bool = False,
) -> torch.nn.Module:
    """
    Build DeiT-Small (deit_small_patch16_224)
    """
    model = timm.create_model(
        "deit_small_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
        img_size=img_size,
        drop_path_rate=drop_path_rate,
    )
    return model


# ────────────────────────────────
# OPTIONAL: For attention visualization
# ────────────────────────────────

def enable_last_attn_capture(model: torch.nn.Module) -> None:
    """
    Patches the last attention layer to store attention weights during forward pass.
    """
    if not hasattr(model, "blocks") or len(model.blocks) == 0:
        raise ValueError("Expected a ViT/DeiT-like model with .blocks")

    attn = model.blocks[-1].attn
    if not all(hasattr(attn, k) for k in ["qkv", "num_heads", "scale", "attn_drop", "proj", "proj_drop"]):
        raise ValueError("Unexpected attention module; cannot patch safely.")

    def patched_forward(self, x: torch.Tensor, attn_mask=None, **kwargs) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_w = (q @ k.transpose(-2, -1)) * self.scale
        attn_w = attn_w.softmax(dim=-1)
        self.last_attn = attn_w.detach()

        attn_w = self.attn_drop(attn_w)
        x = (attn_w @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    attn.forward = types.MethodType(patched_forward, attn)


@torch.no_grad()
def get_cls_to_patch_attention(model: torch.nn.Module, normalize: bool = True) -> torch.Tensor:
    """
    Extracts average attention from CLS token to patches (last layer).
    """
    attn = model.blocks[-1].attn.last_attn  # [B, heads, N, N]
    if attn is None:
        raise RuntimeError("No attention captured. Call enable_last_attn_capture() and run a forward pass.")

    cls_to_patches = attn[:, :, 0, 1:]    # [B, heads, num_patches]
    A = cls_to_patches.mean(dim=1)        # [B, num_patches]
    if normalize:
        A = A / (A.sum(dim=1, keepdim=True) + 1e-8)
    return A


def get_patch_grid_and_size(model: torch.nn.Module) -> Tuple[int, int, int]:
    """
    Returns (grid_h, grid_w, patch_size) used in ViT patch embedding.
    """
    pe = model.patch_embed
    grid = getattr(pe, "grid_size", None)
    patch = getattr(pe, "patch_size", None)
    if grid is None or patch is None:
        raise ValueError("Could not read patch_embed.grid_size/patch_size from model.")

    grid_h, grid_w = int(grid[0]), int(grid[1])
    patch_size = int(patch[0]) if isinstance(patch, tuple) else int(patch)
    return grid_h, grid_w, patch_size
