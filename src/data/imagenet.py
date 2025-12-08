# src/data/imagenet.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


@dataclass
class ImageNetDataConfig:
    data_dir: str = "data/imagenet1k"   # must contain train/ and val/
    img_size: int = 224
    crop_pct: float = 0.875            # 224/256 default convention
    interpolation: str = "bicubic"      # bilinear | bicubic
    batch_size: int = 128
    num_workers: int = 8
    pin_memory: bool = True

    # Augmentation toggles (baseline-friendly; MixPro mixing happens in the train step)
    use_randaugment: bool = True
    ra_num_ops: int = 9
    ra_magnitude: int = 15
    random_erasing_prob: float = 0.0    # set >0 if you want

    # Determinism (optional)
    seed: int = 42


def _interp(mode: str) -> InterpolationMode:
    m = mode.lower().strip()
    if m == "bilinear":
        return InterpolationMode.BILINEAR
    if m == "bicubic":
        return InterpolationMode.BICUBIC
    raise ValueError(f"Unsupported interpolation: {mode} (use bilinear|bicubic)")


def build_transforms(cfg: ImageNetDataConfig):
    interp = _interp(cfg.interpolation)
    img_size = cfg.img_size

    train_t = [
        transforms.RandomResizedCrop(img_size, interpolation=interp),
        transforms.RandomHorizontalFlip(p=0.5),
    ]

    # RandAugment is a good default for ViTs; paper recipe mentions RandAug(9, 0.5).
    # torchvision RandAugment uses (num_ops, magnitude) but not "prob" directly.
    if cfg.use_randaugment:
        train_t.append(transforms.RandAugment(num_ops=cfg.ra_num_ops, magnitude=cfg.ra_magnitude))

    train_t += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]

    if cfg.random_erasing_prob and cfg.random_erasing_prob > 0:
        train_t.append(transforms.RandomErasing(p=cfg.random_erasing_prob))

    train_transform = transforms.Compose(train_t)

    # Eval: Resize -> CenterCrop (canonical ImageNet evaluation)
    resize_size = int(round(img_size / cfg.crop_pct))
    eval_transform = transforms.Compose([
        transforms.Resize(resize_size, interpolation=interp),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, eval_transform


def build_imagenet_loaders(
    cfg: ImageNetDataConfig,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, DataLoader, Optional[torch.utils.data.Sampler], Optional[torch.utils.data.Sampler]]:
    """
    Returns: train_loader, val_loader, train_sampler, val_sampler
    Notes:
      - MixUp/CutMix/MaskMix happen in your training step (not in dataset transforms).
      - This expects folder structure:
            data_dir/train/<wnid>/*.JPEG
            data_dir/val/<wnid>/*.JPEG
    """
    root = Path(cfg.data_dir)
    train_dir = root / "train"
    val_dir = root / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Expected {train_dir} and {val_dir} to exist. "
            f"Did you run scripts/prepare_imagenet.py?"
        )

    train_tf, eval_tf = build_transforms(cfg)
    train_set = ImageFolder(train_dir, transform=train_tf)
    val_set = ImageFolder(val_dir, transform=eval_tf)

    train_sampler = None
    val_sampler = None
    shuffle = True

    if distributed:
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.seed)
        val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False, seed=cfg.seed)
        shuffle = False  # sampler handles shuffling

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, train_sampler, val_sampler
