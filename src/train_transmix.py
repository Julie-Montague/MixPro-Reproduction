from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import yaml

from timm.optim import create_optimizer_v2
from timm.scheduler import CosineLRScheduler

from src.data.imagenet import ImageNetDataConfig, build_imagenet_loaders
from src.models.deit_s import (
    build_deit_s,
    enable_last_attn_capture,
    get_cls_to_patch_attention,
    get_patch_grid_and_size,
)
from src.models.deit_t import build_deit_t
from src.methods.pal import compute_lambda_attn  # reuse your utility: lambda_attn = f(attn, mask_patch)
from src.utils.train_utils import set_seed, accuracy


ALL_RUNS_FIELDS = [
    "run_name",
    "method",
    "seed",
    "best_top1",
    "best_epoch",
    "epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "transmix_prob",
    "cutmix_alpha",
    "lambda_mode",
    "model_name",
    "img_size",
]

HIST_FIELDS = [
    "epoch",
    "train_loss",
    "train_top1",
    "mix_apply_rate",
    "lambda_area_mean",
    "lambda_attn_mean",
    "lambda_final_mean",
    "val_loss",
    "val_top1",
    "val_top5",
    "lr",
]


# ------------------------- DDP helpers -------------------------
def ddp_is_on() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def ddp_setup():
    """
    Returns: (ddp, rank, world_size, local_rank, device)

    Notes:
    - For torchrun, LOCAL_RANK indexes *within* CUDA_VISIBLE_DEVICES.
    """
    if ddp_is_on():
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        return True, rank, world_size, local_rank, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return False, 0, 1, 0, device


def ddp_cleanup():
    if ddp_is_on():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    if ddp_is_on():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


# ------------------------- IO helpers -------------------------
def load_cfg(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: Path, obj: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def append_csv_row(path: Path, fieldnames: list[str], row: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    safe_row = {k: row.get(k, "") for k in fieldnames}
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(safe_row)


def append_jsonl(path: Path, obj: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, DDP):
        return model.module
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model


# ------------------------- losses / utils -------------------------
def soft_ce_loss(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    # logits: [B,C], soft_targets: [B,C]
    logp = torch.log_softmax(logits, dim=1)
    return -(soft_targets * logp).sum(dim=1).mean()


def one_hot(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(y, num_classes=num_classes).float()


def rand_bbox(H: int, W: int, lam_keep: float, device: torch.device) -> tuple[int, int, int, int]:
    """
    CutMix-style bbox.
    lam_keep ~ fraction of ORIGINAL image kept (typical CutMix variable).
    Patch area ~ (1 - lam_keep).
    """
    cut_rat = float((1.0 - lam_keep) ** 0.5)
    cut_w = max(1, int(W * cut_rat))
    cut_h = max(1, int(H * cut_rat))

    cx = int(torch.randint(0, W, (1,), device=device).item())
    cy = int(torch.randint(0, H, (1,), device=device).item())

    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, H)
    return x1, x2, y1, y2


def apply_cutmix_make_maskpatch(
    x: torch.Tensor,
    x2: torch.Tensor,
    alpha: float,
    grid_h: int,
    grid_w: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      x_mix:    [B,C,H,W]
      mask_patch: [B, P] (P=grid_h*grid_w), 1 where pixels come from x2
      lambda_area: [B] fraction of pixels from x2 (same for all samples here)
    """
    B, C, H, W = x.shape
    device = x.device

    # sample CutMix keep-ratio
    lam_keep = float(torch.distributions.Beta(alpha, alpha).sample().item())
    x1, x2b, y1, y2b = rand_bbox(H, W, lam_keep, device)

    x_mix = x.clone()
    x_mix[:, :, y1:y2b, x1:x2b] = x2[:, :, y1:y2b, x1:x2b]

    # pixel mask: 1 in the pasted region (from x2)
    mask = torch.zeros((1, 1, H, W), device=device, dtype=torch.float32)
    mask[:, :, y1:y2b, x1:x2b] = 1.0
    mask = mask.expand(B, -1, -1, -1)  # [B,1,H,W]

    lambda_area = mask.mean(dim=(1, 2, 3))  # [B], fraction from x2 (identical across batch)

    # downsample to ViT patch grid, nearest neighbor to preserve binary region
    mask_small = torch.nn.functional.interpolate(mask, size=(grid_h, grid_w), mode="nearest")  # [B,1,gh,gw]
    mask_patch = mask_small.view(B, -1)  # [B,P]
    return x_mix, mask_patch, lambda_area


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device, rank: int) -> dict[str, float]:
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    loss_sum = 0.0
    top1_sum = 0.0
    top5_sum = 0.0
    n = 0

    iterator = loader
    if is_main(rank):
        iterator = tqdm(loader, desc="val", leave=False)

    for x, y in iterator:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss_sum += float(criterion(logits, y).item())

        acc1, acc5 = accuracy(logits, y, topk=(1, 5))  # percent
        bs = x.size(0)
        top1_sum += float(acc1.item()) * bs
        top5_sum += float(acc5.item()) * bs
        n += bs

    t = torch.tensor([loss_sum, top1_sum, top5_sum, float(n)], device=device, dtype=torch.float64)
    t = all_reduce_sum(t)
    loss_sum, top1_sum, top5_sum, n = t.tolist()
    n = max(1.0, n)

    return {
        "val_loss": float(loss_sum / n),
        "val_top1": float(top1_sum / n),
        "val_top5": float(top5_sum / n),
    }


# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--out", type=str, default="results")
    args = ap.parse_args()

    ddp, rank, world_size, local_rank, device = ddp_setup()

    try:
        cfg = load_cfg(args.config)

        out_dir = Path(args.out)
        run_name = str(cfg["run"]["name"])
        run_dir = out_dir / run_name

        if is_main(rank):
            run_dir.mkdir(parents=True, exist_ok=True)
            save_json(run_dir / "config_resolved.json", cfg)
        if ddp:
            dist.barrier()

        history_csv = run_dir / "history.csv"
        history_jsonl = run_dir / "history.jsonl"

        seed = int(cfg["run"].get("seed", 42))
        set_seed(seed + rank)

        # ------------------ DATA ------------------
        data_dict = dict(cfg["data"])
        global_bs = int(data_dict.get("batch_size", 0))
        if ddp:
            if global_bs <= 0:
                raise ValueError("cfg.data.batch_size must be a positive integer (GLOBAL batch).")
            per_gpu = global_bs // world_size
            if per_gpu < 1:
                raise ValueError(f"Global batch_size={global_bs} too small for world_size={world_size}.")
            if global_bs % world_size != 0 and is_main(rank):
                print(f"[WARN] global batch_size {global_bs} not divisible by world_size {world_size}; using per_gpu={per_gpu}")
            data_dict["batch_size"] = per_gpu

        data_cfg = ImageNetDataConfig(**data_dict)
        train_loader, val_loader, _, _ = build_imagenet_loaders(data_cfg, distributed=ddp)

        # ------------------ MODEL ------------------
        model_name = str(cfg["model"].get("name", "deit_s")).lower().strip()
        model_builder = {"deit_s": build_deit_s, "deit_t": build_deit_t}.get(model_name)
        if model_builder is None:
            raise ValueError(f"Unknown model name: {model_name}. Expected one of: deit_s, deit_t")

        model = model_builder(
            num_classes=int(cfg["model"]["num_classes"]),
            img_size=int(cfg["data"]["img_size"]),
            drop_path_rate=float(cfg["model"].get("drop_path_rate", 0.1)),
            pretrained=bool(cfg["model"].get("pretrained", False)),
        ).to(device)

        if ddp:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
            if is_main(rank):
                 print("ddp=", ddp, "world_size=", world_size, "cuda_count=", torch.cuda.device_count(),
                    "local_rank=", local_rank, "device=", device)
        elif torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = torch.nn.DataParallel(model)

        base = unwrap_model(model)
        enable_last_attn_capture(base)
        grid_h, grid_w, _patch_size_px = get_patch_grid_and_size(base)

        # ------------------ OPTIMIZER + SCHEDULER ------------------
        opt_cfg = cfg["optim"]
        optimizer = create_optimizer_v2(
            model,
            opt=str(opt_cfg.get("name", "adamw")),
            lr=float(opt_cfg.get("lr", 1e-3)),
            weight_decay=float(opt_cfg.get("weight_decay", 0.05)),
        )

        num_epochs = int(cfg["train"]["epochs"])
        updates_per_epoch = len(train_loader)
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs * updates_per_epoch,
            lr_min=float(opt_cfg.get("lr_min", 1e-5)),
            warmup_t=int(cfg["train"].get("warmup_epochs", 20)) * updates_per_epoch,
            warmup_lr_init=float(opt_cfg.get("warmup_lr", 1e-6)),
            t_in_epochs=False,
        )

        use_amp = bool(cfg["train"].get("amp", True))
        scaler = GradScaler(enabled=use_amp)

        grad_accum = int(cfg["train"].get("grad_accum_steps", 1))
        if grad_accum < 1:
            grad_accum = 1

        # ------------------ TRANSMIX CONFIG ------------------
        # Prefer cfg.transmix, but fall back to cfg.maskmix (treat beta as alpha) to minimize YAML edits.
        tm = cfg.get("transmix", None)
        if tm is None:
            tm = cfg.get("maskmix", {})  # fallback
        transmix_prob = float(tm.get("prob", 1.0))
        cutmix_alpha = float(tm.get("alpha", tm.get("beta", 1.0)))
        lambda_mode = str(tm.get("lambda_mode", tm.get("lam_mode", "attn"))).lower()  # "attn" or "mean"

        # ------------------ RESUME ------------------
        start_epoch = 0
        best_top1 = -1.0
        best_epoch = -1
        global_step = 0

        resume_path = cfg["run"].get("resume", None)
        if resume_path:
            ckpt_path = Path(resume_path)
            if ckpt_path.exists():
                if is_main(rank):
                    print(f"Resuming from: {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location="cpu")
                unwrap_model(model).load_state_dict(ckpt["model"], strict=True)
                if "optimizer" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer"])
                if "scaler" in ckpt and use_amp and ckpt["scaler"] is not None:
                    scaler.load_state_dict(ckpt["scaler"])
                start_epoch = int(ckpt.get("epoch", 0))
                best_top1 = float(ckpt.get("best_top1", best_top1))
                best_epoch = int(ckpt.get("best_epoch", best_epoch))
                global_step = int(ckpt.get("global_step", global_step))
            else:
                if is_main(rank):
                    print(f"[WARN] resume path does not exist: {ckpt_path}")

        # ------------------ TRAIN ------------------
        ce_criterion = torch.nn.CrossEntropyLoss()
        eval_every = int(cfg["train"].get("eval_every", 1))
        num_classes = int(cfg["model"]["num_classes"])

        for epoch in range(start_epoch, num_epochs):
            if ddp and hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            model.train()

            train_loss_sum = 0.0
            train_top1_sum = 0.0
            train_n = 0

            # TransMix diagnostics
            lam_area_sum = 0.0
            lam_attn_sum = 0.0
            lam_final_sum = 0.0
            lam_n = 0
            mix_batches = 0
            total_batches = 0

            iterator = train_loader
            if is_main(rank):
                iterator = tqdm(train_loader, desc=f"train epoch {epoch+1}/{num_epochs}")

            optimizer.zero_grad(set_to_none=True)

            for step, (x, y) in enumerate(iterator, start=1):
                total_batches += 1
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                # pair within batch
                x2 = x.flip(0)
                y2 = y.flip(0)

                do_mix = (torch.rand(1, device=device).item() < transmix_prob)

                if do_mix:
                    x_mix, mask_patch, lambda_area = apply_cutmix_make_maskpatch(
                        x, x2, alpha=cutmix_alpha, grid_h=grid_h, grid_w=grid_w
                    )

                    with autocast(enabled=use_amp):
                        logits = model(x_mix)
                        A = get_cls_to_patch_attention(unwrap_model(model), normalize=True)  # [B,P]
                        lambda_attn = compute_lambda_attn(A, mask_patch)  # [B]

                        if lambda_mode == "mean":
                            lambda_final = 0.5 * (lambda_attn + lambda_area)
                        else:
                            lambda_final = lambda_attn

                        y_soft = (1.0 - lambda_final).unsqueeze(1) * one_hot(y, num_classes) + \
                                 lambda_final.unsqueeze(1) * one_hot(y2, num_classes)

                        loss = soft_ce_loss(logits, y_soft)

                    bs = x.size(0)
                    lam_area_sum += float(lambda_area.detach().sum().item())
                    lam_attn_sum += float(lambda_attn.detach().sum().item())
                    lam_final_sum += float(lambda_final.detach().sum().item())
                    lam_n += bs
                    mix_batches += 1

                else:
                    with autocast(enabled=use_amp):
                        logits = model(x)
                        loss = ce_criterion(logits, y)

                # grad accumulation
                loss_scaled = loss / grad_accum
                scaler.scale(loss_scaled).backward()

                # sanity train top1 vs hard labels
                with torch.no_grad():
                    acc1 = accuracy(logits, y, topk=(1,))[0].item()

                bs = x.size(0)
                train_loss_sum += float(loss.item()) * bs
                train_top1_sum += float(acc1) * bs
                train_n += bs

                do_step = (step % grad_accum == 0) or (step == len(train_loader))
                if do_step:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                    scheduler.step_update(global_step)
                    global_step += 1

                if is_main(rank):
                    lr_now = optimizer.param_groups[0]["lr"]
                    avg_loss = train_loss_sum / max(1, train_n)
                    avg_top1 = train_top1_sum / max(1, train_n)
                    iterator.set_postfix(loss=f"{avg_loss:.4f}", top1=f"{avg_top1:.2f}", lr=f"{lr_now:.2e}")

            # reduce train stats
            t_tr = torch.tensor([train_loss_sum, train_top1_sum, float(train_n)], device=device, dtype=torch.float64)
            t_tr = all_reduce_sum(t_tr)
            train_loss_sum, train_top1_sum, train_n = t_tr.tolist()
            train_n = max(1.0, train_n)
            train_loss = float(train_loss_sum / train_n)
            train_top1 = float(train_top1_sum / train_n)

            # reduce transmix diags
            t_mix = torch.tensor(
                [
                    lam_area_sum,
                    lam_attn_sum,
                    lam_final_sum,
                    float(lam_n),
                    float(mix_batches),
                    float(total_batches),
                ],
                device=device,
                dtype=torch.float64,
            )
            t_mix = all_reduce_sum(t_mix)
            lam_area_sum, lam_attn_sum, lam_final_sum, lam_n, mix_batches, total_batches = t_mix.tolist()

            lam_area_mean = float(lam_area_sum / max(1.0, lam_n))
            lam_attn_mean = float(lam_attn_sum / max(1.0, lam_n))
            lam_final_mean = float(lam_final_sum / max(1.0, lam_n))
            mix_apply_rate = float(mix_batches / max(1.0, total_batches))

            do_eval = ((epoch + 1) % eval_every == 0) or ((epoch + 1) == num_epochs)
            if do_eval:
                eval_metrics = evaluate(model, val_loader, device, rank=rank)
            else:
                eval_metrics = {"val_loss": float("nan"), "val_top1": float("nan"), "val_top5": float("nan")}

            if is_main(rank):
                if do_eval:
                    print(f"[epoch {epoch+1}] val_top1={eval_metrics['val_top1']:.2f} val_top5={eval_metrics['val_top5']:.2f}")
                else:
                    print(f"[epoch {epoch+1}] (no val; eval_every={eval_every})")

                row = {
                    "epoch": epoch + 1,
                    "train_loss": float(train_loss),
                    "train_top1": float(train_top1),
                    "mix_apply_rate": float(mix_apply_rate),
                    "lambda_area_mean": float(lam_area_mean),
                    "lambda_attn_mean": float(lam_attn_mean),
                    "lambda_final_mean": float(lam_final_mean),
                    "val_loss": float(eval_metrics["val_loss"]),
                    "val_top1": float(eval_metrics["val_top1"]),
                    "val_top5": float(eval_metrics["val_top5"]),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                }
                append_jsonl(history_jsonl, row)
                append_csv_row(history_csv, HIST_FIELDS, row)

                # checkpoints
                model_state = unwrap_model(model).state_dict()
                torch.save(
                    {
                        "model": model_state,
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict() if use_amp else None,
                        "epoch": epoch + 1,
                        "best_top1": best_top1,
                        "best_epoch": best_epoch,
                        "global_step": global_step,
                        "config": cfg,
                    },
                    run_dir / "last.pt",
                )

                if do_eval and eval_metrics["val_top1"] > best_top1:
                    best_top1 = float(eval_metrics["val_top1"])
                    best_epoch = int(epoch + 1)
                    torch.save(
                        {
                            "model": model_state,
                            "optimizer": optimizer.state_dict(),
                            "scaler": scaler.state_dict() if use_amp else None,
                            "epoch": epoch + 1,
                            "best_top1": best_top1,
                            "best_epoch": best_epoch,
                            "global_step": global_step,
                            "config": cfg,
                        },
                        run_dir / "best.pt",
                    )

                save_json(
                    run_dir / "summary.json",
                    {
                        "run_name": run_name,
                        "method": "transmix",
                        "seed": seed,
                        "best_top1": best_top1,
                        "best_epoch": best_epoch,
                        "last_epoch": epoch + 1,
                        "val": eval_metrics,
                        "train_last": {
                            "loss": float(train_loss),
                            "top1": float(train_top1),
                            "mix_apply_rate": float(mix_apply_rate),
                            "lambda_area_mean": float(lam_area_mean),
                            "lambda_attn_mean": float(lam_attn_mean),
                            "lambda_final_mean": float(lam_final_mean),
                        },
                        "transmix": {
                            "prob": float(transmix_prob),
                            "cutmix_alpha": float(cutmix_alpha),
                            "lambda_mode": str(lambda_mode),
                        },
                    },
                )

        if is_main(rank):
            all_runs_csv = out_dir / "tables" / "all_runs.csv"
            run_row = {
                "run_name": run_name,
                "method": "transmix",
                "seed": seed,
                "best_top1": best_top1,
                "best_epoch": best_epoch,
                "epochs": int(cfg["train"]["epochs"]),
                "batch_size": int(cfg["data"].get("batch_size", 0)),  # GLOBAL in config
                "lr": float(cfg["optim"].get("lr", 0.0)),
                "weight_decay": float(cfg["optim"].get("weight_decay", 0.0)),
                "transmix_prob": float(transmix_prob),
                "cutmix_alpha": float(cutmix_alpha),
                "lambda_mode": str(lambda_mode),
                "model_name": model_name,
                "img_size": int(cfg["data"]["img_size"]),
            }
            append_csv_row(all_runs_csv, ALL_RUNS_FIELDS, run_row)

            print(f"\nDone. Best top1={best_top1:.2f} at epoch {best_epoch}. Outputs in: {run_dir}")

    finally:
        ddp_cleanup()


if __name__ == "__main__":
    main()
