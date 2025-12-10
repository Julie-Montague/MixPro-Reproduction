# src/train_mixpro_improved.py
from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
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
from src.methods.maskmix import make_random_mask_patch, apply_maskmix
from src.methods.pal import compute_lambda_attn, build_mixpro_labels
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
    "maskmix_prob",
    "maskmix_beta",
    "maskmix_scale",
    "model_name",
    "img_size",
    "alpha_mode",
    "alpha_ramp_epochs",
]

HIST_FIELDS = [
    "epoch",
    "train_loss",
    "train_top1",
    "mix_apply_rate",
    "mix_alpha_mean",
    "mix_lambda_final_mean",
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
      So if CUDA_VISIBLE_DEVICES=3 and nproc_per_node=1, LOCAL_RANK=0 maps to physical GPU 3.
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


# ------------------------- losses / eval -------------------------
def soft_ce_loss(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits, dim=1)
    return -(soft_targets * logp).sum(dim=1).mean()


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


def call_build_mixpro_labels(**kwargs):
    """
    Signature-safe wrapper for build_mixpro_labels.
    Lets you pass alpha_mode/progress without breaking older signatures.
    """
    sig = inspect.signature(build_mixpro_labels)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return build_mixpro_labels(**filtered)


# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--out", type=str, default="results")
    args = ap.parse_args()

    ddp, rank, world_size, local_rank, device = ddp_setup()

    try:
        cfg = load_cfg(args.config)

        # PAL improvement knobs (read from YAML)
        pal_cfg = cfg.get("pal", {})
        alpha_mode = str(pal_cfg.get("alpha_mode", "cosine")).lower().strip()
        alpha_ramp_epochs = int(pal_cfg.get("alpha_ramp_epochs", 0))

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

        if is_main(rank):
            print(
                f"[setup] ddp={ddp} world_size={world_size} local_rank={local_rank} "
                f"device={device} alpha_mode={alpha_mode} alpha_ramp_epochs={alpha_ramp_epochs}"
            )

        # ------------------ DATA ------------------
        # Treat cfg.data.batch_size as GLOBAL batch; divide per rank under DDP.
        data_dict = dict(cfg["data"])
        global_bs = int(data_dict.get("batch_size", 0))
        if ddp:
            if global_bs <= 0:
                raise ValueError("cfg.data.batch_size must be a positive integer (GLOBAL batch).")
            per_gpu = global_bs // world_size
            if per_gpu < 1:
                raise ValueError(f"Global batch_size={global_bs} too small for world_size={world_size}.")
            if global_bs % world_size != 0 and is_main(rank):
                print(
                    f"[WARN] global batch_size {global_bs} not divisible by world_size {world_size}; using per_gpu={per_gpu}"
                )
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

        # IMPORTANT: patch attention capture before wrapping with DDP
        enable_last_attn_capture(model)
        grid_h, grid_w, patch_size_px = get_patch_grid_and_size(model)

        if ddp:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
            if is_main(rank):
                print(f"Using DDP with world_size={world_size}")
        else:
            # Avoid DataParallel entirely (it can break attention monkey-patching).
            if torch.cuda.device_count() > 1 and is_main(rank):
                print(
                    "[WARN] Multiple GPUs visible but DDP is off (WORLD_SIZE=1). "
                    "Not using DataParallel. Launch with torchrun --nproc_per_node=N."
                )

        # ------------------ OPTIMIZER + SCHEDULER ------------------
        opt_cfg = cfg["optim"]
        optimizer = create_optimizer_v2(
            model,
            opt=str(opt_cfg.get("name", "adamw")),
            lr=float(opt_cfg.get("lr", 1e-3)),
            weight_decay=float(opt_cfg.get("weight_decay", 0.05)),
        )

        num_epochs = int(cfg["train"]["epochs"])
        grad_accum = int(cfg["train"].get("grad_accum_steps", 1))
        if grad_accum < 1:
            grad_accum = 1

        # Scheduler is stepped per *optimizer update* (after grad accumulation).
        steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs * steps_per_epoch,
            lr_min=float(opt_cfg.get("lr_min", 1e-5)),
            warmup_t=int(cfg["train"].get("warmup_epochs", 20)) * steps_per_epoch,
            warmup_lr_init=float(opt_cfg.get("warmup_lr", 1e-6)),
            t_in_epochs=False,
        )

        use_amp = bool(cfg["train"].get("amp", True))
        scaler = GradScaler(enabled=use_amp)

        # MaskMix params (your script reads from "maskmix")
        mm = cfg.get("maskmix", {})
        scale = int(mm.get("scale", 4))
        beta = float(mm.get("beta", 1.0))
        prob = float(mm.get("prob", 1.0))

        # ------------------ RESUME ------------------
        start_epoch = 0
        best_top1 = -1.0
        best_epoch = -1
        global_step = 0  # counts optimizer updates (scheduler steps)

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

        for epoch in range(start_epoch, num_epochs):
            if ddp and hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            # PAL ramp multiplier
            progress = 1.0 if alpha_ramp_epochs <= 0 else min(1.0, float(epoch + 1) / float(alpha_ramp_epochs))

            model.train()

            train_loss_sum = 0.0
            train_top1_sum = 0.0
            train_n = 0

            mix_alpha_sum = 0.0
            mix_alpha_n = 0
            mix_lam_sum = 0.0
            mix_lam_n = 0
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

                # Pairing strategy (cheap): reversed batch
                x2 = x.flip(0)
                y2 = y.flip(0)

                do_mm = (torch.rand(1, device=device).item() < prob)

                if do_mm:
                    mask_patch = make_random_mask_patch(
                        batch_size=x.size(0),
                        grid_h=grid_h,
                        grid_w=grid_w,
                        scale=scale,
                        beta=beta,
                        device=device,
                    )
                    x_mix, lambda_area = apply_maskmix(x, x2, mask_patch, patch_size_px=patch_size_px)

                    with autocast(enabled=use_amp):
                        logits = model(x_mix)

                        # attention -> lambda_attn
                        A = get_cls_to_patch_attention(unwrap_model(model), normalize=True)
                        lambda_attn = compute_lambda_attn(A, mask_patch)

                        y_soft, lambda_final, alpha = call_build_mixpro_labels(
                            logits=logits,
                            y_i=y,
                            y_j=y2,
                            num_classes=int(cfg["model"]["num_classes"]),
                            lambda_attn=lambda_attn,
                            lambda_area=lambda_area,
                            alpha_mode=alpha_mode,
                            progress=progress,
                        )

                        if y_soft.shape != logits.shape:
                            raise RuntimeError(f"MixPro label shape mismatch: y_soft={y_soft.shape} vs logits={logits.shape}")

                        loss = soft_ce_loss(logits, y_soft)

                    bs = x.size(0)
                    mix_alpha_sum += float(alpha.detach().sum().item())
                    mix_alpha_n += bs
                    mix_lam_sum += float(lambda_final.detach().sum().item())
                    mix_lam_n += bs
                    mix_batches += 1

                else:
                    with autocast(enabled=use_amp):
                        logits = model(x)
                        loss = ce_criterion(logits, y)

                # grad accumulation
                loss_scaled = loss / grad_accum
                scaler.scale(loss_scaled).backward()

                # sanity accuracy (for mixed batches: not strictly meaningful, but useful as a signal)
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

            # reduce mix diagnostics
            t_mix = torch.tensor(
                [mix_alpha_sum, float(mix_alpha_n), mix_lam_sum, float(mix_lam_n), float(mix_batches), float(total_batches)],
                device=device,
                dtype=torch.float64,
            )
            t_mix = all_reduce_sum(t_mix)
            mix_alpha_sum, mix_alpha_n, mix_lam_sum, mix_lam_n, mix_batches, total_batches = t_mix.tolist()

            alpha_mean = float(mix_alpha_sum / max(1.0, mix_alpha_n))
            lambda_final_mean = float(mix_lam_sum / max(1.0, mix_lam_n))
            mix_apply_rate = float(mix_batches / max(1.0, total_batches))

            do_eval = ((epoch + 1) % eval_every == 0) or ((epoch + 1) == num_epochs)
            if do_eval:
                eval_metrics = evaluate(model, val_loader, device, rank=rank)
            else:
                eval_metrics = {"val_loss": float("nan"), "val_top1": float("nan"), "val_top5": float("nan")}

            if is_main(rank):
                if do_eval:
                    print(
                        f"[epoch {epoch+1}] val_top1={eval_metrics['val_top1']:.2f} "
                        f"val_top5={eval_metrics['val_top5']:.2f} alpha_mean={alpha_mean:.3f} mix_rate={mix_apply_rate:.2f}"
                    )
                else:
                    print(f"[epoch {epoch+1}] (no val; eval_every={eval_every})")

                row = {
                    "epoch": epoch + 1,
                    "train_loss": float(train_loss),
                    "train_top1": float(train_top1),
                    "mix_apply_rate": float(mix_apply_rate),
                    "mix_alpha_mean": float(alpha_mean),
                    "mix_lambda_final_mean": float(lambda_final_mean),
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
                        "alpha_mode": alpha_mode,
                        "alpha_ramp_epochs": alpha_ramp_epochs,
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
                            "alpha_mode": alpha_mode,
                            "alpha_ramp_epochs": alpha_ramp_epochs,
                        },
                        run_dir / "best.pt",
                    )

                save_json(
                    run_dir / "summary.json",
                    {
                        "run_name": run_name,
                        "method": "mixpro_improved",
                        "alpha_mode": alpha_mode,
                        "alpha_ramp_epochs": alpha_ramp_epochs,
                        "seed": seed,
                        "best_top1": best_top1,
                        "best_epoch": best_epoch,
                        "last_epoch": epoch + 1,
                        "val": eval_metrics,
                        "train_last": {
                            "loss": float(train_loss),
                            "top1": float(train_top1),
                            "mix_apply_rate": float(mix_apply_rate),
                            "mix_alpha_mean": float(alpha_mean),
                            "mix_lambda_final_mean": float(lambda_final_mean),
                        },
                    },
                )

        if is_main(rank):
            all_runs_csv = out_dir / "tables" / "all_runs.csv"
            run_row = {
                "run_name": run_name,
                "method": "mixpro_improved",
                "seed": seed,
                "best_top1": best_top1,
                "best_epoch": best_epoch,
                "epochs": int(cfg["train"]["epochs"]),
                "batch_size": int(cfg["data"].get("batch_size", 0)),  # GLOBAL in config
                "lr": float(cfg["optim"].get("lr", 0.0)),
                "weight_decay": float(cfg["optim"].get("weight_decay", 0.0)),
                "maskmix_prob": float(prob),
                "maskmix_beta": float(beta),
                "maskmix_scale": int(scale),
                "model_name": model_name,
                "img_size": int(cfg["data"]["img_size"]),
                "alpha_mode": alpha_mode,
                "alpha_ramp_epochs": alpha_ramp_epochs,
            }
            append_csv_row(all_runs_csv, ALL_RUNS_FIELDS, run_row)

            print(f"\nDone. Best top1={best_top1:.2f} at epoch {best_epoch}. Outputs in: {run_dir}")

    finally:
        ddp_cleanup()


if __name__ == "__main__":
    main()
