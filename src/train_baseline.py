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

from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from timm.optim import create_optimizer_v2
from timm.scheduler import CosineLRScheduler

from src.data.imagenet import ImageNetDataConfig, build_imagenet_loaders
from src.models.deit_s import build_deit_s
from src.models.deit_t import build_deit_t
from src.utils.train_utils import set_seed, AverageMeter, accuracy


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
    "mixup_alpha",
    "cutmix_alpha",
    "maskmix_prob",
    "maskmix_beta",
    "maskmix_scale",
    "model_name",
    "img_size",
]

HIST_FIELDS = ["epoch", "train_loss", "val_loss", "val_top1", "val_top5", "lr"]


def ddp_is_on() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def ddp_setup():
    """
    Returns: (ddp, rank, world_size, local_rank, device)
    Works for single GPU too (ddp=False).
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


def build_model_by_name(name: str, **kwargs):
    name = str(name).lower().strip()
    if name == "deit_s":
        return build_deit_s(**kwargs)
    if name == "deit_t":
        return build_deit_t(**kwargs)
    raise ValueError(f"Unsupported model name: {name}")


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
        loss = float(criterion(logits, y).item())

        acc1, acc5 = accuracy(logits, y, topk=(1, 5))  # percent
        bs = x.size(0)

        loss_sum += loss
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

        # Only create dirs / write config on rank 0
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
                raise ValueError("cfg.data.batch_size must be a positive integer.")
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
        model = build_model_by_name(
            name=model_name,
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
                print(f"Using DDP with world_size={world_size}")
        elif torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = torch.nn.DataParallel(model)

        # ------------------ MIXUP/CUTMIX ------------------
        mix = cfg.get("mix", {})
        mixup_fn = None
        if float(mix.get("mixup_alpha", 0.0)) > 0 or float(mix.get("cutmix_alpha", 0.0)) > 0:
            mixup_fn = Mixup(
                mixup_alpha=float(mix.get("mixup_alpha", 0.0)),
                cutmix_alpha=float(mix.get("cutmix_alpha", 0.0)),
                prob=float(mix.get("prob", 1.0)),
                switch_prob=float(mix.get("switch_prob", 0.5)),
                mode=str(mix.get("mode", "batch")),
                label_smoothing=float(cfg["train"].get("label_smoothing", 0.1)),
                num_classes=int(cfg["model"]["num_classes"]),
            )
            criterion = SoftTargetCrossEntropy()
        else:
            ls = float(cfg["train"].get("label_smoothing", 0.0))
            criterion = LabelSmoothingCrossEntropy(smoothing=ls) if ls > 0 else torch.nn.CrossEntropyLoss()

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

        # ------------------ AMP ------------------
        use_amp = bool(cfg["train"].get("amp", True))
        scaler = GradScaler(enabled=use_amp)

        # ------------------ TRAIN ------------------
        best_top1 = -1.0
        best_epoch = -1
        global_step = 0
        eval_every = int(cfg["train"].get("eval_every", 1))

        for epoch in range(num_epochs):
            if ddp and hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            model.train()

            # DDP-global train loss
            train_loss_sum = 0.0
            train_n = 0

            loss_meter = AverageMeter()
            top1_meter = AverageMeter()

            iterator = train_loader
            if is_main(rank):
                iterator = tqdm(train_loader, desc=f"train epoch {epoch+1}/{num_epochs}")

            for x, y in iterator:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                if mixup_fn is not None:
                    x, y = mixup_fn(x, y)

                with autocast(enabled=use_amp):
                    logits = model(x)
                    loss = criterion(logits, y)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                scheduler.step_update(global_step)
                global_step += 1

                bs = x.size(0)
                loss_val = float(loss.item())
                train_loss_sum += loss_val * bs
                train_n += bs

                loss_meter.update(loss_val, bs)

                if is_main(rank):
                    if mixup_fn is None:
                        acc1 = accuracy(logits.detach(), y, topk=(1,))[0].item()
                        top1_meter.update(acc1, bs)
                        iterator.set_postfix(
                            loss=f"{loss_meter.avg:.4f}",
                            top1=f"{top1_meter.avg:.2f}",
                            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                        )
                    else:
                        iterator.set_postfix(loss=f"{loss_meter.avg:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

            # reduce train loss across ranks
            t_tr = torch.tensor([train_loss_sum, float(train_n)], device=device, dtype=torch.float64)
            t_tr = all_reduce_sum(t_tr)
            train_loss_sum, train_n = t_tr.tolist()
            train_n = max(1.0, train_n)
            train_loss = float(train_loss_sum / train_n)

            do_eval = ((epoch + 1) % eval_every == 0) or ((epoch + 1) == num_epochs)
            if do_eval:
                eval_metrics = evaluate(model, val_loader, device, rank)  # ✅ FIXED
            else:
                eval_metrics = {"val_loss": float("nan"), "val_top1": float("nan"), "val_top5": float("nan")}

            if is_main(rank):
                row = {
                    "epoch": epoch + 1,
                    "train_loss": float(train_loss),
                    "val_loss": float(eval_metrics["val_loss"]),
                    "val_top1": float(eval_metrics["val_top1"]),
                    "val_top5": float(eval_metrics["val_top5"]),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                }
                append_jsonl(history_jsonl, row)
                append_csv_row(history_csv, fieldnames=HIST_FIELDS, row=row)

                model_state = unwrap_model(model).state_dict()
                torch.save(
                    {
                        "model": model_state,
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict() if use_amp else None,
                        "epoch": epoch + 1,
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
                            "config": cfg,
                        },
                        run_dir / "best.pt",
                    )

                if do_eval:
                    print(f"[epoch {epoch+1}] val_top1={eval_metrics['val_top1']:.2f} val_top5={eval_metrics['val_top5']:.2f}")

        # ------------------ SUMMARY + ALL RUNS ------------------
        if is_main(rank):
            summary = {
                "run_name": run_name,
                "method": "baseline",
                "seed": seed,
                "best_top1": best_top1,
                "best_epoch": best_epoch,
                "epochs": num_epochs,
                "batch_size": int(cfg["data"]["batch_size"]),  # GLOBAL as written
                "lr": float(cfg["optim"]["lr"]),
                "weight_decay": float(cfg["optim"]["weight_decay"]),
                "mixup_alpha": float(cfg.get("mix", {}).get("mixup_alpha", 0.0)),
                "cutmix_alpha": float(cfg.get("mix", {}).get("cutmix_alpha", 0.0)),
                "maskmix_prob": "",
                "maskmix_beta": "",
                "maskmix_scale": "",
                "model_name": model_name,
                "img_size": int(cfg["data"]["img_size"]),
            }
            save_json(run_dir / "summary.json", summary)
            append_csv_row(out_dir / "tables" / "all_runs.csv", fieldnames=ALL_RUNS_FIELDS, row=summary)
            print(f"\nDone. Best top1={best_top1:.2f} at epoch {best_epoch}. Outputs in: {run_dir}")

    finally:
        ddp_cleanup()


if __name__ == "__main__":
    main()
