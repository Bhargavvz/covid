"""
VoxelMorph registration training script.

Trains a VoxelMorph3D model to learn deformable image registration
between pairs of CT volumes.
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from src.registration.voxelmorph import VoxelMorph3D, RegistrationLoss
from src.preprocessing.dataset import create_dataloaders

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train VoxelMorph 3D registration")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/registration")
    parser.add_argument("--log_dir", type=str, default="./logs/registration")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--volume_size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--similarity", type=str, default="ncc", choices=["ncc", "mse"])
    parser.add_argument("--reg_weight", type=float, default=1.0, help="Regularization weight")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--ddp", action="store_true", default=False)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--use_synthetic", action="store_true")
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=0)
    return parser.parse_args()


def compute_metrics(moved, fixed):
    """Compute registration quality metrics."""
    with torch.no_grad():
        mse = torch.nn.functional.mse_loss(moved, fixed).item()
        # SSIM approximation (simplified)
        mu_moved = moved.mean()
        mu_fixed = fixed.mean()
        sigma_moved = moved.var()
        sigma_fixed = fixed.var()
        sigma_mf = ((moved - mu_moved) * (fixed - mu_fixed)).mean()
        c1 = (0.01 * 1.0) ** 2
        c2 = (0.03 * 1.0) ** 2
        ssim = (
            (2 * mu_moved * mu_fixed + c1) * (2 * sigma_mf + c2)
        ) / (
            (mu_moved**2 + mu_fixed**2 + c1) * (sigma_moved + sigma_fixed + c2)
        )
    return {"mse": mse, "ssim": ssim.item()}


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, args):
    model.train()
    total_loss = 0.0
    loss_components = {"similarity": 0, "regularization": 0}
    num_batches = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        moving = batch["moving"].to(device, non_blocking=True)
        fixed = batch["fixed"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
            moved, flow = model(moving, fixed)
            loss, components = criterion(moved, fixed, flow)
            loss = loss / args.gradient_accumulation

        if args.amp and device.type == "cuda":
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % args.gradient_accumulation == 0:
            if args.amp and device.type == "cuda":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        total_loss += components["total"]
        loss_components["similarity"] += components["similarity"]
        loss_components["regularization"] += components["regularization"]
        num_batches += 1

    n = max(num_batches, 1)
    return {
        "loss": total_loss / n,
        "similarity": loss_components["similarity"] / n,
        "regularization": loss_components["regularization"] / n,
    }


@torch.no_grad()
def validate(model, loader, criterion, device, args):
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_ssim = 0.0
    num_batches = 0

    for batch in loader:
        moving = batch["moving"].to(device, non_blocking=True)
        fixed = batch["fixed"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
            moved, flow = model(moving, fixed)
            loss, _ = criterion(moved, fixed, flow)

        metrics = compute_metrics(moved, fixed)
        total_loss += loss.item()
        total_mse += metrics["mse"]
        total_ssim += metrics["ssim"]
        num_batches += 1

    n = max(num_batches, 1)
    return {
        "loss": total_loss / n,
        "mse": total_mse / n,
        "ssim": total_ssim / n,
    }


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "metrics": metrics,
    }
    torch.save(state, path)
    logger.info(f"Checkpoint saved: {path}")


def main():
    args = parse_args()

    if args.ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f"cuda:{args.local_rank}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info(f"Device: {device}")

    volume_size = tuple(args.volume_size)
    model = VoxelMorph3D(volume_size=volume_size).to(device)

    if args.ddp:
        model = DDP(model, device_ids=[args.local_rank])

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"VoxelMorph parameters: {total_params:,}")

    train_loader, val_loader, _ = create_dataloaders(
        data_dir=args.data_dir,
        task="registration",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=volume_size,
        use_synthetic=args.use_synthetic,
        synthetic_samples=200,
    )

    criterion = RegistrationLoss(similarity=args.similarity, reg_weight=args.reg_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    writer = SummaryWriter(log_dir=args.log_dir)

    start_epoch = 0
    best_loss = float("inf")
    patience_counter = 0

    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        if hasattr(model, "module"):
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        logger.info(f"Resumed from epoch {start_epoch}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logger.info(f"Starting registration training for {args.epochs} epochs...")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, args)
        val_metrics = validate(model, val_loader, criterion, device, args)

        scheduler.step()
        elapsed = time.time() - t0

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} ({elapsed:.1f}s) | "
            f"Train Loss: {train_metrics['loss']:.4f} (sim: {train_metrics['similarity']:.4f}) | "
            f"Val Loss: {val_metrics['loss']:.4f}, MSE: {val_metrics['mse']:.6f}, SSIM: {val_metrics['ssim']:.4f}"
        )

        writer.add_scalars("loss", {"train": train_metrics["loss"], "val": val_metrics["loss"]}, epoch)
        writer.add_scalar("mse/val", val_metrics["mse"], epoch)
        writer.add_scalar("ssim/val", val_metrics["ssim"], epoch)

        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch,
                val_metrics, os.path.join(args.checkpoint_dir, "best_model.pt"),
            )
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch,
                val_metrics, os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt"),
            )

        if patience_counter >= args.patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    writer.close()
    logger.info(f"Training complete. Best loss: {best_loss:.4f}")

    if args.ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
