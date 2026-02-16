"""
3D U-Net segmentation training script.

Supports:
- Mixed precision training (AMP)
- Distributed Data Parallel (DDP)
- Cosine annealing LR scheduler
- TensorBoard logging
- Early stopping
- Checkpoint saving
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

from src.segmentation.unet3d import UNet3D
from src.segmentation.metrics import DiceBCELoss, dice_score, iou_score
from src.preprocessing.dataset import create_dataloaders

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train 3D U-Net for lung segmentation")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/segmentation")
    parser.add_argument("--log_dir", type=str, default="./logs/segmentation")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--volume_size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--channels", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", default=True, help="Mixed precision")
    parser.add_argument("--ddp", action="store_true", default=False, help="Distributed training")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--use_synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=0)
    return parser.parse_args()


def setup_ddp(local_rank: int):
    """Initialize DDP process group."""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    args,
) -> dict:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    num_batches = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["label"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, masks) / args.gradient_accumulation

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

        running_loss += loss.item() * args.gradient_accumulation
        with torch.no_grad():
            running_dice += dice_score(outputs, masks).item()
        num_batches += 1

    return {
        "loss": running_loss / max(num_batches, 1),
        "dice": running_dice / max(num_batches, 1),
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    args,
) -> dict:
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    num_batches = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["label"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, masks)

        running_loss += loss.item()
        running_dice += dice_score(outputs, masks).item()
        running_iou += iou_score(outputs, masks).item()
        num_batches += 1

    return {
        "loss": running_loss / max(num_batches, 1),
        "dice": running_dice / max(num_batches, 1),
        "iou": running_iou / max(num_batches, 1),
    }


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, metrics, path):
    """Save training checkpoint."""
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

    # Setup device
    if args.ddp:
        setup_ddp(args.local_rank)
        device = torch.device(f"cuda:{args.local_rank}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info(f"Device: {device}")

    # Create model
    volume_size = tuple(args.volume_size)
    model = UNet3D(
        in_channels=1,
        out_channels=1,
        channels=tuple(args.channels),
        strides=tuple([2] * (len(args.channels) - 1)),
    ).to(device)

    if args.ddp:
        model = DDP(model, device_ids=[args.local_rank])

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # Create data loaders
    train_loader, val_loader, _ = create_dataloaders(
        data_dir=args.data_dir,
        task="segmentation",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=volume_size,
        use_synthetic=args.use_synthetic,
        synthetic_samples=200,
    )

    # Loss, optimizer, scheduler
    criterion = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    # TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)

    # Resume from checkpoint
    start_epoch = 0
    best_dice = 0.0
    patience_counter = 0

    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        if hasattr(model, "module"):
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_dice = checkpoint["metrics"].get("dice", 0.0)
        logger.info(f"Resumed from epoch {start_epoch}, best dice: {best_dice:.4f}")

    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs...")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch, args)
        val_metrics = validate(model, val_loader, criterion, device, args)

        scheduler.step()
        elapsed = time.time() - t0

        # Logging
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} ({elapsed:.1f}s) | "
            f"Train Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}"
        )

        writer.add_scalars("loss", {"train": train_metrics["loss"], "val": val_metrics["loss"]}, epoch)
        writer.add_scalars("dice", {"train": train_metrics["dice"], "val": val_metrics["dice"]}, epoch)
        writer.add_scalar("iou/val", val_metrics["iou"], epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        # Save best model
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch,
                val_metrics, os.path.join(args.checkpoint_dir, "best_model.pt"),
            )
        else:
            patience_counter += 1

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch,
                val_metrics, os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt"),
            )

        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f"Early stopping at epoch {epoch+1} (patience={args.patience})")
            break

    writer.close()
    logger.info(f"Training complete. Best Dice: {best_dice:.4f}")

    if args.ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
