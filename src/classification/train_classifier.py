"""
CNN classifier training script for Post-COVID severity classification.

Supports:
- Multi-task training (classification + regression)
- Mixed precision (AMP) on H200 GPU
- Distributed Data Parallel (DDP)
- Focal Loss for class imbalance
- Comprehensive evaluation metrics
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

from src.classification.resnet3d import ResNet3D, MultiTaskLoss, SEVERITY_LABELS
from src.classification.metrics import ClassificationMetrics
from src.preprocessing.dataset import create_dataloaders

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train 3D ResNet for Post-COVID classification")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/classifier")
    parser.add_argument("--log_dir", type=str, default="./logs/classifier")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--volume_size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--resnet_depth", type=int, default=50, choices=[18, 34, 50])
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--ddp", action="store_true", default=False)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--use_synthetic", action="store_true")
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    parser.add_argument("--cls_weight", type=float, default=1.0)
    parser.add_argument("--reg_weight", type=float, default=0.5)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--local_rank", type=int, default=0)
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, args):
    model.train()
    total_loss = 0.0
    loss_components = {"classification": 0, "regression": 0}
    metrics = ClassificationMetrics(args.num_classes, list(SEVERITY_LABELS.values()))
    num_batches = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        damage_targets = batch.get("severity_pct")
        if damage_targets is not None:
            damage_targets = damage_targets.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
            outputs = model(images)
            loss, comps = criterion(outputs, labels, damage_targets)
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

        total_loss += comps["total"]
        loss_components["classification"] += comps["classification"]
        loss_components["regression"] += comps["regression"]
        metrics.update(outputs["severity_logits"], labels)
        num_batches += 1

    n = max(num_batches, 1)
    metric_results = metrics.compute()

    return {
        "loss": total_loss / n,
        "cls_loss": loss_components["classification"] / n,
        "reg_loss": loss_components["regression"] / n,
        **metric_results,
    }


@torch.no_grad()
def validate(model, loader, criterion, device, args):
    model.eval()
    total_loss = 0.0
    metrics = ClassificationMetrics(args.num_classes, list(SEVERITY_LABELS.values()))
    num_batches = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        damage_targets = batch.get("severity_pct")
        if damage_targets is not None:
            damage_targets = damage_targets.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
            outputs = model(images)
            loss, _ = criterion(outputs, labels, damage_targets)

        total_loss += loss.item()
        metrics.update(outputs["severity_logits"], labels)
        num_batches += 1

    n = max(num_batches, 1)
    metric_results = metrics.compute()

    return {
        "loss": total_loss / n,
        **metric_results,
        "report": metrics.report(),
    }


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "metrics": {k: v for k, v in metrics.items() if k != "report"},
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
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

    volume_size = tuple(args.volume_size)
    model = ResNet3D(
        in_channels=1,
        num_classes=args.num_classes,
        depth=args.resnet_depth,
    ).to(device)

    if args.ddp:
        model = DDP(model, device_ids=[args.local_rank])

    params = sum(p.numel() for p in model.parameters())
    logger.info(f"ResNet3D-{args.resnet_depth} parameters: {params:,}")

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        task="classification",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=volume_size,
        use_synthetic=args.use_synthetic,
        synthetic_samples=400,
    )

    criterion = MultiTaskLoss(
        num_classes=args.num_classes,
        cls_weight=args.cls_weight,
        reg_weight=args.reg_weight,
        focal_gamma=args.focal_gamma,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    writer = SummaryWriter(log_dir=args.log_dir)

    start_epoch = 0
    best_f1 = 0.0
    patience_counter = 0

    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        if hasattr(model, "module"):
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_f1 = checkpoint["metrics"].get("f1", 0.0)
        logger.info(f"Resumed from epoch {start_epoch}, best F1: {best_f1:.4f}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logger.info(f"Starting classifier training for {args.epochs} epochs...")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, args)
        val_metrics = validate(model, val_loader, criterion, device, args)

        scheduler.step()
        elapsed = time.time() - t0

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} ({elapsed:.1f}s) | "
            f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc_roc']:.4f}"
        )

        # TensorBoard
        writer.add_scalars("loss", {"train": train_metrics["loss"], "val": val_metrics["loss"]}, epoch)
        writer.add_scalars("accuracy", {"train": train_metrics["accuracy"], "val": val_metrics["accuracy"]}, epoch)
        writer.add_scalars("f1", {"train": train_metrics["f1"], "val": val_metrics["f1"]}, epoch)
        writer.add_scalar("auc_roc/val", val_metrics["auc_roc"], epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch,
                val_metrics, os.path.join(args.checkpoint_dir, "best_model.pt"),
            )
            logger.info(f"New best F1: {best_f1:.4f}")
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch,
                val_metrics, os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt"),
            )
            logger.info(f"\n{val_metrics.get('report', '')}")

        if patience_counter >= args.patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # Final evaluation on test set
    logger.info("Evaluating on test set...")
    test_metrics = validate(model, test_loader, criterion, device, args)
    logger.info(
        f"Test Results — Acc: {test_metrics['accuracy']:.4f}, "
        f"F1: {test_metrics['f1']:.4f}, AUC: {test_metrics['auc_roc']:.4f}"
    )
    logger.info(f"\n{test_metrics.get('report', '')}")

    writer.close()
    logger.info(f"Training complete. Best F1: {best_f1:.4f}")

    if args.ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
