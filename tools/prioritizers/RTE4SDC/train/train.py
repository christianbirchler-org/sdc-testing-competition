"""Training and evaluation pipeline for the TransformerEncoder model.

Includes the pairwise ranking loss, per-epoch training and evaluation loops,
learning rate scheduling, early stopping, and checkpoint management.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SensoDat, collate_fn, load_samples, split_samples
from model import TransformerEncoder
from normalizer import Normalizer
from utils import (
    classification_metrics,
    ensure_dir,
    load_yaml,
    ranking_metrics,
    resolve_device,
    set_seed,
)


# ---------------------------------------------------------------------------
# Metrics and output containers
# ---------------------------------------------------------------------------

class Metrics(dict):
    """Dictionary subclass for evaluation metrics (apfd, apfdc, f1, auc, ...)."""
    pass


@dataclass
class TrainOutput:
    """Result of a single training or evaluation epoch."""
    step: int
    loss: float
    metrics: Metrics


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def pairwise_ranking_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Pairwise ranking loss: encourages fail scores > pass scores.

    For every (fail, pass) pair, computes softplus(-(s_fail - s_pass)).
    Returns zero if no pairs exist (all fail or all pass).
    """
    fail_mask = labels > 0.5
    pass_mask = ~fail_mask

    if fail_mask.sum() == 0 or pass_mask.sum() == 0:
        return torch.zeros((), device=scores.device, dtype=scores.dtype)

    s_fail = scores[fail_mask]
    s_pass = scores[pass_mask]
    diff = s_fail.unsqueeze(1) - s_pass.unsqueeze(0)
    return F.softplus(-diff).mean()


# ---------------------------------------------------------------------------
# Training and evaluation loops
# ---------------------------------------------------------------------------

def move_batch(batch: dict, device: torch.device) -> dict:
    """Transfer a collated batch to the target device."""
    return {
        "seq": batch["seq"].to(device),
        "valid_mask": batch["valid_mask"].to(device),
        "glob": batch["glob"].to(device),
        "labels": batch["labels"].to(device),
        "durations": batch["durations"].to(device),
        "test_ids": batch["test_ids"],
    }


def compute_epoch_metrics(all_logits: list, all_labels: list, all_durations: list) -> Metrics:
    """Aggregate per-batch predictions into epoch-level metrics."""
    logits_np = np.concatenate(all_logits, axis=0)
    labels_np = np.concatenate(all_labels, axis=0).astype(np.int64)
    durations_np = np.concatenate(all_durations, axis=0)

    metrics = Metrics()
    metrics.update(classification_metrics(logits_np, labels_np))
    metrics.update(ranking_metrics(logits_np, labels_np, durations_np))
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    bce_criterion: nn.Module,
    classification_loss_weight: float,
    ranking_loss_weight: float,
    grad_clip_norm: float,
    device: torch.device,
    epoch: int,
) -> TrainOutput:
    """Run one training epoch with gradient updates."""
    model.train()
    total_loss = 0.0
    total_count = 0
    all_logits, all_labels, all_durations = [], [], []

    for batch in tqdm(loader, desc=f"train epoch {epoch}", leave=False):
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        logits = model(batch["seq"], batch["valid_mask"], batch["glob"])
        cls_loss = bce_criterion(logits, batch["labels"])
        rank_loss = pairwise_ranking_loss(logits, batch["labels"])
        loss = classification_loss_weight * cls_loss + ranking_loss_weight * rank_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        bs = batch["labels"].shape[0]
        total_count += bs
        total_loss += float(loss.item()) * bs

        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(batch["labels"].detach().cpu().numpy())
        all_durations.append(batch["durations"].detach().cpu().numpy())

    metrics = compute_epoch_metrics(all_logits, all_labels, all_durations)
    return TrainOutput(step=epoch, loss=total_loss / total_count, metrics=metrics)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    bce_criterion: nn.Module,
    classification_loss_weight: float,
    ranking_loss_weight: float,
    device: torch.device,
    epoch: int = 0,
) -> TrainOutput:
    """Evaluate the model without gradient computation."""
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_logits, all_labels, all_durations = [], [], []

    for batch in tqdm(loader, desc="eval", leave=False):
        batch = move_batch(batch, device)

        logits = model(batch["seq"], batch["valid_mask"], batch["glob"])
        cls_loss = bce_criterion(logits, batch["labels"])
        rank_loss = pairwise_ranking_loss(logits, batch["labels"])
        loss = classification_loss_weight * cls_loss + ranking_loss_weight * rank_loss

        bs = batch["labels"].shape[0]
        total_count += bs
        total_loss += float(loss.item()) * bs

        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(batch["labels"].detach().cpu().numpy())
        all_durations.append(batch["durations"].detach().cpu().numpy())

    metrics = compute_epoch_metrics(all_logits, all_labels, all_durations)
    return TrainOutput(step=epoch, loss=total_loss / total_count, metrics=metrics)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the TransformerEncoder model.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from latest.pt")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    set_seed(int(cfg["seed"]))
    device = resolve_device(str(cfg["device"]))
    print(f"Using device: {device}")

    # Load and split data
    samples = load_samples(
        dataset_path=cfg["data"]["dataset_path"],
        min_points=int(cfg["data"]["min_points"]),
    )
    if len(samples) < 20:
        raise RuntimeError("Not enough samples loaded. Check dataset path / content.")

    train_samples, val_samples, test_samples = split_samples(
        samples=samples,
        val_fraction=float(cfg["data"]["val_fraction"]),
        test_fraction=float(cfg["data"]["test_fraction"]),
        seed=int(cfg["seed"]),
    )
    normalizer = Normalizer.fit(train_samples)

    # Create data loaders
    train_loader = DataLoader(
        SensoDat(train_samples, normalizer),
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        SensoDat(val_samples, normalizer),
        batch_size=int(cfg["eval"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        SensoDat(test_samples, normalizer),
        batch_size=int(cfg["eval"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        collate_fn=collate_fn,
    )

    # Model, optimizer, loss
    model = TransformerEncoder(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    bce_criterion = nn.BCEWithLogitsLoss()
    classification_loss_weight = float(cfg["train"]["classification_loss_weight"])
    ranking_loss_weight = float(cfg["train"]["ranking_loss_weight"])
    grad_clip_norm = float(cfg["train"]["grad_clip_norm"])
    epochs = int(cfg["train"]["epochs"])
    early_stopping_patience = int(cfg["train"]["early_stopping_patience"])

    # Learning rate scheduler
    scheduler_name = str(cfg["train"].get("lr_scheduler", "none"))
    if scheduler_name == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
    else:
        scheduler = None

    # Output paths
    out_dir = ensure_dir(cfg["output"]["dir"])
    ckpt_path = out_dir / str(cfg["output"]["checkpoint_name"])
    latest_path = out_dir / "latest.pt"
    history_path = out_dir / "history.json"

    # Training loop state
    best_apfd = -1.0
    bad_epochs = 0
    start_epoch = 1
    history: list[dict] = []

    # Resume from latest checkpoint
    if args.resume and latest_path.exists():
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler is not None and "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_apfd = ckpt["best_apfd"]
        bad_epochs = ckpt["bad_epochs"]
        history = ckpt["history"]
        print(f"Resumed from epoch {ckpt['epoch']}, best_apfd={best_apfd:.4f}")

    for epoch in range(start_epoch, epochs + 1):
        tr = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer,
            bce_criterion=bce_criterion,
            classification_loss_weight=classification_loss_weight,
            ranking_loss_weight=ranking_loss_weight,
            grad_clip_norm=grad_clip_norm, device=device, epoch=epoch,
        )
        va = evaluate(
            model=model, loader=val_loader, bce_criterion=bce_criterion,
            classification_loss_weight=classification_loss_weight,
            ranking_loss_weight=ranking_loss_weight, device=device, epoch=epoch,
        )

        row = {
            "epoch": epoch,
            "train_loss": tr.loss,
            "train_apfd": tr.metrics["apfd"],
            "train_apfdc": tr.metrics["apfdc"],
            "val_loss": va.loss,
            "val_apfd": va.metrics["apfd"],
            "val_apfdc": va.metrics["apfdc"],
            "val_accuracy": va.metrics["accuracy"],
            "val_precision": va.metrics["precision"],
            "val_recall": va.metrics["recall"],
            "val_f1": va.metrics["f1"],
            "val_auc": va.metrics["auc"],
        }
        history.append(row)
        print(row)

        # Step the scheduler
        if scheduler is not None:
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                scheduler.step(va.metrics["apfd"])
            else:
                scheduler.step()

        # Checkpoint on best validation APFD
        if va.metrics["apfd"] > best_apfd:
            best_apfd = va.metrics["apfd"]
            bad_epochs = 0
            torch.save({
                "model_state": model.state_dict(),
                "config": cfg,
                "normalizer": normalizer.to_dict(),
                "epoch": epoch,
                "val_metrics": dict(va.metrics),
            }, ckpt_path)
        else:
            bad_epochs += 1

        # Save latest checkpoint for resume
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_apfd": best_apfd,
            "bad_epochs": bad_epochs,
            "history": history,
            "config": cfg,
            "normalizer": normalizer.to_dict(),
        }, latest_path)

        if bad_epochs >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Save training history
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Final evaluation on test set using best checkpoint
    if test_samples:
        best = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(best["model_state"])
        te = evaluate(
            model=model, loader=test_loader, bce_criterion=bce_criterion,
            classification_loss_weight=classification_loss_weight,
            ranking_loss_weight=ranking_loss_weight, device=device,
        )
        summary = {
            "checkpoint": str(ckpt_path),
            "test_metrics": dict(te.metrics),
            "device": str(device),
            "dataset_path": cfg["data"]["dataset_path"],
        }
        print(summary)
        with (out_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    else:
        print("No test set configured, skipping final evaluation.")


if __name__ == "__main__":
    main()
