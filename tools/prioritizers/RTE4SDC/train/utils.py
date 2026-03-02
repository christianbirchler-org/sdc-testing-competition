"""Utility functions and evaluation metrics.

General-purpose helpers (config loading, seeding, device selection) and
domain-specific metrics (APFD, APFDc, classification metrics) used
during training and evaluation.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def load_yaml(path: str | Path) -> dict:
    """Load a YAML configuration file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(device_name: str) -> torch.device:
    """Resolve device string to a torch.device, with auto-detection."""
    if device_name != "auto":
        return torch.device(device_name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Ranking metrics (competition evaluation)
# ---------------------------------------------------------------------------

def apfd(sorted_labels: np.ndarray) -> float:
    """Average Percentage of Faults Detected.

    Measures how early faults appear in the prioritized ordering.
    Higher is better (1.0 = all faults ranked first).
    """
    n = len(sorted_labels)
    fail_positions = np.where(sorted_labels == 1)[0] + 1
    m = len(fail_positions)
    if n == 0 or m == 0:
        return 1.0
    return float(1.0 - (fail_positions.sum() / (n * m)) + (1.0 / (2.0 * n)))


def apfdc(sorted_labels: np.ndarray, sorted_durations: np.ndarray) -> float:
    """Cost-aware APFD, weighted by test execution duration.

    Penalizes placing long-running failing tests late in the ordering.
    """
    m = int(np.sum(sorted_labels))
    total_cost = float(np.sum(sorted_durations))
    if m == 0 or total_cost == 0.0:
        return 1.0
    cumulative = np.cumsum(sorted_durations)
    cfi = cumulative[sorted_labels == 1]
    return float(1.0 - (np.sum(cfi) / (total_cost * m)) + (1.0 / (2.0 * m)))


# ---------------------------------------------------------------------------
# Classification and ranking metric helpers
# ---------------------------------------------------------------------------

def classification_metrics(logits: np.ndarray, labels: np.ndarray) -> dict:
    """Compute accuracy, precision, recall, F1 and ROC AUC from raw logits."""
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(np.int64)
    out = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }
    if len(np.unique(labels)) > 1:
        out["auc"] = float(roc_auc_score(labels, probs))
    else:
        out["auc"] = 0.5
    return out


def ranking_metrics(scores: np.ndarray, labels: np.ndarray, durations: np.ndarray) -> dict:
    """Compute APFD and APFDc from model scores, sorting by descending score."""
    order = np.argsort(-scores)
    y_sorted = labels[order]
    d_sorted = durations[order]
    return {
        "apfd": apfd(y_sorted),
        "apfdc": apfdc(y_sorted, d_sorted),
    }
