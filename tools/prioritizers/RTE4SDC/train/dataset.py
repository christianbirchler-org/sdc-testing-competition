"""Data loading, preprocessing, and PyTorch Dataset for SensoDat test cases.

Handles JSON parsing (both competition format and SensoDat format),
feature extraction, train/val/test splitting, and batching with padding.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import json
from json import JSONDecodeError
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from features import extract_features
from normalizer import Normalizer


@dataclass
class Sample:
    """A single test case with pre-computed geometric features."""
    test_id: str
    segment_tokens: np.ndarray   # [T, 4]
    global_features: np.ndarray  # [4]
    has_failed: float            # 1.0 = fail, 0.0 = pass
    duration: float              # simulation duration in seconds


def extract_label(item: dict) -> float | None:
    """Extract binary label from a raw JSON test case entry.

    Supports multiple JSON schemas: competition format (meta_data.test_info),
    flat format (test_outcome, has_failed, hasFailed, label).
    """
    # Competition format: nested under meta_data.test_info
    meta = item.get("meta_data", {})
    test_info = meta.get("test_info", {})
    if "test_outcome" in test_info:
        outcome = str(test_info["test_outcome"]).upper()
        if outcome == "FAIL":
            return 1.0
        if outcome == "PASS":
            return 0.0

    # Flat format
    if "test_outcome" in item:
        outcome = str(item["test_outcome"]).upper()
        if outcome == "FAIL":
            return 1.0
        if outcome == "PASS":
            return 0.0
    if "has_failed" in item:
        return 1.0 if bool(item["has_failed"]) else 0.0
    if "hasFailed" in item:
        return 1.0 if bool(item["hasFailed"]) else 0.0
    if "label" in item:
        value = float(item["label"])
        return 1.0 if value >= 0.5 else 0.0
    return None


def extract_duration(item: dict) -> float:
    """Extract test duration in seconds from a raw JSON entry."""
    # Competition format: nested under meta_data.test_info
    meta = item.get("meta_data", {})
    test_info = meta.get("test_info", {})
    if "test_duration" in test_info:
        return float(test_info["test_duration"])

    # Flat format
    for key in ("test_duration", "duration_seconds", "sim_time", "duration"):
        if key in item:
            return float(item[key])
    return 1.0


def extract_road_points(item: dict) -> list | None:
    """Extract road point list from a raw JSON entry."""
    for key in ("road_points", "roadPoints", "interpolated_points"):
        if key in item:
            return item[key]
    return None


def load_samples(dataset_path: str | Path, min_points: int = 3) -> list[Sample]:
    """Load test cases from a JSON file and extract geometric features.

    Supports both the competition dataset format (sdc-test-data.json) and
    flat JSON formats. Skips entries with missing labels or too few road points.
    """
    path = Path(dataset_path)
    with path.open("r", encoding="utf-8") as f:
        try:
            raw = json.load(f)
        except JSONDecodeError as exc:
            f.seek(0)
            first_line = f.readline().strip()
            if first_line.startswith("version https://git-lfs.github.com/spec/v1"):
                raise RuntimeError(
                    f"{path} is a Git LFS pointer. Run `git lfs pull` or download the dataset manually."
                ) from exc
            raise

    if isinstance(raw, dict):
        entries = raw.get("data", list(raw.values())) if "data" in raw else list(raw.values())
    else:
        entries = raw

    samples: list[Sample] = []
    for idx, item in enumerate(entries):
        if not isinstance(item, dict):
            continue

        road_points = extract_road_points(item)
        label = extract_label(item)
        if road_points is None or label is None:
            continue
        if len(road_points) < min_points:
            continue

        result = extract_features(road_points)
        if result is None:
            continue
        segment_tokens, global_features = result

        test_id = str(item.get("_id", {}).get("$oid", item.get("test_id", item.get("testId", idx))))
        duration = extract_duration(item)

        samples.append(Sample(
            test_id=test_id,
            segment_tokens=segment_tokens,
            global_features=global_features,
            has_failed=label,
            duration=duration,
        ))
    return samples


def split_samples(
    samples: list[Sample], val_fraction: float, test_fraction: float, seed: int
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Stratified split into train, validation, and test sets."""
    if not 0.0 <= val_fraction < 1.0 or not 0.0 <= test_fraction < 1.0:
        raise ValueError("val_fraction and test_fraction must be in [0, 1)")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1")

    y = [int(s.has_failed) for s in samples]

    if test_fraction == 0.0:
        train_val = list(samples)
        test = []
    else:
        train_val, test = train_test_split(
            samples, test_size=test_fraction, random_state=seed,
            stratify=y if len(set(y)) > 1 else None,
        )

    y_train_val = [int(s.has_failed) for s in train_val]
    val_size = val_fraction / (1.0 - test_fraction)
    train, val = train_test_split(
        train_val, test_size=val_size, random_state=seed,
        stratify=y_train_val if len(set(y_train_val)) > 1 else None,
    )
    return train, val, test


class SensoDat(Dataset):
    """PyTorch Dataset wrapping a list of Sample objects with optional normalization."""

    def __init__(self, samples: list[Sample], normalizer: Normalizer | None):
        self.samples = samples
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        segment_tokens = s.segment_tokens
        if self.normalizer is not None:
            seq, glob = self.normalizer.normalize(segment_tokens, s.global_features)
        else:
            seq, glob = segment_tokens, s.global_features
        return {
            "test_id": s.test_id,
            "seq": torch.tensor(seq, dtype=torch.float32),
            "glob": torch.tensor(glob, dtype=torch.float32),
            "label": torch.tensor(s.has_failed, dtype=torch.float32),
            "duration": torch.tensor(s.duration, dtype=torch.float32),
        }


def collate_fn(batch: list[dict]) -> dict:
    """Pad variable-length sequences in a batch and create attention masks."""
    batch_size = len(batch)
    max_len = max(item["seq"].shape[0] for item in batch)
    d_seq = batch[0]["seq"].shape[1]

    seq = torch.zeros((batch_size, max_len, d_seq), dtype=torch.float32)
    valid_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    glob = torch.stack([item["glob"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    durations = torch.stack([item["duration"] for item in batch], dim=0)
    test_ids = [item["test_id"] for item in batch]

    for i, item in enumerate(batch):
        t = item["seq"].shape[0]
        seq[i, :t, :] = item["seq"]
        valid_mask[i, :t] = True

    return {
        "seq": seq,
        "valid_mask": valid_mask,
        "glob": glob,
        "labels": labels,
        "durations": durations,
        "test_ids": test_ids,
    }
