"""Z-score normalizer for segment-level and global features.

Computes per-feature mean and standard deviation from the training set,
then applies standardization: z = (x - mean) / std. The normalizer state
is serializable so it can be saved inside model checkpoints and restored
at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Normalizer:
    seq_mean: np.ndarray   # [d_seq]
    seq_std: np.ndarray    # [d_seq]
    glob_mean: np.ndarray  # [d_global]
    glob_std: np.ndarray   # [d_global]

    @staticmethod
    def fit(samples) -> Normalizer:
        """Compute mean and std from a list of Sample objects."""
        all_tokens = np.concatenate([s.segment_tokens for s in samples], axis=0)
        all_globals = np.stack([s.global_features for s in samples], axis=0)

        seq_mean = all_tokens.mean(axis=0).astype(np.float32)
        seq_std = all_tokens.std(axis=0).astype(np.float32)
        glob_mean = all_globals.mean(axis=0).astype(np.float32)
        glob_std = all_globals.std(axis=0).astype(np.float32)

        # Avoid division by zero for constant features
        seq_std[seq_std < 1e-8] = 1.0
        glob_std[glob_std < 1e-8] = 1.0

        return Normalizer(
            seq_mean=seq_mean,
            seq_std=seq_std,
            glob_mean=glob_mean,
            glob_std=glob_std,
        )

    def normalize(self, segment_tokens: np.ndarray, global_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply z-score normalization to a single sample's features."""
        seq = ((segment_tokens - self.seq_mean) / self.seq_std).astype(np.float32)
        glob = ((global_features - self.glob_mean) / self.glob_std).astype(np.float32)
        return seq, glob

    def to_dict(self) -> dict:
        """Serialize normalizer state for checkpoint storage."""
        return {
            "seq_mean": self.seq_mean.tolist(),
            "seq_std": self.seq_std.tolist(),
            "glob_mean": self.glob_mean.tolist(),
            "glob_std": self.glob_std.tolist(),
        }

    @staticmethod
    def from_dict(data: dict) -> Normalizer:
        """Restore normalizer from a checkpoint dict."""
        return Normalizer(
            seq_mean=np.array(data["seq_mean"], dtype=np.float32),
            seq_std=np.array(data["seq_std"], dtype=np.float32),
            glob_mean=np.array(data["glob_mean"], dtype=np.float32),
            glob_std=np.array(data["glob_std"], dtype=np.float32),
        )
