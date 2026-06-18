"""Geometric feature extraction from road point sequences.

Each road is represented as a sequence of (x, y) coordinates. From consecutive
pairs of points we compute per-segment features (distance, heading change,
curvature, curvature derivative) and four global road-level statistics.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-8


def as_xy_array(road_points: list | np.ndarray) -> np.ndarray:
    """Convert road points in any supported format to a float32 array of shape [N, 2]."""
    arr = np.asarray(road_points, dtype=object)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    if isinstance(road_points, np.ndarray):
        return road_points.astype(np.float32).reshape(-1, 2)

    # List of dicts with "x" and "y" keys (competition JSON format)
    if isinstance(road_points[0], dict):
        return np.array(
            [[float(p["x"]), float(p["y"])] for p in road_points],
            dtype=np.float32,
        )

    # List of tuples or lists
    return np.array(
        [[float(p[0]), float(p[1])] for p in road_points], dtype=np.float32
    )


def wrapped_diff(angles: np.ndarray) -> np.ndarray:
    """Compute angle differences wrapped to [-pi, pi]."""
    if angles.size == 0:
        return angles
    delta = np.zeros_like(angles, dtype=np.float32)
    if angles.size > 1:
        raw = angles[1:] - angles[:-1]
        delta[1:] = ((raw + np.pi) % (2 * np.pi)) - np.pi
    return delta


def extract_features(road_points: list | np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract segment-level and global features from a road point sequence.

    Returns a tuple (segment_tokens, global_features) where:
      - segment_tokens: float32 array [T, 4] with columns
        (distance, angle_change, curvature, delta_curvature)
      - global_features: float32 array [4] with
        (total_distance, mean_curvature, max_curvature, sinuosity)

    Returns None if the road has fewer than 3 points.
    """
    points = as_xy_array(road_points)
    if points.shape[0] < 3:
        return None

    # Per-segment geometry
    deltas = points[1:] - points[:-1]
    distance = np.linalg.norm(deltas, axis=1).astype(np.float32)
    heading = np.arctan2(deltas[:, 1], deltas[:, 0]).astype(np.float32)
    angle_change = wrapped_diff(heading).astype(np.float32)
    curvature = (angle_change / (distance + EPS)).astype(np.float32)
    delta_curvature = np.zeros_like(curvature)
    if curvature.size > 1:
        delta_curvature[1:] = curvature[1:] - curvature[:-1]

    segment_tokens = np.column_stack(
        [distance, angle_change, curvature, delta_curvature]
    ).astype(np.float32)

    # Global road statistics
    total_distance = float(np.sum(distance))
    mean_curvature = float(np.mean(np.abs(curvature)))
    max_curvature = float(np.max(np.abs(curvature)))
    direct_distance = float(np.linalg.norm(points[-1] - points[0]))
    sinuosity = float(total_distance / (direct_distance + EPS))

    global_features = np.array(
        [total_distance, mean_curvature, max_curvature, sinuosity], dtype=np.float32
    )

    return segment_tokens, global_features
