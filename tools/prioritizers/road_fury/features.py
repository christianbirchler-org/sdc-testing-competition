"""
Feature engineering for SDC test prioritization — 10-channel version.

Extracts 10 road geometry features per point from road point sequences.
Used by the RoadTransformer model for FAIL/PASS prediction.

Channels:
  0. seg_length      — distance to next point
  1. abs_angle_change — absolute heading change
  2. curvature        — 3-point Menger curvature
  3. curv_jerk        — derivative of curvature
  4. cum_dist_norm    — cumulative distance (normalized 0→1)
  5. heading_sin      — sin(heading angle)
  6. heading_cos      — cos(heading angle)
  7. rel_position     — relative position along road (0→1)
  8. local_curv_std   — local curvature std (window=11)
  9. curv_acceleration — 2nd derivative of curvature
"""

import math
import numpy as np
from scipy.interpolate import interp1d


TARGET_SEQ_LEN = 197  # Competition standard road point count

FEATURE_NAMES = [
    'seg_length', 'abs_angle_change', 'curvature', 'curv_jerk',
    'cum_dist_norm', 'heading_sin', 'heading_cos', 'rel_position',
    'local_curv_std', 'curv_acceleration'
]
NUM_FEATURES = len(FEATURE_NAMES)


def _compute_curvature(pts: np.ndarray) -> np.ndarray:
    """3-point Menger curvature at each interior point."""
    n = len(pts)
    curv = np.zeros(n - 2)
    for i in range(n - 2):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        x3, y3 = pts[i + 2]
        a = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        b = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
        c = math.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
        s = 0.5 * (a + b + c)
        at = s * (s - a) * (s - b) * (s - c)
        if at <= 1e-10:
            curv[i] = 0.0
        else:
            R = a * b * c / (4 * math.sqrt(at))
            curv[i] = 1.0 / R if R > 0 else 0.0
    return curv


def adjust_array_size(pts: np.ndarray, target_size: int = TARGET_SEQ_LEN) -> np.ndarray:
    """Interpolate road points to exactly target_size points."""
    if len(pts) == target_size:
        return pts
    current_idx = np.linspace(0, len(pts) - 1, num=len(pts))
    target_idx = np.linspace(0, len(pts) - 1, num=target_size)
    interp_x = interp1d(current_idx, pts[:, 0], kind='linear')
    interp_y = interp1d(current_idx, pts[:, 1], kind='linear')
    return np.column_stack([interp_x(target_idx), interp_y(target_idx)])


def compute_features(pts_raw) -> np.ndarray:
    """
    Extract 10-channel sequential features from road points.

    Accepts: list of dicts, gRPC RoadPoint objects, list of [x,y], or ndarray.
    Returns: [N, 10] float32 array (N = TARGET_SEQ_LEN after interpolation).
    """
    # --- Parse input format ---
    if isinstance(pts_raw, np.ndarray):
        pts = pts_raw.reshape(-1, 2).astype(np.float64)
    elif isinstance(pts_raw[0], dict):
        pts = np.array([(p['x'], p['y']) for p in pts_raw], dtype=np.float64)
    elif hasattr(pts_raw[0], 'x'):
        # gRPC RoadPoint objects
        pts = np.array([(p.x, p.y) for p in pts_raw], dtype=np.float64)
    else:
        pts = np.array(pts_raw, dtype=np.float64).reshape(-1, 2)

    # --- Interpolate to standard length ---
    if len(pts) != TARGET_SEQ_LEN:
        pts = adjust_array_size(pts, TARGET_SEQ_LEN)

    n = len(pts)

    # 0. Segment lengths
    diffs = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    seg_full = np.pad(seg_lens, (0, 1), mode='edge')

    # 1. Absolute angle changes
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
    ac = np.diff(angles)
    ac = (ac + np.pi) % (2 * np.pi) - np.pi
    abs_ac_full = np.pad(np.abs(ac), (1, 1), mode='constant')

    # 2. Curvature
    curv = np.abs(_compute_curvature(pts))
    curv_full = np.pad(curv, (1, 1), mode='constant')

    # 3. Curvature jerk (1st derivative)
    curv_deriv_full = np.pad(np.diff(curv_full), (0, 1), mode='constant')

    # 4. Cumulative distance (normalized)
    cum_dist = np.cumsum(seg_full)
    cum_dist_norm = cum_dist / (cum_dist[-1] + 1e-8)

    # 5-6. Heading sin/cos
    heading_full = np.pad(angles, (0, 1), mode='edge')
    heading_sin = np.sin(heading_full)
    heading_cos = np.cos(heading_full)

    # 7. Relative position
    rel_pos = np.linspace(0, 1, n)

    # 8. Local curvature std (window=11)
    w, hw = 11, 5
    local_std = np.zeros(n)
    for i in range(n):
        s, e = max(0, i - hw), min(n, i + hw + 1)
        local_std[i] = np.std(curv_full[s:e])

    # 9. Curvature acceleration (2nd derivative)
    curv_accel_full = np.pad(np.diff(curv_deriv_full), (0, 1), mode='constant')

    return np.column_stack([
        seg_full, abs_ac_full, curv_full, curv_deriv_full, cum_dist_norm,
        heading_sin, heading_cos, rel_pos, local_std, curv_accel_full
    ]).astype(np.float32)

