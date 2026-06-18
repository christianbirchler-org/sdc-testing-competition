import math
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy import stats as sp_stats

class RoadData:
    """Centralised data preparation to avoid redundant calculations."""
    def __init__(self, rp):
        self.pts = np.array([[p.x, p.y] for p in rp], dtype=np.float64)
        self.n = len(self.pts)
        
        if self.n < 2:
            self.diffs = self.seg_lens = self.road_len = np.array([])
            return

        self.diffs = np.diff(self.pts, axis=0)
        self.seg_lens = np.linalg.norm(self.diffs, axis=1)
        self.road_len = np.sum(self.seg_lens)
        self.direct_dist = np.linalg.norm(self.pts[-1] - self.pts[0])
        
        if self.n >= 3:
            # Vectorised angle calculation
            v1, v2 = self.diffs[:-1], self.diffs[1:]
            dot = np.einsum('ij,ij->i', v1, v2)
            cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
            self.angles = np.abs(np.arctan2(cross, dot))
            self.crosses = cross
        else:
            self.angles = self.crosses = np.array([])

def _extract_vectorized_consistent_features(data: RoadData):
    """Compute the vectorized subset directly in this module using consistent keys."""
    f = {}

    if data.n < 2:
        return f

    f['f1_direct_distance'] = float(data.direct_dist)
    f['f2_road_distance'] = float(data.road_len)

    if data.n < 3:
        return f

    # --- Group 1: Base Geometry ---
    left_turns = float(np.sum((data.crosses > 0) & (data.angles > 0.035)))
    right_turns = float(np.sum((data.crosses < 0) & (data.angles > 0.035)))
    f['f3_num_left_turns'] = left_turns
    f['f4_num_right_turns'] = right_turns
    f['f5_num_straight'] = float(np.sum(data.angles <= np.deg2rad(0.5)))  # < 0.5° for 197-pt roads
    f['f6_total_angle'] = float(np.sum(data.angles))
    f['f7_median_angle'] = float(np.median(data.angles))
    f['f8_std_angle'] = float(np.std(data.angles))
    f['f9_max_angle'] = float(np.max(data.angles))
    f['f10_min_angle'] = float(np.min(data.angles))
    f['f11_mean_angle'] = float(np.mean(data.angles))
    f['f12_sinuosity'] = float(data.road_len / data.direct_dist) if data.direct_dist > 1e-6 else 1.0
    f['f13_variance'] = float(f['f8_std_angle'] ** 2)
    f['f14_path_efficiency'] = float(data.direct_dist / data.road_len) if data.road_len > 1e-6 else 1.0
    
    seg_len_deltas = np.abs(np.diff(data.seg_lens)) if len(data.seg_lens) >= 2 else np.array([])
    f['f15_jerk'] = float(np.max(seg_len_deltas)) if len(seg_len_deltas) > 0 else 0.0
    cross_sign = np.sign(data.crosses)
    f['f16_inflections'] = float(np.sum(np.diff(cross_sign) != 0)) if len(cross_sign) >= 2 else 0.0

    x_span = float(np.max(data.pts[:, 0]) - np.min(data.pts[:, 0]))
    y_span = float(np.max(data.pts[:, 1]) - np.min(data.pts[:, 1]))
    f['f17_bounding_box_area'] = float(x_span * y_span)
    f['f18_turn_density'] = float((left_turns + right_turns) / data.road_len) if data.road_len > 1e-9 else 0.0
    f['f19_max_segment_length'] = float(np.max(data.seg_lens)) if len(data.seg_lens) > 0 else 0.0
    f['f20_min_segment_length'] = float(np.min(data.seg_lens)) if len(data.seg_lens) > 0 else 0.0
    angle_deltas = np.diff(data.angles) if len(data.angles) >= 2 else np.array([])
    f['f21_angle_variation_rate'] = float(np.std(angle_deltas)) if len(angle_deltas) > 0 else 0.0
    f['f22_curvature_density'] = float(f['f6_total_angle'] / data.road_len) if data.road_len > 1e-9 else 0.0
    f['f23_aspect_ratio'] = float(max(x_span, y_span) / max(min(x_span, y_span), 1e-9))
    turn_total = left_turns + right_turns
    f['f24_path_symmetry'] = float(1.0 - (abs(left_turns - right_turns) / turn_total)) if turn_total > 0 else 1.0

    seg_norm = data.diffs / np.maximum(data.seg_lens[:, None], 1e-12)
    if len(seg_norm) >= 2:
        align = np.einsum('ij,ij->i', seg_norm[:-1], seg_norm[1:])
        f['f25_segment_angle_alignment'] = float(np.mean(np.clip(align, -1.0, 1.0)))
    else:
        f['f25_segment_angle_alignment'] = 1.0

    # --- Group 2: Gaussian Curvature ---
    dx = np.gradient(data.pts[:, 0])
    dy = np.gradient(data.pts[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curv_raw = np.abs(dx * ddy - dy * ddx) / np.maximum((dx ** 2 + dy ** 2) ** 1.5, 1e-12)
    curv = gaussian_filter1d(curv_raw, sigma=2)

    f['f26_curv_mean'] = float(np.mean(curv))
    f['f27_curv_max'] = float(np.max(curv))
    f['f28_curv_std'] = float(np.std(curv))
    f['f29_curv_variability_index'] = float(f['f28_curv_std'] / max(f['f26_curv_mean'], 1e-12))
    f['f30_curv_p75'] = float(np.percentile(curv, 75))
    f['f31_curv_p90'] = float(np.percentile(curv, 90))
    f['f32_curv_p95'] = float(np.percentile(curv, 95))

    headings = np.arctan2(data.diffs[:, 1], data.diffs[:, 0])
    heading_changes = np.diff(headings)
    heading_changes = (heading_changes + np.pi) % (2 * np.pi) - np.pi
    abs_heading_changes = np.abs(heading_changes)
    f['f33_heading_change_mean'] = float(np.mean(abs_heading_changes)) if len(abs_heading_changes) > 0 else 0.0
    f['f34_heading_change_max'] = float(np.max(abs_heading_changes)) if len(abs_heading_changes) > 0 else 0.0
    f['f35_heading_change_std'] = float(np.std(abs_heading_changes)) if len(abs_heading_changes) > 0 else 0.0

    # Thresholds calibrated for 197-point resampled BeamNG roads:
    # per-segment heading changes max ~6°; Gaussian curvature max ~0.076
    mild_thr = np.deg2rad(1)      # >= 1°
    moderate_thr = np.deg2rad(2)  # >= 2°
    sharp_thr = np.deg2rad(4)     # >= 4°
    CURV_SHARP_THR = 0.05
    f['f36_mild_turns'] = float(np.sum((abs_heading_changes >= mild_thr) & (abs_heading_changes < moderate_thr)))
    f['f37_moderate_turns'] = float(np.sum((abs_heading_changes >= moderate_thr) & (abs_heading_changes < sharp_thr)))
    f['f38_sharp_turns'] = float(np.sum(curv >= CURV_SHARP_THR))
    f['f39_sharp_turn_ratio'] = float(f['f38_sharp_turns'] / len(curv)) if len(curv) > 0 else 0.0

    three_pt_curv = _heron_curvature(data.pts, signed=False)
    f['f40_3pt_curv_std'] = float(np.std(three_pt_curv)) if len(three_pt_curv) > 0 else 0.0
    # f41: curvature-onset count — positions where curvature enters the sharp zone
    # (curv >= CURV_SHARP_THR) coming from a low-curvature region (mean of prior 5
    # steps < 50% of threshold).  More meaningful than a per-segment heading jump
    # for densely-resampled roads whose adjacent heading changes are always small.
    _K = 5
    if len(curv) > _K:
        look_back_mean = np.array([np.mean(curv[max(0, i - _K):i]) for i in range(_K, len(curv))])
        f['f41_curv_onset_count'] = float(np.sum(
            (curv[_K:] >= CURV_SHARP_THR) & (look_back_mean < CURV_SHARP_THR * 0.8)
        ))
    else:
        f['f41_curv_onset_count'] = 0.0
    top_k = max(1, int(np.ceil(0.1 * len(curv))))
    f['f42_mean_top10pct_curv'] = float(np.mean(np.partition(curv, -top_k)[-top_k:])) if len(curv) > 0 else 0.0
    f['f43_curv_entropy'] = float(sp_stats.entropy(np.histogram(curv, bins=20)[0] + 1e-12))
    f['f44_heron_curv_max'] = float(np.max(three_pt_curv)) if len(three_pt_curv) > 0 else 0.0
    f['f45_heron_curv_mean'] = float(np.mean(three_pt_curv)) if len(three_pt_curv) > 0 else 0.0

    # --- Group 3: Curvature Profile ---
    profile = np.interp(np.linspace(0, 1, 20), np.linspace(0, 1, len(curv)), curv)
    for i, val in enumerate(profile):
        f[f'f{46 + i}_curv_profile_{i * 5}pct'] = float(val)

    # --- Group 4 & 5: Kinematics & Dynamics ---
    accel = np.diff(data.diffs, axis=0)
    f['f66_dx_mean'] = float(np.mean(data.diffs[:, 0]))
    f['f67_dy_mean'] = float(np.mean(data.diffs[:, 1]))
    f['f68_dx_std'] = float(np.std(data.diffs[:, 0]))
    f['f69_dy_std'] = float(np.std(data.diffs[:, 1]))
    if len(accel) > 0:
        f['f70_accel_x_mean'] = float(np.mean(accel[:, 0]))
        f['f71_accel_y_mean'] = float(np.mean(accel[:, 1]))
        f['f72_accel_x_std'] = float(np.std(accel[:, 0]))
        f['f73_accel_y_std'] = float(np.std(accel[:, 1]))
    else:
        f['f70_accel_x_mean'] = 0.0
        f['f71_accel_y_mean'] = 0.0
        f['f72_accel_x_std'] = 0.0
        f['f73_accel_y_std'] = 0.0
    f['f74_accel_mag_max'] = float(np.max(np.linalg.norm(accel, axis=1))) if len(accel) > 0 else 0.0
    speed = np.linalg.norm(data.diffs, axis=1)
    f['f75_speed_skewness'] = float(sp_stats.skew(speed)) if len(speed) >= 3 else 0.0

    f['f76_heading_change_p25'] = float(np.percentile(abs_heading_changes, 25)) if len(abs_heading_changes) > 0 else 0.0
    f['f77_heading_change_p50'] = float(np.percentile(abs_heading_changes, 50)) if len(abs_heading_changes) > 0 else 0.0
    f['f78_heading_change_p75'] = float(np.percentile(abs_heading_changes, 75)) if len(abs_heading_changes) > 0 else 0.0
    f['f79_seg_length_mean'] = float(np.mean(data.seg_lens)) if len(data.seg_lens) > 0 else 0.0
    f['f80_seg_length_std'] = float(np.std(data.seg_lens)) if len(data.seg_lens) > 0 else 0.0
    f['f81_seg_length_cv'] = float(f['f80_seg_length_std'] / max(f['f79_seg_length_mean'], 1e-12))

    signed_k = _heron_curvature(data.pts, signed=True)
    if len(signed_k) > 0:
        f['f82_signed_curv_mean'] = float(np.mean(signed_k))
        f['f83_signed_curv_std'] = float(np.std(signed_k))
        f['f84_signed_curv_skew'] = float(sp_stats.skew(signed_k)) if len(signed_k) >= 3 else 0.0
        f['f85_signed_curv_kurtosis'] = float(sp_stats.kurtosis(signed_k)) if len(signed_k) >= 3 else 0.0
    else:
        f['f82_signed_curv_mean'] = 0.0
        f['f83_signed_curv_std'] = 0.0
        f['f84_signed_curv_skew'] = 0.0
        f['f85_signed_curv_kurtosis'] = 0.0

    # --- Group 6: Complexity & Interaction ---
    sharp_mask = curv >= CURV_SHARP_THR
    if np.any(sharp_mask):
        padded = np.r_[False, sharp_mask, False]
        changes = np.diff(padded.astype(int))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        cluster_sizes = ends - starts
        f['f86_max_consecutive_sharp'] = float(np.max(cluster_sizes))
        f['f87_num_sharp_clusters'] = float(len(cluster_sizes))
        f['f88_avg_sharp_cluster_size'] = float(np.mean(cluster_sizes))
    else:
        f['f86_max_consecutive_sharp'] = 0.0
        f['f87_num_sharp_clusters'] = 0.0
        f['f88_avg_sharp_cluster_size'] = 0.0

    cum_curv = np.cumsum(curv)
    if len(cum_curv) > 0:
        f['f89_cum_curv_25pct'] = float(np.percentile(cum_curv, 25))
        f['f90_cum_curv_50pct'] = float(np.percentile(cum_curv, 50))
        f['f91_cum_curv_75pct'] = float(np.percentile(cum_curv, 75))
    else:
        f['f89_cum_curv_25pct'] = 0.0
        f['f90_cum_curv_50pct'] = 0.0
        f['f91_cum_curv_75pct'] = 0.0

    f['f92_std_angle_x_sinuosity'] = float(f['f8_std_angle'] * f['f12_sinuosity'])
    f['f93_max_curv_x_road_len'] = float(f['f27_curv_max'] * data.road_len)
    f['f94_sharp_ratio_x_jerk'] = float(f['f39_sharp_turn_ratio'] * f['f15_jerk'])

    if len(curv) > 1:
        c0 = curv[:-1] - np.mean(curv[:-1])
        c1 = curv[1:] - np.mean(curv[1:])
        denom = np.sqrt(np.sum(c0 ** 2) * np.sum(c1 ** 2))
        f['f95_curv_autocorr_lag1'] = float(np.sum(c0 * c1) / denom) if denom > 1e-12 else 0.0
    else:
        f['f95_curv_autocorr_lag1'] = 0.0

    curv_grad = np.gradient(curv) if len(curv) > 1 else np.array([0.0])
    f['f96_curv_gradient_max'] = float(np.max(np.abs(curv_grad)))
    f['f97_curv_gradient_std'] = float(np.std(curv_grad))

    def _window_max_mean(arr, ratio):
        if len(arr) == 0:
            return 0.0
        w = max(1, int(np.ceil(len(arr) * ratio)))
        if len(arr) <= w:
            return float(np.max(arr))
        vals = [np.max(arr[i:i + w]) for i in range(0, len(arr) - w + 1)]
        return float(np.mean(vals)) if vals else 0.0

    f['f98_window_max_10pct'] = _window_max_mean(curv, 0.10)
    f['f99_window_max_25pct'] = _window_max_mean(curv, 0.25)

    f['f100_road_complexity_index'] = float(
        f['f12_sinuosity'] * f['f28_curv_std'] * max(f['f39_sharp_turn_ratio'], 0.01)
    )

    return f

# Stable ordered list of all feature keys produced by extract_all_features.
ALL_FEATURE_KEYS = [
    'f1_direct_distance', 'f2_road_distance', 'f3_num_left_turns', 'f4_num_right_turns',
    'f5_num_straight', 'f6_total_angle', 'f7_median_angle', 'f8_std_angle',
    'f9_max_angle', 'f10_min_angle', 'f11_mean_angle', 'f12_sinuosity',
    'f13_variance', 'f14_path_efficiency', 'f15_jerk', 'f16_inflections',
    'f17_bounding_box_area', 'f18_turn_density', 'f19_max_segment_length', 'f20_min_segment_length',
    'f21_angle_variation_rate', 'f22_curvature_density', 'f23_aspect_ratio', 'f24_path_symmetry',
    'f25_segment_angle_alignment',
    'f26_curv_mean', 'f27_curv_max', 'f28_curv_std', 'f29_curv_variability_index',
    'f30_curv_p75', 'f31_curv_p90', 'f32_curv_p95',
    'f33_heading_change_mean', 'f34_heading_change_max', 'f35_heading_change_std',
    'f36_mild_turns', 'f37_moderate_turns', 'f38_sharp_turns', 'f39_sharp_turn_ratio',
    'f40_3pt_curv_std', 'f41_curv_onset_count', 'f42_mean_top10pct_curv',
    'f43_curv_entropy', 'f44_heron_curv_max', 'f45_heron_curv_mean',
    'f46_curv_profile_0pct', 'f47_curv_profile_5pct', 'f48_curv_profile_10pct',
    'f49_curv_profile_15pct', 'f50_curv_profile_20pct', 'f51_curv_profile_25pct',
    'f52_curv_profile_30pct', 'f53_curv_profile_35pct', 'f54_curv_profile_40pct',
    'f55_curv_profile_45pct', 'f56_curv_profile_50pct', 'f57_curv_profile_55pct',
    'f58_curv_profile_60pct', 'f59_curv_profile_65pct', 'f60_curv_profile_70pct',
    'f61_curv_profile_75pct', 'f62_curv_profile_80pct', 'f63_curv_profile_85pct',
    'f64_curv_profile_90pct', 'f65_curv_profile_95pct',
    'f66_dx_mean', 'f67_dy_mean', 'f68_dx_std', 'f69_dy_std',
    'f70_accel_x_mean', 'f71_accel_y_mean', 'f72_accel_x_std', 'f73_accel_y_std',
    'f74_accel_mag_max', 'f75_speed_skewness',
    'f76_heading_change_p25', 'f77_heading_change_p50', 'f78_heading_change_p75',
    'f79_seg_length_mean', 'f80_seg_length_std', 'f81_seg_length_cv',
    'f82_signed_curv_mean', 'f83_signed_curv_std', 'f84_signed_curv_skew', 'f85_signed_curv_kurtosis',
    'f86_max_consecutive_sharp', 'f87_num_sharp_clusters', 'f88_avg_sharp_cluster_size',
    'f89_cum_curv_25pct', 'f90_cum_curv_50pct', 'f91_cum_curv_75pct',
    'f92_std_angle_x_sinuosity', 'f93_max_curv_x_road_len', 'f94_sharp_ratio_x_jerk',
    'f95_curv_autocorr_lag1', 'f96_curv_gradient_max', 'f97_curv_gradient_std',
    'f98_window_max_10pct', 'f99_window_max_25pct', 'f100_road_complexity_index',
]


def extract_all_features(rp):
    data = RoadData(rp)
    return _extract_vectorized_consistent_features(data)

def _heron_curvature(pts, signed=False):
    """Calculates curvature using Menger/Heron method via vectorised triangles."""
    a = np.linalg.norm(pts[1:-1] - pts[:-2], axis=1)
    b = np.linalg.norm(pts[2:] - pts[1:-1], axis=1)
    c = np.linalg.norm(pts[2:] - pts[:-2], axis=1)
    s = 0.5 * (a + b + c)
    area = np.sqrt(np.maximum(s * (s - a) * (s - b) * (s - c), 0))
    k = 4.0 * area / np.maximum(a * b * c, 1e-12)
    if signed:
        cross = (pts[1:-1,0]-pts[:-2,0])*(pts[2:,1]-pts[1:-1,1]) - \
                (pts[1:-1,1]-pts[:-2,1])*(pts[2:,0]-pts[1:-1,0])
        k *= np.sign(cross)
    return k