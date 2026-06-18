"""Prioritization strategies for SDC test suites.

Implements the **Strategy Pattern** so that new prioritization algorithms
can be added without modifying existing code (Open/Closed Principle).

Adding a new strategy:
    1. Create a class that inherits from ``PrioritizationStrategy``.
    2. Implement the ``prioritize`` method.
    3. Register it in ``STRATEGY_REGISTRY`` with a kebab-case key.

Architecture note:
    Strategies operate on ``TestCaseData`` – a lightweight, transport-
    agnostic domain object.  This keeps the strategy logic decoupled from
    both the REST layer (Pydantic models) and the gRPC layer (Protobuf),
    allowing the same strategies to be reused across transport layers.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Type
import math
import numpy as np

# =============================================================================
#   Exceptions
# =============================================================================
class StrategyNotFoundError(Exception):
    """Raised when the requested prioritization strategy does not exist."""
    pass

# =============================================================================
#   Logger
# =============================================================================
logger = logging.getLogger(Path(__file__).stem)


# =============================================================================
#   Domain Object – transport-agnostic test case representation
# =============================================================================
@dataclass(frozen=True)
class TestCaseData:
    """Transport-agnostic domain object carrying raw road points for feature
    implementation with gRPC.

    Strategies receive full road data and compute  metrics they need internally.
    This keeps strategies self-contained and allows new ones to be added
    without modifying persistence or upload logic to meet Open/Closed
    Principle.
    """
    test_id: str
    road_points: List[Tuple[float, float]]  # ordered (x, y) pairs

# =============================================================================
#   Preparation Functions – Cross methods among strategies
# =============================================================================

def _simplify_by_inflection(pts: List[Tuple[float, float]]) -> List[int]:
    """Find inflection points where road curvature changes sign.

    Right-hand rule:
        - positive cross-product means left turn (anti-clockwise)
        - negative cross-product means right turn (clockwise)

    Uses the signed cross product of consecutive displacement vectors.

    A sign change marks a vertex of the simplified polyline.

    Args:
        pts: Road points as (x, y) tuples.

    Returns:
        Indices of inflection points (including start and end).
    """
    n = len(pts)
    if n < 3:
        return list(range(n))

    vertices = [0]

    # Initial turn direction
    prev_cross = (
        (pts[1][0] - pts[0][0]) * (pts[2][1] - pts[1][1])
        - (pts[1][1] - pts[0][1]) * (pts[2][0] - pts[1][0])
    )

    for i in range(1, n - 1):
        cross = (
            (pts[i][0] - pts[i - 1][0]) * (pts[i + 1][1] - pts[i][1])
            - (pts[i][1] - pts[i - 1][1]) * (pts[i + 1][0] - pts[i][0])
        )

        if cross == 0:
            continue

        # Sign change → inflection point
        if prev_cross != 0 and (cross > 0) != (prev_cross > 0):
            vertices.append(i)

        prev_cross = cross

    vertices.append(n - 1)
    return vertices

def _shoelace_area(points: List[Tuple[float, float]]) -> float:
    """Compute polygon area via Shoelace formula
        Ref: https://en.wikipedia.org/wiki/Shoelace_formula.

    Used to measure the deviation between the actual road curve
    and the simplified straight-line segment:
        Ref: https://doi.org/10.48550/arXiv.2111.04666

    Args:
        points: Polygon vertices in order (auto-closed).

    Returns:
        Absolute area of the polygon.
    """
    n = len(points)
    if n < 3:
        return 0.0

    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        # 2x2 Determinants computation over 2D-coordinates:
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]

    return abs(area) / 2.0

def extract_features(tc: TestCaseData, divide_pi: int = 18) -> List[float]:
    """Extract road geometry features from a test case.

    Features (from literature, Table 1 — Road Characteristics):
        F1:  Direct Distance — Euclidean between start and finish
        F2:  Road Distance   — Total road segment length
        F6:  Total Angle     — Sum of all angle changes
        F9:  Max Angle       — Maximum angle change
        F11: Mean Angle      — Average angle change
        F8:  Std Angle       — Std deviation of angle changes
        F14: Road Safety Sum  — Total area between road and simplified polyline
        F15: Road Safety Mean — Mean area per segment

    Returns:
        Feature vector [F1, F2, F6, F9, F11, F8, F14, F15]
    """
    pts = tc.road_points
    n = len(pts)

    # F1: Direct distance (start to finish)
    direct_dist = math.sqrt(
        (pts[-1][0] - pts[0][0]) ** 2 + (pts[-1][1] - pts[0][1]) ** 2
    )

    # F2: Road distance (total segment length)
    road_dist = 0.0
    for i in range(1, n):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        road_dist += math.sqrt(dx * dx + dy * dy)

    # Angles at each interior point
    angles = []
    for i in range(1, n - 1):
        dx1 = pts[i][0] - pts[i - 1][0]
        dy1 = pts[i][1] - pts[i - 1][1]
        dx2 = pts[i + 1][0] - pts[i][0]
        dy2 = pts[i + 1][1] - pts[i][1]

        len1 = math.sqrt(dx1 * dx1 + dy1 * dy1)
        len2 = math.sqrt(dx2 * dx2 + dy2 * dy2)

        if len1 == 0 or len2 == 0:
            angles.append(0.0)
            continue

        dot = dx1 * dx2 + dy1 * dy2
        cos_a = max(-1.0, min(1.0, dot / (len1 * len2)))
        angles.append(math.acos(cos_a))

    if not angles:
        total_angle = max_angle = mean_angle = std_angle = 0.0
    else:
        total_angle = sum(angles)
        max_angle = max(angles)
        mean_angle = total_angle / len(angles)
        variance = sum((a - mean_angle) ** 2 for a in angles) / len(angles)
        std_angle = math.sqrt(variance)

    # F14, F15: Road Safety — area between road curve and simplified polyline
    safety_sum = 0.0
    safety_mean = 0.0
    if n >= 3:
        vertices = _simplify_by_inflection(pts)
        areas = []
        for i in range(len(vertices) - 1):
            polygon_pts = [pts[j] for j in range(vertices[i], vertices[i + 1] + 1)]
            if len(polygon_pts) >= 3:
                areas.append(_shoelace_area(polygon_pts))
            else:
                areas.append(0.0)
        safety_sum = sum(areas)
        safety_mean = safety_sum / len(areas) if areas else 0.0

    return [
        direct_dist, road_dist, total_angle, max_angle,
        mean_angle, std_angle, safety_sum, safety_mean,
    ]

# =============================================================================
#   Strategy Base Class – Abstract Base Class for validation
# =============================================================================
class PrioritizationStrategy(ABC):
    """Interface that every prioritization strategy must implement."""

    @abstractmethod
    def prioritize(self, test_cases: List[TestCaseData]) -> List[str]:
        """Return test IDs ordered by this strategy's criterion.

        Args:
            test_cases: Unordered collection of test case metrics.

        Returns:
            List of ``test_id`` strings in prioritized order.
        """


# =============================================================================
#   Strategies Logic - Our Strategies
# =============================================================================
class OutlierSortStrategy(PrioritizationStrategy):
    """Sort tests by distance from mean in feature space — most anomalous first.

    Supports two distance metrics:
        - euclidean:    z-score normalize, then Euclidean from origin
        - mahalanobis:  accounts for feature correlations via inverse covariance

    Tests with unusual road geometry (sharp turns, abnormal distances)
    score higher and are prioritized first — targeting likely failures.
    """

    def __init__(self, method: str = "euclidean") -> None:
        self._method = method

    def prioritize(self, test_cases: List[TestCaseData]) -> List[str]:
        features = np.array([extract_features(tc) for tc in test_cases])
        n, d = features.shape
        mean = features.mean(axis=0) # mean over all tests
        diffs = features - mean

        if self._method == "euclidean":
            stds = features.std(axis=0) # std over all tests
            stds[stds == 0] = 1.0 # clip
            z_scores = diffs / stds
            distances = np.sqrt(np.sum(z_scores ** 2, axis=1))

        elif self._method == "mahalanobis":
            cov = np.cov(features, rowvar=False)
            # Regularize for numerical stability (small sample / correlated features)
            reg = 1e-6 * np.eye(d)
            cov_inv = np.linalg.inv(cov + reg)
            distances = np.sqrt(np.sum(diffs @ cov_inv * diffs, axis=1))

        else:
            raise ValueError(f"Unknown method: {self._method}")

        order = np.argsort(-distances)
        return [test_cases[i].test_id for i in order]


class LessSafeStrategy(PrioritizationStrategy):
    """Sort tests by safety sum value — least safe roads first.

    Safety is measured as the total area between the actual road curve
    and the simplified polyline through inflection points (Shoelace formula).
    Higher area = more deviation from straight segments = less safe.
    """
    def prioritize(self, test_cases: List[TestCaseData]) -> List[str]:
        safety_sums = []

        for tc in test_cases:
            pts = tc.road_points
            n = len(pts)

            if n >= 3:

                vertices = _simplify_by_inflection(pts)
                areas = []
                for i in range(len(vertices) - 1):
                    polygon_pts = [pts[j] for j in range(vertices[i], vertices[i + 1] + 1)]
                    if len(polygon_pts) >= 3:
                        areas.append(_shoelace_area(polygon_pts))
                    else:
                        areas.append(0.0)
                safety_sum = sum(areas)
                safety_sums.append(safety_sum)

        order = np.argsort(-np.array(safety_sums))
        return [test_cases[i].test_id for i in order]

# =============================================================================
#   Strategies Logic - Baseline
# =============================================================================
class LongestRoadFirstStrategy(PrioritizationStrategy):
    """Prioritize test cases with the highest number of road points first."""

    def prioritize(self, test_cases: List[TestCaseData]) -> List[str]:
        sorted_cases = sorted(
            test_cases,
            key=lambda tc: len(tc.road_points),
            reverse=True,
        )
        return [tc.test_id for tc in sorted_cases]


class TotalDistanceFirstStrategy(PrioritizationStrategy):
    """Prioritize test cases with the highest total geometric distance first."""

    def prioritize(self, test_cases: List[TestCaseData]) -> List[str]:
        sorted_cases = sorted(
            test_cases,
            key=lambda tc: self._total_distance(tc),
            reverse=True,
        )
        return [tc.test_id for tc in sorted_cases]

    @staticmethod
    def _total_distance(tc: TestCaseData) -> float:
        total = 0.0
        pts = tc.road_points
        for i in range(1, len(pts)):
            dx = pts[i][0] - pts[i - 1][0]
            dy = pts[i][1] - pts[i - 1][1]
            total += math.sqrt(dx * dx + dy * dy)
        return total


# =============================================================================
#   Strategy Registry
# =============================================================================
STRATEGY_REGISTRY: Dict[str, Type[PrioritizationStrategy]] = {
    "longest-first": LongestRoadFirstStrategy,
    "total-distance-first": TotalDistanceFirstStrategy,
    "euclidean-outlier-first": lambda: OutlierSortStrategy("euclidean"),
    "mahalanobis-outlier-first": lambda: OutlierSortStrategy("mahalanobis"),
    "less-safe-first": LessSafeStrategy
}


def get_strategy(name: str) -> PrioritizationStrategy:
    """Look up and instantiate a strategy by its registered name.

    Args:
        name: Kebab-case strategy identifier (e.g. ``"longest-first"``).

    Returns:
        An instance of the requested strategy.

    Raises:
        StrategyNotFoundError: If the name is not in the registry.
    """

    strategy_class = STRATEGY_REGISTRY.get(name)
    if strategy_class is None:
        available = ", ".join(sorted(STRATEGY_REGISTRY.keys()))
        raise StrategyNotFoundError(
            f"Unknown strategy '{name}'. Available strategies: {available}"
        )

    logger.debug("Resolved strategy '%s' → %s", name, strategy_class.__name__)
    return strategy_class()


def available_strategies() -> List[str]:
    """Return the list of registered strategy names."""
    return sorted(STRATEGY_REGISTRY.keys())
