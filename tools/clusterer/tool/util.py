import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pb.competition_pb2 as pb
from dtw import dtw
from frechetdist import frdist
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from tool import model


def compute_curvature_profile(test_case: pb.SDCTestCase) -> float:
    def angle_between_vectors(v1, v2):
        # Calculate dot product and magnitudes of the vectors
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        mag_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
        # Return the angle in radians between vectors using the dot product formula
        if mag_v1 == 0 or mag_v2 == 0:
            return 0  # to handle division by zero in case of duplicate points
        return math.acos(dot_product / (mag_v1 * mag_v2))

    road_points = test_case.roadPoints
    n = len(road_points)

    if n < 3:
        return 0  # Not enough points to compute curvature

    curvature_sum = 0

    for i in range(1, n - 1):
        # Create vectors between consecutive points
        v1 = (
            road_points[i].x - road_points[i - 1].x,
            road_points[i].y - road_points[i - 1].y,
        )
        v2 = (
            road_points[i + 1].x - road_points[i].x,
            road_points[i + 1].y - road_points[i].y,
        )

        # Compute the angle between vectors and add to the curvature sum
        curvature_sum += angle_between_vectors(v1, v2)

    # Return the average curvature (or you could return the sum, depending on the requirement)
    curvature_profile = curvature_sum / (n - 2)

    return curvature_profile


# Function to compute the Frechet distance between two test cases
def frechet_distance(tc1: pb.SDCTestCase, tc2: pb.SDCTestCase) -> float:
    road1 = [
        (tc1.roadPoints[i].x, tc1.roadPoints[i].y) for i in range(len(tc1.roadPoints))
    ]
    road2 = [
        (tc2.roadPoints[i].x, tc2.roadPoints[i].y) for i in range(len(tc2.roadPoints))
    ]

    # Calculate Frechet distance
    frechet_dist = frdist(road1, road2)
    return frechet_dist


# # Calculate Frechet distance
def dtw_distance(tc1: pb.SDCTestCase, tc2: pb.SDCTestCase) -> float:
    road1 = [
        (tc1.roadPoints[i].x, tc1.roadPoints[i].y) for i in range(len(tc1.roadPoints))
    ]
    road2 = [
        (tc2.roadPoints[i].x, tc2.roadPoints[i].y) for i in range(len(tc2.roadPoints))
    ]

    DTW = dtw(road1, road2, dist_method="euclidean")
    return DTW.distance


def euclidean_distance(tc1: pb.SDCTestCase, tc2: pb.SDCTestCase) -> float:
    road1 = [
        (tc1.roadPoints[i].x, tc1.roadPoints[i].y) for i in range(len(tc1.roadPoints))
    ]
    road2 = [
        (tc2.roadPoints[i].x, tc2.roadPoints[i].y) for i in range(len(tc2.roadPoints))
    ]

    euclidean_dist = np.linalg.norm(np.array(road1) - np.array(road2))
    return euclidean_dist


# Function to compute the distance between two objects
def compute_distance(i, j, objects):
    return i, j, euclidean_distance(objects[i], objects[j])


# Function to compute the distance matrix in parallel
def compute_distance_matrix_parallel(objects):
    n = len(objects)
    dist_matrix = np.zeros((n, n))

    # Create a list of all pairs (i, j) where i < j to avoid redundant calculations
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # Use ThreadPoolExecutor to compute distances in parallel
    with ThreadPoolExecutor() as executor:
        # Map the function to compute distances in parallel for each pair
        results = executor.map(
            lambda pair: compute_distance(pair[0], pair[1], objects), pairs
        )

    # Fill in the distance matrix with the results
    for i, j, dist in results:
        dist_matrix[i, j] = dist
        dist_matrix[j, i] = dist  # Since the matrix is symmetric

    # Normalize the distance matrix
    min_val = np.min(dist_matrix)
    max_val = np.max(dist_matrix)

    dist_matrix = (dist_matrix - min_val) / (max_val - min_val)

    return dist_matrix


def cluster_test_cases(test_cases):
    # Compute the distance matrix in parallel
    dist_matrix = compute_distance_matrix_parallel(test_cases)

    # Assume `distance_matrix` is your computed distance matrix
    min_val = np.min(dist_matrix)
    max_val = np.max(dist_matrix)

    # Normalize the distance matrix
    dist_matrix = (dist_matrix - min_val) / (max_val - min_val)

    # Z = linkage(dist_matrix, method='average')
    # clusters = fcluster(Z, t=1.4, criterion='distance')

    db = DBSCAN(eps=0.05, min_samples=4, metric="precomputed")
    clusters = db.fit_predict(dist_matrix)

    # Group test cases by their clusters
    clustered_test_cases = {}
    for idx, cluster_id in enumerate(clusters):
        if cluster_id not in clustered_test_cases:
            clustered_test_cases[cluster_id] = []
        clustered_test_cases[cluster_id].append(test_cases[idx])

    return clustered_test_cases


def calculate_distance(x, y):
    dist = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    return np.sum(dist)


def calculate_sinuosity(x, y):
    actual_distance = calculate_distance(x, y)
    straight_line_distance = np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2)
    return (
        actual_distance / straight_line_distance if straight_line_distance != 0 else 1
    )


def calculate_curvature(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2) ** 1.5
    curvature = gaussian_filter1d(curvature, sigma=2)
    return curvature


def count_segments(curvature, curvature_threshold=0.05):
    """Count the number of straight and curved segments based on curvature threshold."""
    straight_segments = np.sum(curvature < curvature_threshold)
    curved_segments = np.sum(curvature >= curvature_threshold)
    return straight_segments, curved_segments


def categorize_turns(curvature, mild_threshold=0.05, sharp_threshold=0.15):
    """Categorize turns into mild, moderate, and sharp based on curvature thresholds."""
    mild_turns = np.sum((curvature >= mild_threshold) & (curvature < sharp_threshold))
    moderate_turns = np.sum(
        (curvature >= sharp_threshold) & (curvature < sharp_threshold * 2)
    )
    sharp_turns = np.sum(curvature >= sharp_threshold * 2)
    return mild_turns, moderate_turns, sharp_turns


def calculate_segment_lengths(x, y, segment_indices):
    """Calculate the lengths of segments based on indices where segments change."""
    segment_lengths = []
    for i in range(1, len(segment_indices)):
        segment_length = calculate_distance(
            x[segment_indices[i - 1] : segment_indices[i]],
            y[segment_indices[i - 1] : segment_indices[i]],
        )
        segment_lengths.append(segment_length)
    return np.mean(segment_lengths) if segment_lengths else 0


def calculate_heading_changes(x, y):
    headings = np.arctan2(np.diff(y), np.diff(x))
    heading_changes = np.diff(headings)
    heading_changes = gaussian_filter1d(heading_changes, sigma=2)

    # Count left and right turns based on heading change sign
    left_turns = np.sum(heading_changes > 0)
    right_turns = np.sum(heading_changes < 0)

    return left_turns, right_turns, np.abs(heading_changes)


def extract_road_features(x, y) -> model.RoadFeatures:
    sinuosity = calculate_sinuosity(x, y)
    curvature = calculate_curvature(x, y)

    curvature_mean = np.mean(curvature)
    curvature_max = np.max(curvature)
    curvature_std = np.std(curvature)

    # Curvature variability index
    curvature_variability_index = (
        curvature_std / curvature_mean if curvature_mean != 0 else 0
    )

    # Count segments
    straight_segments, curved_segments = count_segments(curvature)
    total_segments = straight_segments + curved_segments

    # Average segment length
    segment_indices = np.where(np.diff(np.sign(curvature - 0.05)))[0]
    average_segment_length = calculate_segment_lengths(x, y, segment_indices)

    left_turns, right_turns, heading_change_rate = calculate_heading_changes(x, y)
    heading_change_mean = np.mean(heading_change_rate)
    heading_change_max = np.max(heading_change_rate)
    heading_change_std = np.std(heading_change_rate)

    mild_turns, moderate_turns, sharp_turns = categorize_turns(curvature)

    total_turns = mild_turns + moderate_turns + sharp_turns
    sharp_turn_ratio = sharp_turns / total_turns if total_turns > 0 else 0

    return model.RoadFeatures(
        sinuosity=sinuosity,
        curvature_mean=curvature_mean,
        curvature_max=curvature_max,
        curvature_std=curvature_std,
        curvature_variability_index=curvature_variability_index,
        heading_change_mean=heading_change_mean,
        heading_change_max=heading_change_max,
        heading_change_std=heading_change_std,
        straight_segments=straight_segments,
        curved_segments=curved_segments,
        total_segments=total_segments,
        mild_turns=mild_turns,
        moderate_turns=moderate_turns,
        sharp_turns=sharp_turns,
        sharp_turn_ratio=sharp_turn_ratio,
        left_turns=left_turns,
        right_turns=right_turns,
        average_segment_length=average_segment_length,
    )


def cluster_road_segments(test_cases):
    feature_matrix = []
    for test_case in test_cases:
        x = [point.x for point in test_case.roadPoints]
        y = [point.y for point in test_case.roadPoints]
        features = extract_road_features(x, y)
        feature_vector = features.to_vector()
        feature_matrix.append(feature_vector)

    # Normalize the feature matrix
    ss = StandardScaler()
    feature_matrix = ss.fit_transform(feature_matrix)

    # Cluster the feature matrix using DBSCAN
    # db = DBSCAN(eps=2.5, min_samples=2).fit(feature_matrix)
    # labels = db.labels_

    # Cluster the feature matrix using K-means
    # kmeans = KMeans(n_clusters=20, random_state=42).fit(feature_matrix)
    # labels = kmeans.labels_

    # Cluster the feature matrix using Agglomerative Clustering
    Z = linkage(feature_matrix, method="average")
    labels = fcluster(Z, t=2.7, criterion="distance")

    # Group test cases by their clusters
    clustered_test_cases = {}
    for idx, cluster_id in enumerate(labels):
        if cluster_id not in clustered_test_cases:
            clustered_test_cases[cluster_id] = []
        clustered_test_cases[cluster_id].append(test_cases[idx])

    return clustered_test_cases
