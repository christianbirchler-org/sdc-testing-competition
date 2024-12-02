import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import grpc
import competition_pb2_grpc
import competition_pb2
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import List, Tuple, Dict
from dataclasses import dataclass
import argparse

@dataclass
class RoadFeatures:
    complexity: float
    length: float
    angles: np.ndarray
    curvatures: np.ndarray
    feature_vector: np.ndarray

class RefinedEnsembleSelector(competition_pb2_grpc.CompetitionToolServicer):
    def __init__(self):
        self.scaler = StandardScaler()
        self.rf_classifier = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            class_weight='balanced'
        )
        self.gb_classifier = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=5,
            learning_rate=0.1
        )
        
        self.feature_cache = {}
        self.selected_tests = []
        self.failure_patterns = {}
        
        # Selection parameters
        self.selection_ratio = 0.5
        self.failure_weight = 0.4
        self.diversity_weight = 0.4
        self.complexity_weight = 0.2
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def Name(self, request, context):
        return competition_pb2.NameReply(name="refined_ensemble_selector")
        
    def extract_features(self, road_points: List[Tuple[float, float]]) -> RoadFeatures:
        """Efficient feature extraction focusing on key characteristics"""
        points = np.array(road_points)
        segments = np.diff(points, axis=0)
        
        # Basic geometric features
        segment_lengths = np.sqrt(np.sum(segments**2, axis=1))
        total_length = np.sum(segment_lengths)
        
        # Angle features
        segment_vectors = segments / np.expand_dims(segment_lengths + 1e-8, axis=1)
        dot_products = np.sum(segment_vectors[1:] * segment_vectors[:-1], axis=1)
        angles = np.arccos(np.clip(dot_products, -1, 1))
        
        # Curvature features
        curvature = np.abs(angles) / (segment_lengths[1:] + 1e-8)
        
        # Key statistical features
        percentiles = [25, 50, 75]
        angle_stats = np.array([
            np.mean(angles),
            np.std(angles),
            np.max(np.abs(angles)),
            *np.percentile(angles, percentiles),
            np.sum(angles > np.pi/4)  # Sharp turns
        ])
        
        curvature_stats = np.array([
            np.mean(curvature),
            np.std(curvature),
            np.max(curvature),
            *np.percentile(curvature, percentiles)
        ])
        
        # Complexity metrics
        complexity_score = (
            np.mean(curvature) * np.std(angles) * 
            np.log1p(total_length) * np.max(angles)
        )
        
        # Combined feature vector
        feature_vector = np.concatenate([
            angle_stats,
            curvature_stats,
            [total_length / 1000, complexity_score]  # Normalized length
        ])
        
        return RoadFeatures(
            complexity=complexity_score,
            length=total_length,
            angles=angles,
            curvatures=curvature,
            feature_vector=feature_vector
        )

    def Initialize(self, request_iterator, context):
        """Initialize with efficient model training"""
        self.logger.info("Starting initialization...")
        
        features_list = []
        labels = []
        
        for oracle in request_iterator:
            road_points = [(pt.x, pt.y) for pt in oracle.testCase.roadPoints]
            features = self.extract_features(road_points)
            
            self.feature_cache[oracle.testCase.testId] = features
            features_list.append(features.feature_vector)
            labels.append(oracle.hasFailed)
            
            if oracle.hasFailed:
                if len(self.failure_patterns) < 2:  # Store only two patterns for reference
                    self.failure_patterns[oracle.testCase.testId] = features
        
        if not features_list:
            return competition_pb2.InitializationReply(ok=False)
        
        X = np.array(features_list)
        y = np.array(labels)
        
        # Scale features and train models
        X_scaled = self.scaler.fit_transform(X)
        self.rf_classifier.fit(X_scaled, y)
        self.gb_classifier.fit(X_scaled, y)
        
        self.logger.info(f"Initialization complete with {len(features_list)} samples")
        return competition_pb2.InitializationReply(ok=True)

    def compute_diversity_score(self, features: RoadFeatures) -> float:
        """Efficient diversity computation"""
        if not self.selected_tests:
            return 1.0
        
        current_features = features.feature_vector.reshape(1, -1)
        selected_features = np.array([
            self.feature_cache[tid].feature_vector 
            for tid in self.selected_tests[-5:]  # Compare only with last 5 selections
        ])
        
        similarities = cosine_similarity(current_features, selected_features)
        return 1.0 - np.max(similarities)

    def compute_selection_score(self, test_id: str, features: RoadFeatures) -> float:
        """Compute balanced selection score"""
        X = features.feature_vector.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Combined failure prediction
        rf_prob = self.rf_classifier.predict_proba(X_scaled)[0][1]
        gb_prob = self.gb_classifier.predict_proba(X_scaled)[0][1]
        failure_prob = 0.6 * rf_prob + 0.4 * gb_prob
        
        # Diversity score
        diversity_score = self.compute_diversity_score(features)
        
        # Complexity score
        complexity_score = np.clip(features.complexity / 10, 0, 1)
        
        # Combined score with balanced weights
        score = (
            self.failure_weight * failure_prob +
            self.diversity_weight * diversity_score +
            self.complexity_weight * complexity_score
        )
        
        return score

    def Select(self, request_iterator, context):
        """Efficient test selection process"""
        self.logger.info("Starting test selection...")
        self.selected_tests = []
        
        test_cases = []
        for test_case in request_iterator:
            road_points = [(pt.x, pt.y) for pt in test_case.roadPoints]
            
            if test_case.testId not in self.feature_cache:
                features = self.extract_features(road_points)
                self.feature_cache[test_case.testId] = features
                
            test_cases.append(test_case)
        
        total_tests = len(test_cases)
        target_selections = int(total_tests * self.selection_ratio)
        
        # Compute scores for all tests
        scores = {
            test_case.testId: self.compute_selection_score(
                test_case.testId,
                self.feature_cache[test_case.testId]
            )
            for test_case in test_cases
        }
        
        # Sort and select top tests
        sorted_tests = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected_count = 0
        
        for test_id, score in sorted_tests:
            if selected_count >= target_selections:
                break
                
            self.selected_tests.append(test_id)
            selected_count += 1
            yield competition_pb2.SelectionReply(testId=test_id)
        
        self.logger.info(f"Selected {selected_count} out of {total_tests} tests")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", required=True)
    args = parser.parse_args()
    
    server = grpc.server(ThreadPoolExecutor(max_workers=4))
    competition_pb2_grpc.add_CompetitionToolServicer_to_server(
        RefinedEnsembleSelector(), server)
    
    server.add_insecure_port(f"[::]:{args.port}")
    print(f"Starting server on port {args.port}")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    main()