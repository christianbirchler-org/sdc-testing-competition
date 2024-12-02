import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import grpc
import competition_pb2_grpc
import competition_pb2
from concurrent.futures import ThreadPoolExecutor
import logging
import argparse
from functools import lru_cache
from sklearn.preprocessing import StandardScaler

class FaultPredictionGNN(nn.Module):
    """
    Neural Network model for predicting test failures based on road features.
    Architecture: Multi-layer feed-forward network with normalization and dropout.
    """
    def __init__(self, feature_dim=8):
        super().__init__()
        # Sequential layers with progressive dimension reduction
        self.layers = nn.Sequential(
            # First layer: Input to 32 units
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.LayerNorm(32),  # Normalize activations
            nn.Dropout(0.1),   # Prevent overfitting
            
            # Second layer: 32 to 16 units
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.LayerNorm(16),
            
            # Third layer: 16 to 8 units
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.LayerNorm(8),
            
            # Output layer: 8 to 1 unit (binary classification)
            nn.Linear(8, 1),
            nn.Sigmoid()  # Output probability between 0 and 1
        )
        
        # Initialize weights using orthogonal initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

    def forward(self, x):
        return self.layers(x)

class GraphSelector(competition_pb2_grpc.CompetitionToolServicer):
    """
    Main selector class implementing test selection logic using GNN-based predictions.
    Combines road geometry analysis with historical data and diversity measures.
    """
    def __init__(self):
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize models and data structures
        self.model = FaultPredictionGNN().to(self.device)
        self.scaler = StandardScaler()
        
        # Cache structures
        self.features = {}  # Store feature vectors
        self.failures = {}  # Track historical failures
        self.selected_ids = set()  # Currently selected tests
        
        # Selection parameters
        self.selection_ratio = 0.4  # Maximum portion of tests to select
        self.min_selections = 40    # Minimum number of selections
        self.min_score = 0.3       # Minimum score threshold
        self.history_multiplier = 2.0  # Bonus for historical failures
        self.failure_weight = 0.7     # Weight for failure prediction
        
    def Name(self, request, context):
        """Return identifier for this selector"""
        return competition_pb2.NameReply(name="gnn_fault_detector")

    @lru_cache(maxsize=1024)
    def extract_features(self, points_tuple):
        """
        Extract geometric features from road points.
        Features include: length, directness, curvature, turns, complexity.
        Uses caching for efficiency.
        """
        points = np.array(points_tuple)
        
        if len(points) < 2:
            return np.zeros(8)

        # Compute basic geometric properties
        segments = np.diff(points, axis=0)
        segment_lengths = np.sqrt(np.sum(segments**2, axis=1))
        total_length = np.sum(segment_lengths)
        direct_dist = np.linalg.norm(points[-1] - points[0])
        
        # Compute angles and curvature
        vectors = segments / (segment_lengths[:, np.newaxis] + 1e-8)
        angles = np.arccos(np.clip(np.sum(vectors[1:] * vectors[:-1], axis=1), -1, 1))
        curvature = angles / (segment_lengths[1:] + 1e-8)
        
        # Analyze turns and complexity
        sharp_turns = np.sum(angles > np.pi/4)
        turn_density = sharp_turns / (len(points) + 1e-8)
        mean_curvature = np.mean(curvature) if len(curvature) > 0 else 0
        max_curvature = np.max(curvature) if len(curvature) > 0 else 0
        
        # Compute road complexity (ratio of actual to direct length)
        complexity = total_length / (direct_dist + 1e-8)
        
        # Create feature vector
        features = np.array([
            total_length / 100.0,       # Normalized total length
            direct_dist / 100.0,        # Normalized direct distance
            mean_curvature,             # Average curvature
            max_curvature,              # Maximum curvature
            np.std(curvature) if len(curvature) > 0 else 0,  # Curvature variation
            turn_density,               # Density of sharp turns
            min(complexity, 5.0) / 5.0, # Normalized complexity
            sharp_turns / max(3, len(angles))  # Turn ratio
        ], dtype=np.float32)
        
        return features
    
    def train_model(self, features, labels):
        """
        Train the GNN model with balanced weighting and early stopping.
        Uses AdamW optimizer and BCELoss with class weighting.
        """
        self.model.train()
        features_scaled = self.scaler.fit_transform(features)
        
        # Calculate class weights for imbalanced data
        pos_count = sum(labels)
        neg_count = len(labels) - pos_count
        pos_weight = torch.tensor([neg_count / (pos_count + 1e-8)]).to(self.device)
        
        # Setup optimizer and loss
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            weight_decay=0.01
        )
        criterion = nn.BCELoss(weight=pos_weight)
        
        # Convert data to tensors
        features_tensor = torch.FloatTensor(features_scaled).to(self.device)
        labels_tensor = torch.FloatTensor(labels).to(self.device)
        
        # Training loop with early stopping
        best_loss = float('inf')
        best_state = None
        patience = 3
        no_improve = 0
        min_epochs = 10
        
        for epoch in range(20):
            # Multiple passes per epoch
            for _ in range(2): 
                optimizer.zero_grad()
                outputs = self.model(features_tensor)
                loss = criterion(outputs.squeeze(), labels_tensor)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                optimizer.step()
            
            # Log progress
            if epoch % 2 == 0:
                self.logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
            # Early stopping check
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            
            if epoch >= min_epochs and no_improve >= patience:
                break
                
        # Load best model state
        if best_state:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
    
    def Initialize(self, request_iterator, context):
        """
        Initialize selector with training data.
        Process Oracle data to train model and establish baseline patterns.
        """
        self.logger.info("Starting initialization...")
        
        train_features = []
        train_labels = []
        failed_features = []
        
        # Process each Oracle (test case with known outcome)
        for oracle in request_iterator:
            road_points = [(pt.x, pt.y) for pt in oracle.testCase.roadPoints]
            features = self.extract_features(tuple(map(tuple, road_points)))
            
            # Store features and failure status
            self.features[oracle.testCase.testId] = features
            self.failures[oracle.testCase.testId] = oracle.hasFailed
            
            train_features.append(features)
            train_labels.append(float(oracle.hasFailed))
            
            if oracle.hasFailed:
                failed_features.append(features)
        
        # Train model if we have data
        if train_features:
            train_features = np.stack(train_features)
            self.train_model(train_features, train_labels)
        
        # Compute mean feature vector of failed tests
        self.failure_mean = np.mean(failed_features, axis=0) if failed_features else None
        
        return competition_pb2.InitializationReply(ok=True)
    
    def compute_score(self, test_id, features):
        """
        Compute selection score combining multiple factors:
        - Failure probability from model
        - Historical performance
        - Pattern matching with known failures
        - Diversity from selected tests
        """
        # Get model prediction
        with torch.no_grad():
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            features_tensor = torch.FloatTensor(features_scaled).to(self.device)
            failure_prob = self.model(features_tensor).item()
        
        # Apply history bonus
        history_bonus = self.history_multiplier if self.failures.get(test_id, False) else 1.0
        
        # Calculate pattern matching bonus
        pattern_bonus = 0.0
        if self.failure_mean is not None:
            similarity = np.dot(features, self.failure_mean) / (
                np.linalg.norm(features) * np.linalg.norm(self.failure_mean) + 1e-8
            )
            pattern_bonus = max(0, similarity)
        
        # Calculate diversity score
        diversity_score = 1.0
        if self.selected_ids:
            similarities = []
            for selected_id in list(self.selected_ids)[-3:]:
                selected_features = self.features[selected_id]
                similarity = np.dot(features, selected_features) / (
                    np.linalg.norm(features) * np.linalg.norm(selected_features) + 1e-8
                )
                similarities.append(similarity)
            diversity_score = 1.0 - min(1.0, max(similarities))
        
        # Combine all factors
        return (
            self.failure_weight * failure_prob * history_bonus +
            (1 - self.failure_weight) * (
                0.7 * pattern_bonus +
                0.3 * diversity_score
            )
        )
    
    def Select(self, request_iterator, context):
        """
        Main selection logic:
        1. Score all test cases
        2. Select minimum required number
        3. Add additional high-scoring tests up to maximum
        """
        self.logger.info("Starting selection...")
        self.model.eval()
        self.selected_ids.clear()
        
        test_scores = {}
        total_tests = 0
        
        # Score all tests
        for test_case in request_iterator:
            total_tests += 1
            road_points = [(pt.x, pt.y) for pt in test_case.roadPoints]
            features = self.extract_features(tuple(map(tuple, road_points)))
            
            self.features[test_case.testId] = features
            test_scores[test_case.testId] = self.compute_score(test_case.testId, features)
        
        # Calculate selection bounds
        min_selections = max(self.min_selections, int(total_tests * 0.25))
        max_selections = int(total_tests * self.selection_ratio)
        
        # Sort tests by score
        sorted_tests = sorted(test_scores.items(), key=lambda x: x[1], reverse=True)
        
        # First pass: minimum required selections
        for test_id, _ in sorted_tests[:min_selections]:
            self.selected_ids.add(test_id)
            yield competition_pb2.SelectionReply(testId=test_id)
        
        # Second pass: additional high-scoring tests
        for test_id, score in sorted_tests[min_selections:]:
            if len(self.selected_ids) >= max_selections or score < self.min_score:
                break
            
            self.selected_ids.add(test_id)
            yield competition_pb2.SelectionReply(testId=test_id)
        
        self.logger.info(f"Selected {len(self.selected_ids)} out of {total_tests} tests")

def main():
    """Setup and run the gRPC server"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", required=True)
    args = parser.parse_args()
    
    server = grpc.server(ThreadPoolExecutor(max_workers=4))
    competition_pb2_grpc.add_CompetitionToolServicer_to_server(
        GraphSelector(), server)
    
    server.add_insecure_port(f"[::]:{args.port}")
    print(f"Starting server on port {args.port}")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    main()