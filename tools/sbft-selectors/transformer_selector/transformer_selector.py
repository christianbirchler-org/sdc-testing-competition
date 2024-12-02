import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
import grpc
import competition_pb2_grpc
import competition_pb2
from concurrent.futures import ThreadPoolExecutor
import logging
from dataclasses import dataclass
import argparse

@dataclass
class RoadFeatures:
    """
    Dataclass to store comprehensive road geometry features.
    Includes both raw geometric properties and processed features.
    """
    points: np.ndarray        # Original road points
    curvature: np.ndarray    # Curvature at each point
    lengths: np.ndarray      # Segment lengths
    angles: np.ndarray       # Angles between segments
    complexity: float        # Overall road complexity
    turn_count: int         # Number of significant turns
    max_angle: float        # Maximum angle in road
    total_length: float     # Total road length
    mean_curvature: float   # Average curvature
    feature_vector: np.ndarray  # Processed features for model

class TransformerModel(nn.Module):
    """
    Transformer-based model for processing road sequences.
    Uses self-attention to capture relationships between road segments.
    """
    def __init__(self, input_dim=8, hidden_dim=48):
        super().__init__()
        # Input projection layer
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Normalize for stable training
            nn.GELU()                  # Activation function
        )
        
        # Positional encoding for sequence awareness
        self.pos_encoding = nn.Parameter(
            self._create_pos_encoding(hidden_dim, 96),
            requires_grad=False  # Fixed positional encoding
        )
        
        # Transformer encoder layer
        self.encoder = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=3,              # Number of attention heads
            dim_feedforward=96,   # FFN dimension
            dropout=0.15,         # Dropout rate
            batch_first=True,     # Batch dimension first
            norm_first=True       # Pre-norm architecture
        )
        
        # Output projection layers
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)      # Final binary classification
        )
        
        self._init_weights()
    
    def _create_pos_encoding(self, hidden_dim: int, max_len: int) -> torch.Tensor:
        """
        Create sinusoidal positional encodings for sequence positions.
        Uses frequency-based encoding for position awareness.
        """
        pos = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-np.log(10000.0) / hidden_dim))
        pe = torch.zeros(max_len, hidden_dim)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        return pe
    
    def _init_weights(self):
        """Initialize model weights using Xavier uniform initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model.
        Args:
            x: Input tensor of road features
            mask: Attention mask for padding
        Returns:
            Probability of test failure
        """
        x = self.input_proj(x)  # Project input to hidden dimension
        x = x + self.pos_encoding[:x.size(1)].unsqueeze(0)  # Add positional encoding
        x = self.encoder(x, src_key_padding_mask=mask)  # Apply transformer encoder
        x = torch.mean(x, dim=1)  # Global average pooling
        return torch.sigmoid(self.output_head(x))  # Output probability

class RoadProcessor:
    """
    Processes road geometry into features suitable for the transformer model.
    Handles feature extraction and preparation for model input.
    """
    def __init__(self, max_points=96):
        self.max_points = max_points
        self.min_turn_angle = np.pi / 6  # Minimum angle to consider as turn
        
    def compute_road_features(self, points: List[Tuple[float, float]]) -> RoadFeatures:
        """
        Extract comprehensive geometric features from road points.
        Computes various metrics including curvature, angles, and complexity.
        """
        points_array = np.array(points)
        segments = np.diff(points_array, axis=0)
        
        # Basic geometric properties
        lengths = np.sqrt(np.sum(segments**2, axis=1))
        total_length = np.sum(lengths)
        
        # Compute angles and turns
        vectors = segments / (lengths[:, np.newaxis] + 1e-8)
        angles = np.arccos(np.clip(np.sum(vectors[1:] * vectors[:-1], axis=1), -1, 1))
        max_angle = np.max(angles) if len(angles) > 0 else 0
        
        # Compute curvature
        curvature = angles / (lengths[1:] + 1e-8)
        mean_curvature = np.mean(curvature)
        
        # Count significant turns
        turn_count = np.sum(angles > self.min_turn_angle)
        
        # Compute complexity factors
        length_factor = np.clip(total_length / 100, 0.5, 2.0)
        curvature_factor = np.clip(mean_curvature / 0.1, 0, 3.0)
        turn_density = turn_count / max(total_length / 50, 1)
        
        # Overall complexity score
        complexity = (
            0.4 * curvature_factor +
            0.3 * turn_density +
            0.2 * (np.max(curvature) / 0.3) +
            0.1 * length_factor
        )
        
        # Create feature vector for model input
        feature_vector = np.column_stack([
            lengths[:-1] / (total_length + 1e-8),  # Normalized segment lengths
            np.sin(angles),                        # Angle features
            np.cos(angles),                        # Angle features
            curvature,                            # Curvature profile
            np.cumsum(lengths[:-1]) / total_length,  # Cumulative distance
            np.ones_like(angles) * complexity,     # Complexity score
            np.ones_like(angles) * turn_density,   # Turn density
            np.ones_like(angles) * (max_angle / np.pi)  # Normalized max angle
        ])
        
        return RoadFeatures(
            points=points_array,
            curvature=curvature,
            lengths=lengths,
            angles=angles,
            complexity=complexity,
            turn_count=turn_count,
            max_angle=max_angle,
            total_length=total_length,
            mean_curvature=mean_curvature,
            feature_vector=feature_vector
        )
        
    def process_road(self, points: List[Tuple[float, float]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process road points into model-ready tensors.
        Handles padding and masking for fixed-length input.
        """
        features = self.compute_road_features(points)
        
        # Create padded feature matrix
        feature_matrix = np.zeros((self.max_points, features.feature_vector.shape[1]))
        mask = torch.ones(self.max_points, dtype=torch.bool)
        
        # Fill actual values and mask padding
        n_points = min(len(features.feature_vector), self.max_points)
        feature_matrix[:n_points] = features.feature_vector[:n_points]
        mask[:n_points] = False
        
        return torch.FloatTensor(feature_matrix), mask

class TransformerSelector(competition_pb2_grpc.CompetitionToolServicer):
    """
    Main selector class using transformer-based model for test selection.
    Implements the gRPC service interface and manages test selection logic.
    """
    def __init__(self):
        # Setup device and model
        self.device = torch.device("cpu")
        self.model = TransformerModel().to(self.device)
        self.processor = RoadProcessor()
        
        # Cache and tracking structures
        self.feature_cache: Dict[str, RoadFeatures] = {}  # Store computed features
        self.tensor_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}  # Store processed tensors
        self.selected_ids: Set[str] = set()  # Currently selected tests
        self.historical_failures: Dict[str, bool] = {}  # Track past failures
        
        # Selection hyperparameters
        self.base_selection_ratio = 0.45  # Base ratio of tests to select
        self.min_score_threshold = 0.3    # Minimum score for selection
        self.max_selections = 100         # Maximum number of selections
        self.min_selection_ratio = 0.2    # Minimum ratio to select
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def Name(self, request, context):
        """Return identifier for this selector"""
        return competition_pb2.NameReply(name="transformer_selector")
        
    def train_model(self, features_batch, masks_batch, labels_batch):
        """
        Train the transformer model on provided data.
        Implements early stopping and learning rate scheduling.
        
        Args:
            features_batch: Batch of road features
            masks_batch: Attention masks for padding
            labels_batch: Binary labels (pass/fail)
        """
        self.model.train()
        # Setup optimizer and scheduler
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )
        
        # Handle class imbalance
        pos_count = sum(labels_batch)
        pos_weight = torch.FloatTensor([len(labels_batch)/max(pos_count, 1)]).to(self.device)
        criterion = nn.BCELoss(weight=pos_weight)
        
        # Prepare data
        features = torch.stack(features_batch).to(self.device)
        masks = torch.stack(masks_batch).to(self.device)
        labels = torch.FloatTensor(labels_batch).to(self.device)
        
        # Training loop with early stopping
        best_loss = float('inf')
        patience = 3
        no_improve = 0
        
        for epoch in range(8):
            optimizer.zero_grad()
            outputs = self.model(features, masks).squeeze()
            loss = criterion(outputs, labels)
            
            # Early stopping check
            if loss.item() < best_loss * 0.99:
                best_loss = loss.item()
                no_improve = 0
            else:
                no_improve += 1
                
            if no_improve >= patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
                
            # Optimization step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step(loss)
            
            if epoch % 2 == 0:
                self.logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
                
        self.logger.info(f"Final Loss: {best_loss:.4f}")

    def Initialize(self, request_iterator, context):
        """
        Initialize the selector with training data.
        Process and cache features for each test case.
        """
        self.logger.info("Starting initialization")
        features_batch = []
        masks_batch = []
        labels_batch = []
        
        for oracle in request_iterator:
            # Extract road points and compute features
            road_points = [(pt.x, pt.y) for pt in oracle.testCase.roadPoints]
            road_features = self.processor.compute_road_features(road_points)
            tensor_features, mask = self.processor.process_road(road_points)
            
            # Cache computed features
            self.feature_cache[oracle.testCase.testId] = road_features
            self.tensor_cache[oracle.testCase.testId] = (tensor_features, mask)
            self.historical_failures[oracle.testCase.testId] = oracle.hasFailed
            
            # Collect training data
            features_batch.append(tensor_features)
            masks_batch.append(mask)
            labels_batch.append(float(oracle.hasFailed))
        
        # Train model if we have data
        if features_batch:
            self.train_model(features_batch, masks_batch, labels_batch)
            
        return competition_pb2.InitializationReply(ok=True)

    def compute_diversity(self, test_id: str) -> float:
        """
        Compute diversity score for a test case compared to selected tests.
        Uses cosine similarity between feature vectors.
        """
        if not self.selected_ids:
            return 1.0
        
        current_tensor = self.tensor_cache[test_id][0]
        # Compare with last 5 selected tests
        selected_tensors = torch.stack([
            self.tensor_cache[tid][0]
            for tid in list(self.selected_ids)[-5:]
        ]).to(self.device)
        
        # Compute mean features
        current = current_tensor.mean(dim=0, keepdim=True)
        selected = selected_tensors.mean(dim=1)
        
        # Calculate cosine similarity
        distances = F.cosine_similarity(
            current.expand(selected.shape[0], -1),
            selected,
            dim=1
        )
        
        return float(1 - distances.max())

    def compute_score(self, test_id: str, failure_prob: float, diversity: float) -> float:
        """
        Compute final selection score combining multiple factors:
        - Predicted failure probability
        - Historical performance
        - Diversity from selected tests
        - Road complexity
        """
        features = self.feature_cache[test_id]
        historical_bonus = 1.2 if self.historical_failures.get(test_id, False) else 1.0
        
        # Compute complexity score
        complexity_score = features.complexity * features.turn_count / (features.max_angle + 1e-8)
        length_factor = np.clip(features.total_length / 100, 0.5, 2.0)
        
        # Weighted combination of factors
        score = (
            0.45 * failure_prob * historical_bonus +  # Failure prediction
            0.25 * diversity +                        # Diversity
            0.2 * complexity_score +                  # Road complexity
            0.1 * length_factor                       # Length factor
        )
        
        return score

    def Select(self, request_iterator, context):
        """
        Main selection logic:
        1. Process all test cases
        2. Score based on multiple criteria
        3. Select tests meeting minimum requirements
        4. Add additional high-scoring tests
        """
        self.logger.info("Starting selection")
        self.model.eval()
        self.selected_ids.clear()
        
        test_scores = {}
        all_tests = []
        
        # Process and score all test cases
        for test_case in request_iterator:
            try:
                # Compute features if not cached
                if test_case.testId not in self.feature_cache:
                    road_points = [(pt.x, pt.y) for pt in test_case.roadPoints]
                    road_features = self.processor.compute_road_features(road_points)
                    tensor_features, mask = self.processor.process_road(road_points)
                    
                    self.feature_cache[test_case.testId] = road_features
                    self.tensor_cache[test_case.testId] = (tensor_features, mask)
                
                all_tests.append(test_case.testId)
                features, mask = self.tensor_cache[test_case.testId]
                
                # Compute selection score
                with torch.no_grad():
                    features = features.to(self.device).unsqueeze(0)
                    mask = mask.to(self.device).unsqueeze(0)
                    failure_prob = self.model(features, mask).item()
                    diversity = self.compute_diversity(test_case.testId)
                    
                    score = self.compute_score(test_case.testId, failure_prob, diversity)
                    test_scores[test_case.testId] = score
                    
            except Exception as e:
                self.logger.error(f"Error processing test case {test_case.testId}: {str(e)}")
                continue
        
        # Calculate selection bounds
        total_tests = len(all_tests)
        min_selections = max(int(total_tests * self.min_selection_ratio), 1)
        max_selections = min(int(total_tests * self.base_selection_ratio), self.max_selections)
        
        sorted_tests = sorted(test_scores.items(), key=lambda x: x[1], reverse=True)
        
        # First pass: minimum required selections
        for test_id, score in sorted_tests[:min_selections]:
            self.selected_ids.add(test_id)
            yield competition_pb2.SelectionReply(testId=test_id)
        
        # Second pass: additional high-scoring tests
        for test_id, score in sorted_tests[min_selections:]:
            if len(self.selected_ids) >= max_selections or score < self.min_score_threshold:
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
        TransformerSelector(), server)
    
    server.add_insecure_port(f"[::]:{args.port}")
    print(f"Starting server on port {args.port}")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    main()