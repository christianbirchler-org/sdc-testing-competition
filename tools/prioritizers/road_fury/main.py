"""
RoadFury — SDC Test Prioritizer for ICST/SBFT 2026 Competition

A novel test prioritization tool for simulation-based testing of self-driving cars.
Uses a Transformer Encoder with Stochastic Weight Averaging (SWA) to predict
test failures directly from 10-channel road geometry features.

Architecture:
  - 10-channel sequential feature extraction (curvature, heading, jerk, etc.)
  - Transformer Encoder (d=128, 4 layers, 8 heads) with [CLS] token
  - Pre-LN (norm_first) for stable training
  - SWA for flatter minima and better generalization
  - Trained on 36K SensoDat labeled tests (APFD=0.8042 ± 0.0120)

gRPC interface: tools/prioritizers/interface_2026.proto
"""

import os
import sys
import argparse
import logging
import warnings
import concurrent.futures as fut

import grpc
import numpy as np
import torch
import torch.nn as nn

import competition_2026_pb2
import competition_2026_pb2_grpc
from features import compute_features, NUM_FEATURES, TARGET_SEQ_LEN

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger('RoadFury')


class RoadTransformer(nn.Module):
    """
    Transformer Encoder for road geometry classification.

    Input: [B, 10, 197] → permute → [B, 197, 10]
    → Linear projection to d_model
    → Prepend learnable [CLS] token + positional embedding
    → Pre-LN TransformerEncoder (4 layers, 8 heads)
    → CLS-only pooling → MLP classifier → logit
    """
    def __init__(self, in_channels=10, seq_len=197, d_model=128,
                 nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: [B, C, L] → [B, L, C]
        x = x.permute(0, 2, 1)
        B, L, C = x.shape
        x = self.input_proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embedding[:, :L + 1, :]
        x = self.transformer(x)
        return self.classifier(x[:, 0, :]).squeeze(-1)


class RoadFuryPrioritizer(competition_2026_pb2_grpc.CompetitionToolServicer):
    """
    RoadFury test prioritizer using Transformer + SWA.
    """

    TOOL_NAME = "RoadFury"

    def __init__(self, model_path: str = 'roadfury_best.pt'):
        """Load pre-trained Transformer+SWA model."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Device: {self.device}")

        if not os.path.exists(model_path):
            logger.error(f"Model file {model_path} not found!")
            self.model = None
            self.means = None
            self.stds = None
            return

        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        self.model = RoadTransformer(
            in_channels=NUM_FEATURES,
            seq_len=TARGET_SEQ_LEN
        ).to(self.device)
        self.model.load_state_dict(checkpoint['state'])
        self.model.eval()

        self.means = np.array(checkpoint['means'], dtype=np.float32)
        self.stds = np.array(checkpoint['stds'], dtype=np.float32)

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model loaded: {n_params:,} params, {NUM_FEATURES} channels")
        logger.info(f"Checkpoint APFD: {checkpoint.get('apfd', 'N/A')}")

    def Name(self, request, context):
        """Return tool name."""
        logger.info("Name() called")
        return competition_2026_pb2.NameReply(name=self.TOOL_NAME)

    def Initialize(self, request_iterator, context):
        """
        Receive historical test results (Oracle stream).
        Our model is pre-trained offline on 36K SensoDat tests,
        so we acknowledge the oracle data without online retraining.
        """
        logger.info("Initialize() started — receiving oracle data")
        count = 0
        fail_count = 0
        for oracle in request_iterator:
            count += 1
            if oracle.hasFailed:
                fail_count += 1
        logger.info(f"Initialize() done: {count} tests ({fail_count} FAIL)")
        return competition_2026_pb2.InitializationReply(ok=True)

    def Prioritize(self, request_iterator, context):
        """
        Receive test cases via gRPC stream, score with Transformer,
        yield test IDs in descending failure-probability order.
        """
        logger.info("Prioritize() started")

        test_ids = []
        features_list = []

        for sdc_test_case in request_iterator:
            test_id = sdc_test_case.testId
            try:
                road_points = sdc_test_case.roadPoints
                if len(road_points) < 3:
                    logger.warning(f"Test {test_id}: too few road points")
                    features = np.zeros((TARGET_SEQ_LEN, NUM_FEATURES), dtype=np.float32)
                else:
                    features = compute_features(road_points)
                test_ids.append(test_id)
                features_list.append(features)
            except Exception as e:
                logger.warning(f"Feature error for {test_id}: {e}")
                test_ids.append(test_id)
                features_list.append(np.zeros((TARGET_SEQ_LEN, NUM_FEATURES), dtype=np.float32))

        n_tests = len(test_ids)
        logger.info(f"Received {n_tests} test cases")

        if n_tests == 0:
            logger.warning("No test cases received!")
            return

        # Score with model
        if self.model is not None and self.means is not None:
            try:
                X = np.array(features_list, dtype=np.float32)  # [N, 197, 10]
                # Normalize
                X = (X - self.means) / self.stds
                # [N, 197, 10] → [N, 10, 197]
                X_tensor = torch.tensor(X, dtype=torch.float32).transpose(1, 2).to(self.device)

                with torch.no_grad():
                    logits = self.model(X_tensor)
                    scores = torch.sigmoid(logits).cpu().numpy()

                logger.info("Transformer scoring complete")
            except Exception as e:
                logger.error(f"Prediction failed: {e}. Falling back to random.")
                scores = np.random.rand(n_tests)
        else:
            logger.warning("Model not available — random fallback")
            scores = np.random.rand(n_tests)

        # Sort descending (highest P(FAIL) first)
        order = np.argsort(-scores)

        if n_tests >= 3:
            logger.info(f"Top-3 scores: {scores[order[0]]:.4f}, "
                        f"{scores[order[1]]:.4f}, {scores[order[2]]:.4f}")

        for idx in order:
            yield competition_2026_pb2.PrioritizationReply(testId=test_ids[idx])

        logger.info(f"Yielded {n_tests} prioritized test IDs")
        
        logger.info(f"Yielded {n_tests} prioritized test IDs")


def serve(port: str):
    """Start the gRPC server."""
    server = grpc.server(fut.ThreadPoolExecutor(max_workers=4))
    prioritizer = RoadFuryPrioritizer()
    competition_2026_pb2_grpc.add_CompetitionToolServicer_to_server(prioritizer, server)

    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)

    logger.info(f"Starting RoadFury gRPC server on port {port}")
    server.start()
    logger.info("Server is running. Waiting for requests...")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.stop(0)

    logger.info("Server terminated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RoadFury SDC Test Prioritizer (Transformer+SWA)')
    parser.add_argument('-p', '--port', required=True, help='gRPC server port')
    args = parser.parse_args()
    serve(args.port)
