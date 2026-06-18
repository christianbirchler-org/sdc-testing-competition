import argparse
import concurrent.futures as fut
from pathlib import Path

import grpc
import joblib
import numpy as np

import competition_2026_pb2
import competition_2026_pb2_grpc
from features import extract_all_features

MODEL_FILE = Path(__file__).with_name("road_complexity_model.joblib")


class Road_Full_Of_Features_ML_Prioritizer(competition_2026_pb2_grpc.CompetitionToolServicer):
    def __init__(self):
        self.model = None
        self.feature_keys = []
        self.using_fallback = True
        self.loaded_model_name = "unknown"

        try:
            payload = joblib.load(MODEL_FILE)
            self.model = payload["model"]
            self.feature_keys = list(payload["feature_keys"])
            self.loaded_model_name = str(payload.get("model_name", "unknown"))
            self.using_fallback = False
            print(f"Loaded model: {payload.get('model_name', 'unknown')} ({len(self.feature_keys)} features)")
        except Exception as ex:
            print(f"Failed to load model")
            self.model = None
            self.using_fallback = True

    def Name(self, request, context):
        if self.using_fallback:
            return competition_2026_pb2.NameReply(name="road_full_of_features")
        return competition_2026_pb2.NameReply(name=f"road_full_of_features_{self.loaded_model_name}")

    def _extract_matrix(self, tests):
        rows = []
        for test in tests:
            points = list(test.roadPoints)
            feats = extract_all_features(points)
            rows.append([feats.get(k, 0.0) for k in self.feature_keys])
        return np.asarray(rows, dtype=np.float64)

    def _score(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x)[:, 1]

    def Initialize(self, request_iterator, context):
        return competition_2026_pb2.InitializationReply(ok=True)

    def Prioritize(self, request_iterator, context):
        tests = list(request_iterator)
        if not tests:
            return

        x = self._extract_matrix(tests)
        scores = self._score(x)

        pairs = [(tests[i].testId, float(scores[i])) for i in range(len(tests))]
        pairs.sort(key=lambda item: item[1], reverse=True)

        for test_id, _ in pairs:
            yield competition_2026_pb2.PrioritizationReply(testId=test_id)

# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", default="50051")
    args = parser.parse_args()
    
    server = grpc.server(fut.ThreadPoolExecutor(max_workers=4))
    competition_2026_pb2_grpc.add_CompetitionToolServicer_to_server(Road_Full_Of_Features_ML_Prioritizer(), server)
    server.add_insecure_port(f"[::]:{args.port}")
    print(f"Road_Full_Of_Features_ML_Prioritizer Server active on port {args.port}")
    server.start()
    server.wait_for_termination()