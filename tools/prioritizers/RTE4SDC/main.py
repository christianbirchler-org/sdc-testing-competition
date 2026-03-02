import argparse
from concurrent.futures import ThreadPoolExecutor
import grpc
import numpy as np
import onnxruntime as ort
import competition_2026_pb2
import competition_2026_pb2_grpc

EPS = 1e-8

def extract_features(road_points):
    """Extract segment-level [T,4] and global [4] features from road points.

    Returns (segment_tokens, global_features) or None if fewer than 3 points.
    """
    points = np.array(
        [[float(p[0]), float(p[1])] for p in road_points], dtype=np.float32
    )
    if points.shape[0] < 3:
        return None

    deltas = points[1:] - points[:-1]
    distance = np.linalg.norm(deltas, axis=1).astype(np.float32)
    heading = np.arctan2(deltas[:, 1], deltas[:, 0]).astype(np.float32)

    angle_change = np.zeros_like(heading, dtype=np.float32)
    if heading.size > 1:
        raw = heading[1:] - heading[:-1]
        angle_change[1:] = ((raw + np.pi) % (2 * np.pi)) - np.pi

    curvature = (angle_change / (distance + EPS)).astype(np.float32)
    delta_curvature = np.zeros_like(curvature)
    if curvature.size > 1:
        delta_curvature[1:] = curvature[1:] - curvature[:-1]

    segment_tokens = np.column_stack(
        [distance, angle_change, curvature, delta_curvature]
    ).astype(np.float32)

    total_distance = float(np.sum(distance))
    mean_curvature = float(np.mean(np.abs(curvature)))
    max_curvature = float(np.max(np.abs(curvature)))
    direct_distance = float(np.linalg.norm(points[-1] - points[0]))
    sinuosity = float(total_distance / (direct_distance + EPS))

    global_features = np.array(
        [total_distance, mean_curvature, max_curvature, sinuosity], dtype=np.float32
    )

    return segment_tokens, global_features


class RTE4SDCPrioritizer(competition_2026_pb2_grpc.CompetitionToolServicer):

    def __init__(self):
        self.session = ort.InferenceSession("rte4sdc.onnx")

    def Name(self, request, context):
        # Send the name of the tool
        return competition_2026_pb2.NameReply(name="RTE4SDC")

    def Initialize(self, request_iterator, context):
        # Consunme the stream of oracle messages
        for _ in request_iterator:
            pass
        return competition_2026_pb2.InitializationReply(ok=True)

    def Prioritize(self, request_iterator, context):
        scored = []
        for test_case in request_iterator:
            road_points = [(rp.x, rp.y) for rp in test_case.roadPoints]
            result = extract_features(road_points)
            if result is None:
                scored.append((test_case.testId, 0.0))
                continue

            segment_tokens, global_features = result

            # ONNX inputs: batch dimension (normalization is baked into the model)
            seq_input = segment_tokens[np.newaxis, :, :] # [1, T, 4]
            mask_input = np.ones((1, segment_tokens.shape[0]), dtype=np.bool_) # [1, T]
            glob_input = global_features[np.newaxis, :] # [1, 4]

            outputs = self.session.run(
                None,
                {"seq": seq_input, "valid_mask": mask_input, "glob": glob_input},
            )
            score = float(outputs[0][0])
            scored.append((test_case.testId, score))

        # Higher score = more likely to fail → sort descending
        scored.sort(key=lambda x: x[1], reverse=True)

        for test_id, _ in scored:
            yield competition_2026_pb2.PrioritizationReply(testId=test_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", required=True)
    args = parser.parse_args()
    server = grpc.server(ThreadPoolExecutor(max_workers=4))
    competition_2026_pb2_grpc.add_CompetitionToolServicer_to_server(RTE4SDCPrioritizer(), server)
    server.add_insecure_port("[::]:{}".format(args.port))
    print("RTE4SDC server starting on port {}".format(args.port))
    server.start()
    print("RTE4SDC server running")
    server.wait_for_termination()