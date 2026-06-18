import grpc
import competition_2026_pb2
import competition_2026_pb2_grpc
import concurrent.futures as fut
import argparse

import onnxruntime

from scipy.interpolate import interp1d
import numpy as np


def set_road_array_length(road_points, target_length=197):
    if road_points.shape[0] == target_length:
        return road_points
    else:
        current_indices = np.linspace(0, road_points.shape[0] - 1, road_points.shape[0])
        target_indices = np.linspace(0, road_points.shape[0] - 1, target_length)

        interpolated_x = interp1d(current_indices, road_points[:, 0], kind='cubic')(target_indices)
        interpolated_y = interp1d(current_indices, road_points[:, 1], kind='cubic')(target_indices)

        return np.column_stack((interpolated_x, interpolated_y))


def extract_road_features(road_points):
    points_diffs = np.diff(road_points, axis=0)
    angle_displacement = np.insert(np.diff(np.arctan2(points_diffs[:, 1], points_diffs[:, 0])), 0, 0)
    segment_lengths = np.linalg.norm(points_diffs, axis=1)
    curvature = angle_displacement / segment_lengths
    return np.column_stack((segment_lengths, angle_displacement, curvature))


class MiniEagle(competition_2026_pb2_grpc.CompetitionToolServicer):

    def Name(self, request, context):
        return competition_2026_pb2.NameReply(name='EAGLE-SEMBLE')

    def Initialize(self, request_iterator, context):
        return competition_2026_pb2.InitializationReply(ok=True)

    def Prioritize(self, request_iterator, context):

        ort_session = onnxruntime.InferenceSession('mini_eagle.onnx')
        onnx_input_name = ort_session.get_inputs()[0].name

        mk1_session = onnxruntime.InferenceSession('mini_eagle_mk1.onnx')
        mk1_input_name = mk1_session.get_inputs()[0].name

        norm_session = onnxruntime.InferenceSession('mini_eagle_norm.onnx')
        norm_input_name = norm_session.get_inputs()[0].name

        predictions = []

        for sdc_test_case in request_iterator:
            sdc_test_case: competition_2026_pb2.SDCTestCase = sdc_test_case

            road = np.array([[point.x, point.y] for point in sdc_test_case.roadPoints], dtype=np.float32)

            road = extract_road_features(set_road_array_length(road))

            onnx_input = {onnx_input_name: np.expand_dims(road, axis=0)}
            onnx_output = ort_session.run(None, onnx_input)
            prediction = 1.0-np.squeeze(onnx_output)

            mk1_input = {mk1_input_name: np.expand_dims(road, axis=0)}
            mk1_output = mk1_session.run(None, mk1_input)
            mk1_prediction = 1.0-np.squeeze(mk1_output)

            mean = np.array([0.9040628671646118, -0.0034271094482392073, -0.003616937668994069])
            std = np.array([0.22406981885433197, 0.3781600892543793, 0.43492621183395386])
            norm_input = {norm_input_name: np.expand_dims(((road-mean)/std).astype('float32'), axis=0)}
            norm_output = norm_session.run(None, norm_input)
            norm_prediction = 1.0-np.squeeze(norm_output)

            predictions.append((prediction, mk1_prediction, norm_prediction, sdc_test_case.testId))

        predictions.sort(key=lambda x: sum(x[:-1])/(len(x)-1))

        for test in predictions:
            yield competition_2026_pb2.PrioritizationReply(testId=test[-1])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', default='49791')
    args = parser.parse_args()

    GRPC_PORT = args.port
    GRPC_URL = '[::]:' + GRPC_PORT

    server = grpc.server(fut.ThreadPoolExecutor(max_workers=2))
    competition_2026_pb2_grpc.add_CompetitionToolServicer_to_server(MiniEagle(), server)

    server.add_insecure_port(GRPC_URL)
    print(f'start server on port {GRPC_PORT}')
    server.start()
    print('server is running')
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print('shutting down server...')
        server.stop(0)
    print('server terminated')
