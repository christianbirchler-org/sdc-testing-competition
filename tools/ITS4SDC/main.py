import argparse
import grpc
import concurrent.futures as fut
import onnxruntime
import numpy as np
import competition_pb2
import competition_pb2_grpc
from scipy.interpolate import interp1d

def adjust_array_size(array, target_size=197):
    
    if array.shape[0] == target_size:
        return array
    
    elif array.shape[0] > target_size:
        indices = np.linspace(0, array.shape[0] - 1, target_size, dtype=int)
        return array[indices]
    
    
    else:
        current_indices = np.linspace(0, array.shape[0] - 1, array.shape[0])
        target_indices = np.linspace(0, array.shape[0] - 1, target_size)

        interpolator_x = interp1d(current_indices, array[:, 0], kind='linear')
        interpolator_y = interp1d(current_indices, array[:, 1], kind='linear')

        interpolated_x = interpolator_x(target_indices)
        interpolated_y = interpolator_y(target_indices)

        return np.column_stack((interpolated_x, interpolated_y))
        
class Its4Sdc(competition_pb2_grpc.CompetitionToolServicer):
    def Name(self, request, context):
        return competition_pb2.NameReply(name="ITS4SDC")

    def Initialize(self, request_iterator, context):
        for oracle in request_iterator:
            oracle: competition_pb2.Oracle = oracle
        return competition_pb2.InitializationReply(ok=True)

    def Select(self, request_iterator, context):
        onnxrunner_session = onnxruntime.InferenceSession("its4sdc.onnx")

        for sdc_test_case in request_iterator:
            sdc_test_case: competition_pb2.SDCTestCase = sdc_test_case
            
            roadPointsArray = np.array([[point.x, point.y] for point in sdc_test_case.roadPoints], dtype=np.float32)
               
            roadPointsArray = adjust_array_size(array=roadPointsArray)
            
            dx = roadPointsArray[1:, 0] - roadPointsArray[:-1, 0]
            dy = roadPointsArray[1:, 1] - roadPointsArray[:-1, 1]

            raw_angles = np.degrees(np.arctan2(dy, dx))

            segment_angles = np.zeros_like(raw_angles)
            segment_angles[1:] = np.diff(raw_angles)
            
            _differences = roadPointsArray[1:] - roadPointsArray[:-1]
            segment_lengths = np.linalg.norm(_differences, axis=1)

            feature_input_data = np.column_stack((segment_angles, segment_lengths)).astype(np.float32)
            feature_input_data = feature_input_data.reshape(1, -1, 2)

            prediction = onnxrunner_session.run(None, {onnxrunner_session.get_inputs()[0].name: feature_input_data})

            if prediction[0][0][0] < 0.5:
                yield competition_pb2.SelectionReply(testId=sdc_test_case.testId)
                
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", default="50051")
    args = parser.parse_args()
    GRPC_PORT = args.port
    GRPC_URL = "[::]:" + GRPC_PORT

    server = grpc.server(fut.ThreadPoolExecutor(max_workers=4))
    competition_pb2_grpc.add_CompetitionToolServicer_to_server(Its4Sdc(), server)
    server.add_insecure_port(GRPC_URL)
    server.start()
    server.wait_for_termination()
