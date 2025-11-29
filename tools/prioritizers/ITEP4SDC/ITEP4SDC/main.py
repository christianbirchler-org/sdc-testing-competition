import grpc
import argparse
import concurrent.futures as fut
import competition_2026_pb2
import competition_2026_pb2_grpc
from scipy.interpolate import interp1d

import os
import sys
import json
import joblib
import numpy as np
import onnxruntime
from datetime import datetime


class RoadCharacteristics:
    def __init__(self):
        self.scaler_length = joblib.load('scaler_length.pkl')
        self.scaler_angle = joblib.load('scaler_angle.pkl')
        self.scaler_curvature = joblib.load('scaler_curvature.pkl')
    

    def calculate_angle_changes_in_segments(self, roadpointsarray):

        dx = np.diff(roadpointsarray[:, 0])
        dy = np.diff(roadpointsarray[:, 1])

        raw_angles = np.rad2deg(np.arctan2(dy, dx))

        changes_in_angles = np.diff(raw_angles)

        changes_in_angles = (changes_in_angles + 180) % 360 - 180
        changes_in_angles = np.insert(changes_in_angles, 0, 0)

        return changes_in_angles

    def calculate_segment_lengths(self, roadpointsarray):

        diffs = np.diff(roadpointsarray, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)

        return segment_lengths

    def calculate_curvature(self, roadpointsarray):

        curvature_list = []

        for i in range(0, roadpointsarray.shape[0]-2):
            x1, y1 = roadpointsarray[i][0], roadpointsarray[i][1]
            x2, y2 = roadpointsarray[i+1][0], roadpointsarray[i+1][1]
            x3, y3 = roadpointsarray[i+2][0], roadpointsarray[i+2][1]

            a = np.sqrt((x2-x1) ** 2 + (y2-y1) ** 2)
            b = np.sqrt((x3-x2) ** 2 + (y3-y2) ** 2)
            c = np.sqrt((x3-x1) ** 2 + (y3-y1) ** 2)

            # Heron Formula
            s = 0.5 * (a+b+c)

            area_term = s * (s-a) * (s-b) * (s-c)

            if area_term <= 1e-7:
                curvature = 0.0
            else:
                area = np.sqrt(area_term)
                radius = a * b * c / (4 * area)

                cross = (x2-x1) * (y3-y1) - (y2-y1) * (x3-x1)
                sign = np.sign(cross)
                curvature = sign  * (1 / radius)

            curvature_list.append(curvature)

        curvature_list.insert(0, 0)

        return np.array(curvature_list, dtype=np.float32)

    def calculate_features(self, roadpointsarray):

        if isinstance(roadpointsarray, list):
            roadpointsarray = np.array(roadpointsarray)

        roadpointsarray = roadpointsarray.reshape(-1, 2)

        angle_changes_in_segments = self.calculate_angle_changes_in_segments(roadpointsarray)
        segment_lengths = self.calculate_segment_lengths(roadpointsarray)
        segment_curvatures = self.calculate_curvature(roadpointsarray)

        ### Scale the features ###
        ###### !!! burada değişiklik yaptım shape ve flatten'lar yeni. kontrol edilecek.

        angle_changes_in_segments = self.scaler_angle.transform(angle_changes_in_segments.reshape(-1, 1)).flatten()
        segment_lengths = self.scaler_length.transform(segment_lengths.reshape(-1, 1)).flatten()
        segment_curvatures = self.scaler_curvature.transform(segment_curvatures.reshape(-1, 1)).flatten()

        return angle_changes_in_segments, segment_lengths, segment_curvatures

    def get_feature_list(self, roadpointsarray):
        angle_changes_in_segments, segment_lengths, segment_curvatures = self.calculate_features(roadpointsarray)
        return segment_lengths, angle_changes_in_segments, segment_curvatures

    def get_feature_vector(self, roadpointsarray):

        angle_changes_in_segments, segment_lengths, segment_curvatures = self.calculate_features(roadpointsarray)
        feature_vector = np.column_stack((segment_lengths, angle_changes_in_segments, segment_curvatures)).reshape(-1, 3)
        return feature_vector

    @staticmethod
    def adjust_array_size(road_list, target_size=197):

        if isinstance(road_list, list):
            road_list = np.array(road_list).reshape(-1, 2) # (index, (x,y))

        road_list = road_list.reshape(-1, 2)

        if road_list.shape[0] == target_size:
            return road_list

        else:
            current_index = np.linspace(0, road_list.shape[0]-1, num=road_list.shape[0])

            interpolator_x = interp1d(current_index, road_list[:, 0], kind='linear')
            interpolator_y = interp1d(current_index, road_list[:, 1], kind='linear')

            target_index = np.linspace(0, road_list.shape[0]-1, num=target_size)

            interpolated_x = interpolator_x(target_index)
            interpolated_y = interpolator_y(target_index)

            interpolated_road_list = np.column_stack((interpolated_x, interpolated_y)).reshape(-1, 2)

            return interpolated_road_list
        

class ITEP4SDC(competition_2026_pb2_grpc.CompetitionToolServicer):
    
    def __init__(self):
        
        ### LOAD ONNX MODEL ####
        self.onnx_session = onnxruntime.InferenceSession('itep4sdc.onnx')
        self.onnx_input_name = self.onnx_session.get_inputs()[0].name
        self.onnx_output_name = self.onnx_session.get_outputs()[0].name
        self.rc = RoadCharacteristics()

    def Name(self, request, context):
        return competition_2026_pb2.NameReply(name="ITEP4SDC")
    
    def Initialize(self, request_iterator, context):
        #for oracle in request_iterator:
            #print("initialization received testId={}".format(oracle.testCase.testId))
        return competition_2026_pb2.InitializationReply(ok=True)
    
    def Prioritize(self, request_iterator, context):
        
        scored_tests = []

        for sdc_test_case in request_iterator:
            sdc_test_case: competition_2026_pb2.SDCTestCase = sdc_test_case
            
            roadPointsArray = np.array([[point.x, point.y] for point in sdc_test_case.roadPoints], dtype=np.float32)

            roadPointsArray = self.rc.adjust_array_size(road_list=roadPointsArray)
            feature_vector = self.rc.get_feature_vector(roadpointsarray=roadPointsArray) # output: (-1, 3)

            onnx_input_data = np.expand_dims(feature_vector, axis=0)
            onnx_prediction_result = self.onnx_session.run([self.onnx_output_name], {self.onnx_input_name: onnx_input_data})
            onnx_prediction_result = float(np.squeeze(onnx_prediction_result)) # (fail: 0, pass: 1)

            scored_tests.append((onnx_prediction_result, sdc_test_case.testId))

        def sorting_function(inputTuple):
            return inputTuple[0]

        scored_tests.sort(key=sorting_function)
        
        for each_test in scored_tests:

            _, test_id = each_test

            yield competition_2026_pb2.PrioritizationReply(testId=test_id)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', required=True, help='The port number to run th gRPC server on.')
    args = parser.parse_args()

    GRPC_PORT = args.port
    GRPC_URL = "[::]:" + GRPC_PORT

    server = grpc.server(fut.ThreadPoolExecutor(max_workers=4))
    competition_2026_pb2_grpc.add_CompetitionToolServicer_to_server(ITEP4SDC(), server)

    server.add_insecure_port(GRPC_URL)
    print("start server on port {}".format(GRPC_PORT))

    server.start()
    print("server is running")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("server shutting down...")
        server.stop(0)
    print("server terminated")
