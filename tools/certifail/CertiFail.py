import random
import argparse
import os
import competition_pb2_grpc
import competition_pb2
import grpc
import concurrent.futures as fut
from features import calculate_features
from rf_prediction import predict_outcome





class SampleTestSelector(competition_pb2_grpc.CompetitionToolServicer):
    """
    This is a sample test selector implementing the TestSelector interface.
    """

    def Name(self, request, context):
        return competition_pb2.NameReply(name="CertiFail")

    def Initialize(self, request_iterator, context):

        """Missing associated documentation comment in .proto file."""
        
        
        for oracle in request_iterator:
            oracle: competition_pb2.Oracle = oracle
            print("hasFailed={}\ttestId={}".format(oracle.hasFailed, oracle.testCase.testId))

        return competition_pb2.InitializationReply(ok=True)

    def Select(self, request_iterator, context):
        """bidirectional streaming for high flexibility"""
        for sdc_test_case in request_iterator:
            sdc_test_case: competition_pb2.SDCTestCase = sdc_test_case
            print("Received Test Case: testId={}".format(sdc_test_case.testId))
            #print("roadpoints={}".format(sdc_test_case.roadPoints))
            #print ("the features are ", calculate_features(sdc_test_case))
            # print ("tthe road coordinates are ", extract_coordinates(sdc_test_case))
            features = calculate_features(sdc_test_case)
            # print ("the features are ", features)


            # get the prediction outcome 
            rf_prediction_outcome = predict_outcome(features)
         
            print("Random Forest predicts", rf_prediction_outcome)


            if rf_prediction_outcome < 1:
                yield competition_pb2.SelectionReply(testId=sdc_test_case.testId)
                print("Sending Selection Reply: testId={}".format(sdc_test_case.testId))




if __name__ == "__main__":

    print("start test selector")
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port")
    args = parser.parse_args()
    GRPC_PORT = args.port
    GRPC_URL = "[::]:" + GRPC_PORT

    server = grpc.server(fut.ThreadPoolExecutor(max_workers=2))
    competition_pb2_grpc.add_CompetitionToolServicer_to_server(SampleTestSelector(), server)

    server.add_insecure_port(GRPC_URL)
    print("start server on port {}".format(GRPC_PORT))
    server.start()
    print("server is running")
    server.wait_for_termination()
    print("server terminated")
