import grpc
import argparse
import concurrent.futures as fut

import competition_2026_pb2
import competition_2026_pb2_grpc


class SampleTestPrioritizer(competition_2026_pb2_grpc.CompetitionToolServicer):
    """
    This is a sample test prioritizer implementing the gRPC interface.
    """
    def Name(self, request, context):
        return competition_2026_pb2.NameReply(name="random_sample_test_prioritizer")

    def Initialize(self, request_iterator, context):
        for oracle in request_iterator:
            print("initialization received testId={}".format(oracle.testCase.testId))

        return competition_2026_pb2.InitializationReply(ok=True)  # if initialization failed then ok=False

    def Prioritize(self, request_iterator, context):
        """TODO: Implement the random prioritization."""
        for sdc_test_case in request_iterator:
            sdc_test_case: competition_2026_pb2.SDCTestCase = sdc_test_case
            print("prioritization received testId={}".format(sdc_test_case.testId))
            yield competition_2026_pb2.PrioritizationReply(testId=sdc_test_case.testId)


if __name__ == "__main__":
    print("start test prioritizer")
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port")
    args = parser.parse_args()
    GRPC_PORT = args.port
    GRPC_URL = "[::]:" + GRPC_PORT

    server = grpc.server(fut.ThreadPoolExecutor(max_workers=2))
    competition_2026_pb2_grpc.add_CompetitionToolServicer_to_server(SampleTestPrioritizer(), server)

    server.add_insecure_port(GRPC_URL)
    print("start server on port {}".format(GRPC_PORT))
    server.start()
    print("server is running")
    server.wait_for_termination()
    print("server terminated")
