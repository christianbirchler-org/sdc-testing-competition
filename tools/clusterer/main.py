import argparse
from concurrent import futures

import grpc
import pb.competition_pb2_grpc as pb_grpc
from tool.selector_features import FeaturesSelector

if __name__ == "__main__":
    print("start test selector")
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port")
    args = parser.parse_args()
    GRPC_PORT = args.port
    GRPC_URL = "[::]:" + GRPC_PORT

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    # pb_grpc.add_CompetitionToolServicer_to_server(RandomSelector(), server)
    # pb_grpc.add_CompetitionToolServicer_to_server(DTWDistanceSelector(), server)
    pb_grpc.add_CompetitionToolServicer_to_server(FeaturesSelector(), server)

    server.add_insecure_port(GRPC_URL)
    print("start server on port {}".format(GRPC_PORT))
    server.start()
    print("server is running")
    server.wait_for_termination()
    print("server terminated")
