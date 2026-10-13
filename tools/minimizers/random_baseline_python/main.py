#!python

from concurrent import futures

import grpc

import interface_2027_pb2 as messages
import interface_2027_pb2_grpc as services


class RandomBaselineTool(services.CompetitionToolServicer):
    def Name(self, request, context):
        return messages.NameReply(name="random_baseline_tool_python")

    def Initialize(self, request_iterator, context):
        pass

    def Minimize(self, request_iterator, context):
        pass



if __name__ == '__main__':
    print('\nrandom baseline with python')

    tool = RandomBaselineTool()

    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))

    services.add_CompetitionToolServicer_to_server(servicer=tool, server=grpc_server)

    grpc_server.add_insecure_port("[::]:50051")
    grpc_server.start()
    grpc_server.wait_for_termination()



