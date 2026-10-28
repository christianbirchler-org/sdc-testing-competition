#!python

from concurrent import futures

import grpc

import interface_2027_pb2 as messages
import interface_2027_pb2_grpc as services

# service CompetitionTool {
#   rpc Name(Empty) returns (NameReply) {}
# 
#   rpc Initialize (stream SDCTestExecution) returns (InitializationReply) {}
# 
#   rpc Minimize (stream SDCTestCase) returns (stream MinimizationReply) {}
# }

class RandomBaselineTool(services.CompetitionToolServicer):
    def Name(self, request, context):
        return messages.NameReply(name="random_baseline_tool_python")

    def Initialize(self, request_iterator, context):
        el: messages.SDCTestExecution
        for el in request_iterator:
            print(el.hasFailed)
        return messages.InitializationReply(ok=True)

    def Minimize(self, request_iterator, context):
        el: messages.SDCTestCase
        for el in request_iterator:
            print(el)


if __name__ == '__main__':
    print('\nrandom baseline with python')

    tool = RandomBaselineTool()

    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))

    services.add_CompetitionToolServicer_to_server(servicer=tool, server=grpc_server)

    grpc_server.add_insecure_port("[::]:50051")
    grpc_server.start()
    grpc_server.wait_for_termination()



