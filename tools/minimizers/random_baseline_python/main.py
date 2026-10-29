#!python

from concurrent import futures
import random

import grpc

import interface_2027_pb2 as messages
import interface_2027_pb2_grpc as services


class RandomBaselineTool(services.CompetitionToolServicer):
    def Name(self, request, context):
        return messages.NameReply(name="random_baseline_tool_python")

    def Initialize(self, request_iterator, context):
        el: messages.SDCTestExecution
        for el in request_iterator:
            print(el.hasFailed)
        return messages.InitializationReply(ok=True)

    def Minimize(self, request, context):
        test_case: messages.SDCTestCase = request
        number_of_points = len(test_case.roadPoints)
        guess = random.randint(-1, number_of_points-1)
        return messages.MinimizationReply(testId=test_case.testId, roadPoint=guess)


if __name__ == '__main__':
    print('\nrandom baseline with python')

    tool = RandomBaselineTool()

    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))

    services.add_CompetitionToolServicer_to_server(servicer=tool, server=grpc_server)

    grpc_server.add_insecure_port("[::]:50051")
    grpc_server.start()
    grpc_server.wait_for_termination()



