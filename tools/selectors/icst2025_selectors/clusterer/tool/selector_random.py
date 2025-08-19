import random

import pb.competition_pb2 as pb
import pb.competition_pb2_grpc as pb_grpc


class RandomSelector(pb_grpc.CompetitionToolServicer):
    def Name(self, request, context):
        return pb.NameReply(name="random_selector")

    def Initialize(self, request_iterator, context):
        for oracle in request_iterator:
            oracle: pb.Oracle = oracle
            # print("hasFailed={}\ttestId={}".format(oracle.hasFailed, oracle.testCase.testId))

        return pb.InitializationReply(ok=True)

    def Select(self, request_iterator, context):
        for sdc_test_case in request_iterator:
            sdc_test_case: pb.SDCTestCase = sdc_test_case
            if random.randint(0, 1) < 1:
                yield pb.SelectionReply(testId=sdc_test_case.testId)
