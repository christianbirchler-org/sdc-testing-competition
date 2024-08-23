import random
import competition_pb2_grpc
import competition_pb2


class SampleTestSelector(competition_pb2_grpc.CompetitionToolServicer):
    """
    This is a sample test selector implementing the TestSelector interface.
    """

    def Initialize(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        for sdc_test_case in request_iterator:
            print("received: %s".format(sdc_test_case))
        return competition_pb2.InitializationReply()

    def Select(self, request_iterator, context):
        """bidirectional streaming for high flexibility"""
        for sdc_test_case in request_iterator:
            sdc_test_case: competition_pb2.SDCTestCase = sdc_test_case
            print("received: %s".format(sdc_test_case))
            if random.randint(0, 1) < 1:
                yield competition_pb2.SelectionReply(testId=sdc_test_case.testId)
