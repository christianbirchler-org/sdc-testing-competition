import random
import competition_pb2_grpc



class SampleTestSelector(competition_pb2_grpc.CompetitionToolServicer):
    """
    This is a sample test selector implementing the TestSelector interface.
    """
    def Initialize(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Select(self, request_iterator, context):
        """bidirectional streaming for high flexibility
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')
