from dataclasses import dataclass
import grpc
import competition_pb2_grpc
import competition_pb2


class MetricEvaluator:
    def __init__(self):
        pass

    def fault_to_time_ratio(self, test_suite, selection) -> float:
        """
        ratio between the number of detected faults and the required simulation time
        """
        # TODO
        return 0.0

    def fault_to_selection_ratio(self, test_suite, selection) -> float:
        """
        ratio between the number of detected faults and the number of selected tests
        """
        # TODO
        return 0.0

    def processing_time(self, test_suite, selection) -> float:
        """
        overall time to compute the selection
        """
        # TODO
        return 0.0

    def diversity(self, test_suite, selection) -> float:
        """
        diversity of the selected test cases
        """
        # TODO
        return 0.0


@dataclass
class EvaluationReport:
    """Class holding evaluation metrics of a tool"""

    tool_name: str
    fault_to_time_ratio: float
    fault_to_selection_ratio: float
    processing_time: float
    diversity: float


class SampleTestLoader:
    def load_next(self):
        pass

    def load_all(self):
        pass

    def has_next(self):
        pass


def init_iterator():
    for i in range(10):
        yield competition_pb2.SDCTestCase(testId="I" + str(i))


def test_suite_iterator():
    for i in range(7):
        yield competition_pb2.SDCTestCase(testId="E" + str(i))


class ToolEvaluator:
    def __init__(
        self,
        metric_evaluator: MetricEvaluator,
    ):
        self.metric_evaluator = metric_evaluator

    def evaluate(self, stub: competition_pb2_grpc.CompetitionToolStub) -> EvaluationReport:

        name_reply: competition_pb2.NameReply = stub.Name(competition_pb2.Empty())

        stub.Initialize(init_iterator())
        selection_iterator = stub.Select(test_suite_iterator())

        for test_case in selection_iterator:
            print(test_case)

        fault_to_time_ratio = 0.0
        fault_to_selection_ratio = 0.0
        processing_time = 0.0
        diversity = 0.0

        return EvaluationReport(
            tool_name=name_reply.name,
            fault_to_time_ratio=fault_to_time_ratio,
            fault_to_selection_ratio=fault_to_selection_ratio,
            processing_time=processing_time,
            diversity=diversity,
        )


if __name__ == "__main__":
    GRPC_URL = "localhost:50051"

    channel = grpc.insecure_channel(GRPC_URL)
    stub = competition_pb2_grpc.CompetitionToolStub(channel)
    te = ToolEvaluator(MetricEvaluator())

    report = te.evaluate(stub)
    print(report)
