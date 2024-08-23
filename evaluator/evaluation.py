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
        yield competition_pb2.SDCTestCase(testId=str(i))

class ToolEvaluator:
    def __init__(
        self,
        grpc_url: str,
        metric_evaluator: MetricEvaluator,
    ):
        self.grpc_url = grpc_url
        self.metric_evaluator = metric_evaluator
        self.channel = grpc.insecure_channel(grpc_url)
        self.stub = competition_pb2_grpc.CompetitionToolStub(self.channel)

    def evaluate(self) -> EvaluationReport:

        self.stub.Initialize(init_iterator())



        fault_to_time_ratio = 0.0
        fault_to_selection_ratio = 0.0
        processing_time = 0.0
        diversity = 0.0

        return EvaluationReport(
            tool_name="someName",
            fault_to_time_ratio=fault_to_time_ratio,
            fault_to_selection_ratio=fault_to_selection_ratio,
            processing_time=processing_time,
            diversity=diversity,
        )


if __name__ == "__main__":
    GRPC_URL = "localhost:50051"
    tl = SampleTestLoader()

    te = ToolEvaluator(GRPC_URL, MetricEvaluator())

    report = te.evaluate()
    print(report)
