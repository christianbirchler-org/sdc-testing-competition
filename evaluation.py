from dataclasses import dataclass

from interfaces import TestSelector, SDCTest
from tools.sample_tool.sample_test_selector import SampleTestSelector


class MetricEvaluator:
    def __init__(self):
        pass

    def fault_to_time_ratio(self, test_suite: list[SDCTest], selection: list[bool]) -> float:
        return 0.0


@dataclass
class EvaluationReport:
    """Class holding evaluation report of a tool"""
    tool_name: str
    fault_to_time_ratio: float
    fault_to_selection_ratio: float
    processing_time: float

class ToolEvaluator:
    def __init__(self, test_suite: list[SDCTest], metric_evaluator: MetricEvaluator):
        self.test_suite = test_suite
        self.metric_evaluator = metric_evaluator

    def evaluate(self, tool: TestSelector) -> EvaluationReport:
        tool.initialize(self.test_suite)
        selection = tool.select(self.test_suite)
        fault_to_time_ratio = self.metric_evaluator.fault_to_time_ratio(self.test_suite, selection)
        return EvaluationReport(tool_name=tool.name(), fault_to_time_ratio=fault_to_time_ratio, fault_to_selection_ratio=0.0, processing_time=0.0)


if __name__ == '__main__':
    te = ToolEvaluator([SDCTest(), SDCTest()], MetricEvaluator())
    ts = SampleTestSelector()

    report = te.evaluate(ts)
    print(report)

