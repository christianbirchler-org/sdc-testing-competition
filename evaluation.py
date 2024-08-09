from interfaces import TestSelector, SDCTest
from tools.sample_tool.sample_test_selector import SampleTestSelector


class MetricEvaluator:
    def __init__(self):
        pass

    def bug_time_ratio(self, test_suite: list[SDCTest], selection: list[bool]) -> float:
        return 0.0


class EvaluationReport:
    pass

class ToolEvaluator:
    def __init__(self, test_suite: list[SDCTest], metric_evaluator: MetricEvaluator):
        self.test_suite = test_suite
        self.metric_evaluator = metric_evaluator

    def evaluate(self, tool: TestSelector):
        tool.initialize(self.test_suite)
        selection = tool.select(self.test_suite)
        self.metric_evaluator.bug_time_ratio(self.test_suite, selection)


if __name__ == '__main__':
    print('start evaluation')
    te = ToolEvaluator([SDCTest(), SDCTest()], MetricEvaluator())
    ts = SampleTestSelector()

    te.evaluate(ts)

