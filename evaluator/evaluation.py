from dataclasses import dataclass
import abc
import sys
import json
import grpc
import argparse
import competition_pb2_grpc
import competition_pb2


class InitializationError(Exception):
    pass


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


class EvaluationTestLoader(abc.ABC):
    @abc.abstractmethod
    def load_next_oracle(self) -> competition_pb2.Oracle:
        pass

    @abc.abstractmethod
    def load_next_test(self) -> competition_pb2.SDCTestCase:
        pass

    @abc.abstractmethod
    def load_all(self):
        pass

    @abc.abstractmethod
    def reset_index(self):
        pass

    @abc.abstractmethod
    def has_next_oracle(self) -> bool:
        pass

    @abc.abstractmethod
    def has_next_test(self) -> bool:
        pass


class SampleEvaluationTestLoader(EvaluationTestLoader):
    def __init__(self, file_path: str, training_prop: float):
        super().__init__()
        self.raw_test_cases: list = None
        with open(file_path, 'r') as fp:
            self.raw_test_cases = json.load(fp)
        self.training_prop: float = training_prop
        self.current_oracle_index = 0
        self.split_index = int(training_prop*len(self.raw_test_cases))
        self.current_test_index = self.split_index

    def load_next_oracle(self) -> competition_pb2.Oracle:
        raw_test = self.raw_test_cases[self.current_oracle_index]
        tc = competition_pb2.SDCTestCase(
            testId=raw_test['_id']['$oid'],
            roadPoints=[competition_pb2.RoadPoint(sequenceNumber=i, x=pts['x'], y=pts['y']) for i, pts in enumerate(raw_test['road_points'])]
        )
        oracle = competition_pb2.Oracle(testCase=tc, hasFailed= True if raw_test['meta_data']['test_info']['test_outcome'] == 'FAIL' else False)
        self.current_oracle_index += 1
        return oracle

    def load_next_test(self) -> competition_pb2.SDCTestCase:
        raw_test = self.raw_test_cases[self.current_test_index]
        tc = competition_pb2.SDCTestCase(
            testId=raw_test['_id']['$oid'],
            roadPoints=[competition_pb2.RoadPoint(sequenceNumber=i, x=pts['x'], y=pts['y']) for i, pts in enumerate(raw_test['road_points'])]
        )
        self.current_test_index += 1
        return tc

    def reset_index(self):
        self.current_index = 0

    def load_all(self) -> None:
        raise NotImplementedError()

    def has_next_oracle(self) -> bool:
        return self.split_index > self.current_oracle_index

    def has_next_test(self) -> bool:
        return len(self.raw_test_cases) > self.current_test_index




# iterators are used to implement gRPC streams
def _init_iterator(test_loader: EvaluationTestLoader):
    while test_loader.has_next_oracle():
        oracle: competition_pb2.Oracle = test_loader.load_next_oracle()
        yield oracle


# iterators are used to implement gRPC streams
def _test_suite_iterator(test_loader: EvaluationTestLoader):
    while test_loader.has_next_test():
        test: competition_pb2.SDCTestCase = test_loader.load_next_test()
        yield test


class ToolEvaluator:
    def __init__(self, metric_evaluator: MetricEvaluator, test_loader: EvaluationTestLoader):
        self.metric_evaluator = metric_evaluator
        self.test_loader: EvaluationTestLoader = test_loader

    def evaluate(self, stub: competition_pb2_grpc.CompetitionToolStub) -> EvaluationReport:

        # get the tool name
        name_reply: competition_pb2.NameReply = stub.Name(competition_pb2.Empty())

        # initialize the tool with training data (i.e., Oracles)
        init_resp: competition_pb2.InitializationReply = stub.Initialize(_init_iterator(test_loader=self.test_loader))
        if not init_resp.ok:
            raise InitializationError()

        # tool returns a selection of test cases
        selection_iterator = stub.Select(_test_suite_iterator(test_loader=self.test_loader))
        for test_case in selection_iterator:
            test_case: competition_pb2.SDCTestCase = test_case
            print(test_case.testId)

        return EvaluationReport(
            tool_name=name_reply.name,
            fault_to_time_ratio=self.metric_evaluator.fault_to_time_ratio(test_suite=None, selection=None),
            fault_to_selection_ratio=self.metric_evaluator.fault_to_selection_ratio(test_suite=None, selection=None),
            processing_time=self.metric_evaluator.processing_time(test_suite=None, selection=None),
            diversity=self.metric_evaluator.diversity(test_suite=None, selection=None),
        )


if __name__ == "__main__":
    # handle CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--url")
    parser.add_argument("-t", "--tests")
    args = parser.parse_args()
    if args.url:
        GRPC_URL = args.url
    else:
        print('provide url to the tool with -u/--url')
        sys.exit()

    if args.tests:
        TESTS_FILE = args.tests
    else:
        print('provide path to test cases -t/--tests')
        TESTS_FILE = "../sample_tests/sdc-test-data.json"


    # initialization of business objects
    tl = SampleEvaluationTestLoader(file_path=TESTS_FILE, training_prop=0.8)
    me = MetricEvaluator()
    te = ToolEvaluator(me, tl)

    # set up gRPC conection stub
    channel = grpc.insecure_channel(GRPC_URL)
    stub = competition_pb2_grpc.CompetitionToolStub(channel) # stub represents the tool

    # start evaluation
    report = te.evaluate(stub)
    print(report)
