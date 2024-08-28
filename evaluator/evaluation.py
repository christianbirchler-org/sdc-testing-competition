from dataclasses import dataclass
import time
import abc
import sys
import json
import grpc
import argparse
import competition_pb2_grpc
import competition_pb2


class InitializationError(Exception):
    pass


class TestDoesNotExistError(Exception):
    pass


@dataclass
class TestDetails:
    test_id: str
    hasFailed: bool
    sim_time: float
    road_points: list[tuple[float, float]]


class MetricEvaluator:
    def __init__(self):
        pass

    def time_to_fault_ratio(self, test_suite: list[TestDetails], selection: list[str]) -> float:
        """
        ratio between the required simulation time and the number of detected faults
        """
        nr_detected_faults = 0
        total_simulation_time = 0.0

        test_oracle_mapping = {test.test_id: test for test in test_suite}

        for test_id in selection:
            if test_id not in test_oracle_mapping.keys(): raise TestDoesNotExistError()
            test = test_oracle_mapping[test_id]
            if test.hasFailed: nr_detected_faults += 1
            total_simulation_time += test.sim_time

        return total_simulation_time / nr_detected_faults

    def fault_to_selection_ratio(self, test_suite: list[TestDetails], selection: list[str]) -> float:
        """
        ratio between the number of detected faults and the number of selected tests
        """
        nr_detected_faults = 0

        test_oracle_mapping = {test.test_id: test for test in test_suite}

        for test_id in selection:
            if test_id not in test_oracle_mapping.keys(): raise TestDoesNotExistError()
            test = test_oracle_mapping[test_id]
            if test.hasFailed: nr_detected_faults += 1

        return nr_detected_faults / len(test_suite)


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
    time_to_initialize: float
    time_to_select_tests: float
    time_to_fault_ratio: float
    fault_to_selection_ratio: float
    diversity: float


class EvaluationTestLoader(abc.ABC):
    @abc.abstractmethod
    def get_test_details_lst(self) -> list[TestDetails]:
        pass

    @abc.abstractmethod
    def get_test_details_dict(self) -> dict:
        pass


def _make_test_details_list(raw_test_cases) -> list[TestDetails]:
    test_details = []
    for raw_test in raw_test_cases:
        test_id = raw_test['_id']['$oid']
        road_points = [(pts['x'], pts['y']) for pts in raw_test['road_points']]
        hasFailed = raw_test['meta_data']['test_info']['test_outcome'] == "FAIL"
        sim_time = raw_test['meta_data']['test_info']['test_duration']
        test_details.append(TestDetails(test_id=test_id, hasFailed=hasFailed, sim_time=sim_time, road_points=road_points))
    return test_details


class SampleEvaluationTestLoader(EvaluationTestLoader):
    def __init__(self, file_path: str, training_prop: float):
        super().__init__()
        self.raw_test_cases: list = None
        with open(file_path, 'r') as fp:
            self.raw_test_cases = json.load(fp)

        self.test_details_lst = _make_test_details_list(self.raw_test_cases)
        self.test_details_dict = {test_details.test_id: test_details for test_details in self.test_details_lst}
        self.training_prop: float = training_prop
        self.current_oracle_index = 0
        self.split_index = int(training_prop*len(self.raw_test_cases))
        self.current_test_index = self.split_index


    def get_test_details_lst(self) -> list[TestDetails]:
        return self.test_details_lst
        self.test_details_dict = {test_details.test_id: test_details for test_details in self.test_details_lst}

    def get_test_details_dict(self) -> dict:
        return self.test_details_dict

    def load(self, test_id: str) -> TestDetails:
        return self.test_details_dict[test_id]

    def get_test_ids(self):
        return self.test_details_dict.keys()



# generators/iterators are used to implement gRPC streams
def _init_iterator(train_set: list[TestDetails]):
    for test_detail in train_set:
        road_points = [competition_pb2.RoadPoint(sequenceNumber=i, x=pts[0], y=pts[1]) for i, pts in enumerate(test_detail.road_points)]
        test_case = competition_pb2.SDCTestCase(testId=test_detail.test_id, roadPoints=road_points)
        oracle: competition_pb2.Oracle = competition_pb2.Oracle(testCase=test_case, hasFailed=test_detail.hasFailed)
        yield oracle


# generators/iterators are used to implement gRPC streams
def _test_suite_iterator(test_set: list[TestDetails]):
    for test_detail in test_set:
        road_points = [competition_pb2.RoadPoint(sequenceNumber=i, x=pts[0], y=pts[1]) for i, pts in enumerate(test_detail.road_points)]
        test_case = competition_pb2.SDCTestCase(testId=test_detail.test_id, roadPoints=road_points)
        yield test_case


class ToolEvaluator:
    def __init__(self, metric_evaluator: MetricEvaluator, test_loader: EvaluationTestLoader, train_proportion=0.8):
        self.metric_evaluator = metric_evaluator
        self.test_loader: EvaluationTestLoader = test_loader
        self.train_proportion = train_proportion
        self.train_max_idx = int(len(test_loader.get_test_details_lst()) * train_proportion)
        self.train_set: list[TestDetails] = test_loader.get_test_details_lst()[:self.train_max_idx]
        self.test_set: list[TestDetails] = test_loader.get_test_details_lst()[self.train_max_idx:]

    def evaluate(self, stub: competition_pb2_grpc.CompetitionToolStub) -> EvaluationReport:

        # get the tool name
        name_reply: competition_pb2.NameReply = stub.Name(competition_pb2.Empty())

        # initialize the tool with training data (i.e., Oracles)
        init_start_time = time.time()
        init_resp: competition_pb2.InitializationReply = stub.Initialize(_init_iterator(train_set=self.train_set))
        if not init_resp.ok:
            raise InitializationError()
        init_end_time = time.time()

        # tool returns a selection of test cases
        selection_start_time = time.time()
        selection_iterator = stub.Select(_test_suite_iterator(test_set=self.test_set))
        selection = []
        for test_case in selection_iterator:
            test_case: competition_pb2.SDCTestCase = test_case
            selection.append(test_case.testId)
            print(test_case.testId)
        selection_end_time = time.time()

        return EvaluationReport(
            time_to_initialize=(init_end_time-init_start_time),
            time_to_select_tests=(selection_end_time-selection_start_time),
            tool_name=name_reply.name,
            time_to_fault_ratio=self.metric_evaluator.time_to_fault_ratio(test_suite=self.test_set, selection=selection),
            fault_to_selection_ratio=self.metric_evaluator.fault_to_selection_ratio(test_suite=self.test_set, selection=selection),
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
