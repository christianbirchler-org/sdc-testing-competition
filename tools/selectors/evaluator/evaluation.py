from dotenv import load_dotenv
from dataclasses import dataclass
from sklearn.metrics.pairwise import pairwise_distances
from pymongo import MongoClient
from pymongo.collection import Collection
from pathlib import Path
import numpy as np
import os
import shapely
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


def _curvature_profile(test_detail: TestDetails) -> list[float]:
    """
    Compute the curvature for every meter of the road.

    The following website was used as a reference: https://de.wikipedia.org/wiki/Kr%C3%BCmmung
    """
    #print("compute curvature profile")
    road_shape = shapely.LineString(test_detail.road_points)

    delta_s = 2  # 10 meters

    curvature_profile = np.zeros(int(road_shape.length)) # we want the curvature for every meter
    for s in range(len(curvature_profile)):
        #s = (i+1)*delta_s

        # ignore the edge cases close to the ends of the road
        if (s < delta_s/2) or (s > road_shape.length-delta_s/2):
            continue


        pt_q: shapely.Point = road_shape.interpolate(s-delta_s, normalized=False)
        pt_r: shapely.Point = road_shape.interpolate(s-delta_s/2, normalized=False)

        pt_s: shapely.Point = road_shape.interpolate(s, normalized=False)

        pt_t: shapely.Point = road_shape.interpolate(s+delta_s/2, normalized=False)
        pt_u: shapely.Point = road_shape.interpolate(s+delta_s, normalized=False)

        tangent_r_vec = np.array((pt_s.x-pt_q.x, pt_s.y-pt_q.y))
        tangent_t_vec = np.array((pt_u.x-pt_s.x, pt_u.y-pt_s.y))

        cos_phi = np.dot(tangent_r_vec, tangent_t_vec)/(np.linalg.norm(tangent_r_vec)*np.linalg.norm(tangent_t_vec))
        phi = np.arccos(cos_phi)

        kappa = phi/delta_s
        if np.isnan(kappa):
            continue

        curvature_profile[s] = kappa

    return curvature_profile


class MetricEvaluator:
    """Computation of all evaluation metrics."""

    def __init__(self):
        pass

    def time_to_fault_ratio(self, test_suite: list[TestDetails], selection: list[str]) -> float:
        """
        Ratio between the required simulation time and the number of detected faults.
        """
        nr_detected_faults = 0
        total_simulation_time = 0.0

        test_oracle_mapping = {test.test_id: test for test in test_suite}

        for test_id in selection:
            if test_id not in test_oracle_mapping.keys():
                raise TestDoesNotExistError()
            test = test_oracle_mapping[test_id]
            if test.hasFailed:
                nr_detected_faults += 1
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

        return nr_detected_faults / len(selection)


    def diversity(self, test_suite: list[TestDetails], selection: list[str]) -> float:
        """
        TODO: Diversity of the selected test cases
        Current implementation is not final!!!
        We might consider different definitions of diversity.
        """
        curvature_profiles_stats = []

        for test_detail in test_suite:
            if test_detail.test_id in selection:
                profile = _curvature_profile(test_detail)
                profile_stat = {
                    'k_mean': np.mean(profile),
                    'k_std': np.std(profile)
                }
                pt = [profile_stat['k_mean'], profile_stat['k_std']]

                curvature_profiles_stats.append(pt)

        curvature_stats_distances = pairwise_distances(curvature_profiles_stats, curvature_profiles_stats)

        return float(np.mean(curvature_stats_distances[0, :]))


@dataclass
class EvaluationReport:
    """This class holding evaluation metrics of a tool."""

    tool_name: str
    benchmark: str
    test_suite_cnt: int
    selection_cnt: int
    time_to_initialize: float
    time_to_select_tests: float
    time_to_fault_ratio: float
    fault_to_selection_ratio: float
    diversity: float


def save_csv(report: EvaluationReport, file_path: Path):
    """Persist the evaluation report."""
    output = ''
    output += str(report.tool_name)
    output += ','
    output += str(report.benchmark)
    output += ','
    output += str(report.test_suite_cnt)
    output += ','
    output += str(report.selection_cnt)
    output += ','
    output += str(report.time_to_initialize)
    output += ','
    output += str(report.time_to_select_tests)
    output += ','
    output += str(report.time_to_fault_ratio)
    output += ','
    output += str(report.fault_to_selection_ratio)
    output += ','
    output += str(report.diversity)
    output += '\n'

    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'a') as fp:
        fp.write(output)


class EvaluationTestLoader(abc.ABC):
    """Abstract test loader for loading the evaluation data."""

    @abc.abstractmethod
    def benchmark(self) -> str:
        """Return the name of the benchmark."""
        pass

    @abc.abstractmethod
    def get_test_details_lst(self) -> list[TestDetails]:
        """Return list of all test cases and their oracle."""
        pass

    @abc.abstractmethod
    def get_test_details_dict(self) -> dict:
        """Get test cases by their hash id."""
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
    """Sample test loader for the provided data."""

    def __init__(self, file_path: str, training_prop: float):
        """Initialize test loader with path to dataset."""
        super().__init__()
        self.file_path = file_path
        self.raw_test_cases: list = None
        with open(file_path, 'r') as fp:
            self.raw_test_cases = json.load(fp)

        self.test_details_lst = _make_test_details_list(self.raw_test_cases)
        self.test_details_dict = {test_details.test_id: test_details for test_details in self.test_details_lst}
        self.training_prop: float = training_prop
        self.current_oracle_index = 0
        self.split_index = int(training_prop*len(self.raw_test_cases))
        self.current_test_index = self.split_index

    def benchmark(self) -> str:
        """Return the name of the benchmark."""
        return self.file_path

    def get_test_details_lst(self) -> list[TestDetails]:
        """Return test cases in a list."""
        return self.test_details_lst

    def get_test_details_dict(self) -> dict:
        """Return test cases in a dictionary."""
        return self.test_details_dict

    def load(self, test_id: str) -> TestDetails:
        """Return test case with a specific id."""
        return self.test_details_dict[test_id]

    def get_test_ids(self):
        """Return al test case ids."""
        return self.test_details_dict.keys()


def _init_iterator(train_set: list[TestDetails]):
    """Python generator for the initialization interface for gRPC."""
    for test_detail in train_set:
        road_points = [competition_pb2.RoadPoint(sequenceNumber=i, x=pts[0], y=pts[1]) for i, pts in enumerate(test_detail.road_points)]
        test_case = competition_pb2.SDCTestCase(testId=test_detail.test_id, roadPoints=road_points)
        oracle: competition_pb2.Oracle = competition_pb2.Oracle(testCase=test_case, hasFailed=test_detail.hasFailed)
        yield oracle


def _test_suite_iterator(test_set: list[TestDetails]):
    """Python generator for the selection interface for gRPC."""
    for test_detail in test_set:
        road_points = [competition_pb2.RoadPoint(sequenceNumber=i, x=pts[0], y=pts[1]) for i, pts in enumerate(test_detail.road_points)]
        test_case = competition_pb2.SDCTestCase(testId=test_detail.test_id, roadPoints=road_points)
        yield test_case


class SensoDatTestLoader(EvaluationTestLoader):
    """Test Loader for tests stored in MongoDB."""

    def __init__(self, collection: Collection):
        """Initialize with a MongoDB collection."""
        self.collection = collection

    def benchmark(self) -> str:
        """Return the name of the benchmark."""
        return self.collection.name

    def get_test_details_lst(self) -> list[TestDetails]:
        """Return list of all test cases and their oracle."""
        querry = [
            {
                '$project': {
                    '_id': 0,
                    'sim_time': {'$toDouble': '$OpenDRIVE.header.sdc_test_info.@test_duration'},
                    'test_id': {'$toString': '$_id'},
                    'hasFailed': {
                        '$eq': [
                            '$OpenDRIVE.header.sdc_test_info.@test_outcome', 'FAIL'
                        ]
                    },
                    'road_points': {
                        '$map': {
                            'input': '$OpenDRIVE.road.planView.geometry',
                            'as': 'road_points',
                            'in': [
                                {
                                    '$toDouble': '$$road_points.@x'
                                }, {
                                    '$toDouble': '$$road_points.@y'
                                }
                            ]
                        }
                    }
                }
            }
        ]

        cursor = self.collection.aggregate(querry)
        test_details: list = []
        for item in cursor:
            td = TestDetails(
                test_id=item['test_id'],
                hasFailed=item['hasFailed'],
                sim_time=item['sim_time'],
                road_points=[(pt[0], pt[1]) for pt in item['road_points']]  # convert to list of tuples
            )
            test_details.append(td)

        return test_details

    def get_test_details_dict(self) -> dict:
        """Get test cases by their hash id."""
        return {'0': TestDetails(test_id='0', hasFailed=True, sim_time=1.0, road_points=[(0.0, 0.0)])}


class ToolEvaluator:
    """Evaluator to compute evaluation metrics for test selectors."""

    def __init__(self, metric_evaluator: MetricEvaluator, test_loader: EvaluationTestLoader, train_proportion=0.8):
        """Initialize evaluator with test loader and metric evaluator."""
        self.metric_evaluator = metric_evaluator
        self.test_loader: EvaluationTestLoader = test_loader
        self.train_proportion = train_proportion
        self.train_max_idx = int(len(test_loader.get_test_details_lst()) * train_proportion)
        self.train_set: list[TestDetails] = test_loader.get_test_details_lst()[:self.train_max_idx]
        self.test_set: list[TestDetails] = test_loader.get_test_details_lst()[self.train_max_idx:]

    def evaluate(self, stub: competition_pb2_grpc.CompetitionToolStub) -> EvaluationReport:
        """Generate evaluation report for the given tool (stub)."""
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
            #print(test_case.testId)
        selection_end_time = time.time()

        return EvaluationReport(
            test_suite_cnt=len(self.test_set),
            benchmark=self.test_loader.benchmark(),
            selection_cnt=len(selection),
            time_to_initialize=(init_end_time-init_start_time),
            time_to_select_tests=(selection_end_time-selection_start_time),
            tool_name=name_reply.name,
            time_to_fault_ratio=self.metric_evaluator.time_to_fault_ratio(test_suite=self.test_set, selection=selection),
            fault_to_selection_ratio=self.metric_evaluator.fault_to_selection_ratio(test_suite=self.test_set, selection=selection),
            diversity=self.metric_evaluator.diversity(test_suite=self.test_set, selection=selection),
        )


if __name__ == "__main__":
    # load environment variables fro .env file
    load_dotenv()
    uri = os.getenv('SENSODAT_URI')

    # handle CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--url")
    parser.add_argument("-t", "--tests")
    parser.add_argument("-c", "--collection")
    args = parser.parse_args()

    tl: EvaluationTestLoader = None
    if args.url:
        GRPC_URL = args.url
    else:
        print('provide url to the tool with -u/--url')
        sys.exit(1)

    if args.tests:
        test_file = args.tests
        tl = SampleEvaluationTestLoader(test_file, 0.8)
    elif args.collection:
        sensodat_collection = args.collection
        client = MongoClient(uri)
        coll: Collection = client.get_database('sdc_sim_data').get_collection(sensodat_collection)
        tl = SensoDatTestLoader(coll)
    else:
        print('provide path to test cases -t/--tests or a collection -c/--collection of SensoDat')
        exit(1)

    me = MetricEvaluator()
    te = ToolEvaluator(me, tl)

    # set up gRPC conection stub
    channel = grpc.insecure_channel(GRPC_URL)
    stub = competition_pb2_grpc.CompetitionToolStub(channel)  # stub represents the tool

    # start evaluation
    report = te.evaluate(stub)
    save_csv(report, Path(os.getcwd()+'/output/results.csv'))
    print(report)
