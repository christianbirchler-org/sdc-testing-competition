import os
import time
import abc
import sys
import json
import argparse

from dataclasses import dataclass
from pathlib import Path

import competition_2026_pb2
import competition_2026_pb2_grpc

import grpc
import shapely
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import pairwise_distances
from pymongo import MongoClient
from pymongo.collection import Collection


class InitializationError(Exception):
    pass


class TestDoesNotExistError(Exception):
    pass


class PrioritizationError(Exception):
    """Raised when the prioritization list is invalid."""
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
    """TODO: Computation of all evaluation metrics."""

    def __init__(self):
        pass

    def compute_apdf(self, test_suite: list[TestDetails], prioritized_list: list[str]):
        """
        Computes Average Percentage of Fault Detected (APFD) metric
        """
        test_dict = {test.test_id: test for test in test_suite}
        num_tests = len(prioritized_list)
        failed_positions = []

        for i, test_id in enumerate(prioritized_list):
            test = test_dict.get(test_id)
            if test and test.hasFailed:
                failed_positions.append(i + 1)  # indexing starts at 1

        num_failed = len(failed_positions)

        if num_tests == 0 or num_failed == 0:
            return 1.0  # perfect score if no faults or no tests

        sum_failed_ranks = sum(failed_positions)
        apfd = 1 - (sum_failed_ranks / (num_tests * num_failed)) + 1 / (2 * num_tests)

        return apfd


    def compute_apdfc(self, test_suite: list[TestDetails], prioritized_list: list[str]):
        """
        Compute Cost-aware variant of Average Percentage of Fault Detected metric (APFD_C)
        """

        test_dict = {test.test_id: test for test in test_suite}
        sim_times = []
        cumulative_costs_to_faults = []

        cumulative_time = 0.0
        total_cost = 0.0
        num_failed = 0

        for test_id in prioritized_list:
            test = test_dict.get(test_id)
            if test is None:
                continue
            sim_times.append(test.sim_time)
            cumulative_time += test.sim_time
            total_cost += test.sim_time
            if test.hasFailed:
                cumulative_costs_to_faults.append(cumulative_time)
                num_failed += 1

        if num_failed == 0 or total_cost == 0:
            return 1.0  # perfect score if no faults or no cost

        sum_cfi = sum(cumulative_costs_to_faults)
        apfdc = 1 - (sum_cfi / (total_cost * num_failed)) + 1 / (2 * num_failed)

        return apfdc


    def compute_time_to_first_fault(self, test_suite: list[TestDetails], prioritized_list: list[str]):
        """
        Returns Time to first fault as a float, or None if no fault is found.
        """
        cumulative_time = 0.0
        test_dict = {test.test_id: test for test in test_suite}

        for test_id in prioritized_list:
            test = test_dict.get(test_id)
            cumulative_time += test.sim_time
            if test.hasFailed:
                return cumulative_time
        return None  # no failure found in selection


    def compute_time_to_last_fault(self, test_suite: list[TestDetails], prioritized_list: list[str]):
        """
        Returns Time to last fault as a float, or None if no fault is found.
        """
        cumulative_time = 0.0
        cumulative_time_to_last_fault = None

        test_dict = {test.test_id: test for test in test_suite}

        for i, test_id in enumerate(prioritized_list):
            test = test_dict.get(test_id)
            cumulative_time += test.sim_time
            if test.hasFailed:
                cumulative_time_to_last_fault = cumulative_time

        return cumulative_time_to_last_fault


def check_prioritization_validity(self, test_suite: list[TestDetails], prioritized_list: list[str]):
        """
        Check if the prioritized list is well-formed:
        - All IDs in the prioritized_list exist in the test_suite
        - No duplicates
        - No missing test cases

        Raises:
        - PrioritizationError if the prioritized_list is invalid.
        """
        test_ids = {test.test_id for test in test_suite}
        seen = set()

        for test_id in prioritized_list:
            if test_id not in test_ids:
                raise PrioritizationError(f"Invalid test ID in prioritization: {test_id}")
            if test_id in seen:
                raise PrioritizationError(f"Duplicate test ID in prioritization: {test_id}")
            seen.add(test_id)

        if len(seen) != len(test_ids):
            missing = test_ids - seen
            raise PrioritizationError(f"Prioritization list does not cover all test cases. Missing: {missing}")



@dataclass
class EvaluationReport:
    """ This class holds evaluation metrics of a tool."""

    test_suite_cnt: int
    benchmark: str
    time_to_initialize: float
    time_to_prioritize_tests: float
    tool_name: str
    time_to_first_fault: float | None
    time_to_last_fault: float | None
    apfd: float
    apfdc: float



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
        road_points = [competition_2026_pb2.RoadPoint(sequenceNumber=i, x=pts[0], y=pts[1]) for i, pts in enumerate(test_detail.road_points)]
        test_case = competition_2026_pb2.SDCTestCase(testId=test_detail.test_id, roadPoints=road_points)
        oracle: competition_2026_pb2.Oracle = competition_2026_pb2.Oracle(testCase=test_case, hasFailed=test_detail.hasFailed)
        yield oracle


def _test_suite_iterator(test_set: list[TestDetails]):
    """Python generator for the selection interface for gRPC."""
    for test_detail in test_set:
        road_points = [competition_2026_pb2.RoadPoint(sequenceNumber=i, x=pts[0], y=pts[1]) for i, pts in enumerate(test_detail.road_points)]
        test_case = competition_2026_pb2.SDCTestCase(testId=test_detail.test_id, roadPoints=road_points)
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
    """Evaluator to compute evaluation metrics for test prioritizers."""

    def __init__(self, metric_evaluator: MetricEvaluator, test_loader: EvaluationTestLoader, train_proportion=0.8):
        """Initialize evaluator with test loader and metric evaluator."""
        self.metric_evaluator = metric_evaluator
        self.test_loader: EvaluationTestLoader = test_loader
        self.train_proportion = train_proportion
        self.train_max_idx = int(len(test_loader.get_test_details_lst()) * train_proportion)
        self.train_set: list[TestDetails] = test_loader.get_test_details_lst()[:self.train_max_idx]
        self.test_set: list[TestDetails] = test_loader.get_test_details_lst()[self.train_max_idx:]

    def evaluate(self, stub: competition_2026_pb2_grpc.CompetitionToolStub) -> EvaluationReport:
        """Generate evaluation report for the given tool (stub)."""
        # get the tool name
        name_reply: competition_2026_pb2.NameReply = stub.Name(competition_2026_pb2.Empty())

        # initialize the tool with training data (i.e., Oracles)
        init_start_time = time.time()
        init_resp: competition_2026_pb2.InitializationReply = stub.Initialize(_init_iterator(train_set=self.train_set))
        if not init_resp.ok:
            raise InitializationError()
        init_end_time = time.time()

        # tool returns a prioritization of test cases
        prioritization_start_time = time.time()
        prioritization_iterator = stub.Prioritize(_test_suite_iterator(self.train_set))

        prioritization_end_time = time.time()

        #build the prioritized list
        prioritized_list = []
        for test_case in prioritization_iterator:
            test_case: competition_2026_pb2.SDCTestCase = test_case
            prioritized_list.append(test_case.testId)

        # check if the prioritized list is well-formed (throws PrioritizationError on fail)
        self.metric_evaluator.check_prioritization_validity(test_suite=self.test_set, prioritized_list=prioritized_list)

        return EvaluationReport(
            test_suite_cnt=len(self.test_set),
            benchmark=self.test_loader.benchmark(),
            time_to_initialize=(init_end_time-init_start_time),
            time_to_prioritize_tests=(prioritization_end_time - prioritization_start_time),
            tool_name=name_reply.name,
            time_to_first_fault=self.metric_evaluator.compute_time_to_first_fault(test_suite=self.test_set, prioritized_list=prioritized_list),
            time_to_last_fault=self.metric_evaluator.compute_time_to_last_fault(test_suite=self.test_set, prioritized_list=prioritized_list),
            apfd=self.metric_evaluator.compute_apfd(test_suite=self.test_set, prioritized_list=prioritized_list),
            apfdc=self.metric_evaluator.compute_apfdc(test_suite=self.test_set, prioritized_list=prioritized_list)
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
    stub = competition_2026_pb2_grpc.CompetitionToolStub(channel)  # stub represents the tool

    # start evaluation
    report = te.evaluate(stub)
    # save_csv(report, Path(os.getcwd()+'/output/results.csv'))
    print(report)
