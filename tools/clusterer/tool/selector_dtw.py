import math
import random

import pb.competition_pb2 as pb
import pb.competition_pb2_grpc as pb_grpc
from tool import util


class DTWDistanceSelector(pb_grpc.CompetitionToolServicer):
    def __init__(self):
        self.batch_size = 192 // 4
        self.outlier_cluster = -1

    def Name(self, request, context):
        return pb.NameReply(name="dtw_distance_selector")

    def Initialize(self, request_iterator, context):
        for oracle in request_iterator:
            oracle: pb.Oracle = oracle
            # print("hasFailed={}\ttestId={}".format(oracle.hasFailed, oracle.testCase.testId))

        return pb.InitializationReply(ok=True)

    def cluster_batch(self, batch):
        print(f"Clustering {len(batch)} test cases")
        test_cases = []
        for sdc_test_case in batch:
            sdc_test_case: pb.SDCTestCase = sdc_test_case
            test_cases.append(sdc_test_case)

        clusters = util.cluster_test_cases(test_cases)
        # Choose at most 3 test cases from each cluster
        print(f"Found {len(clusters)} clusters")

        selected_test_cases = []
        ratio = 0.3
        for cluster_id, test_cases in clusters.items():
            print(f"Cluster {cluster_id}: {len(test_cases)} test cases")
            # If the cluster is an outlier cluster, select all test cases
            if cluster_id == self.outlier_cluster:
                ratio = 0.5

            count = math.ceil(ratio * len(test_cases))
            selected_test_cases.extend(random.sample(test_cases, count))

        return selected_test_cases

    def Select(self, request_iterator, context):
        test_cases = []
        batch = []
        for sdc_test_case in request_iterator:
            sdc_test_case: pb.SDCTestCase = sdc_test_case
            test_cases.append(sdc_test_case)
            batch.append(sdc_test_case)

            if len(batch) == self.batch_size:
                selected_test_cases = self.cluster_batch(batch)
                for test_case in selected_test_cases:
                    yield pb.SelectionReply(testId=test_case.testId)

                batch = []

        if len(batch) > 0:
            selected_test_cases = self.cluster_batch(batch)
            for test_case in selected_test_cases:
                yield pb.SelectionReply(testId=test_case.testId)
