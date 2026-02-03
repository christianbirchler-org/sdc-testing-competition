import psycopg
import json
import abc

import pymongo
from pymongo.collection import Collection
from pymongo.server_api import ServerApi
import subject
import mydata

import competition_2026_pb2
import utils

class Sampler(abc.ABC):
    """Abstract test loader for loading the evaluation data."""

    @abc.abstractmethod
    def sample(self, pg_conn: psycopg.Connection, nr_subjects: int, subject_size: int, experiment_id: int) -> mydata.Sample:
        pass

    @abc.abstractmethod
    def benchmark(self) -> str:
        """Return the name of the benchmark."""
        pass

    #@abc.abstractmethod
    #def get_test_details_lst(self) -> list[mydata.TestDetails]:
    #    """Return list of all test cases and their oracle."""
    #    pass

    #@abc.abstractmethod
    #def get_test_details_dict(self) -> dict:
    #    """Get test cases by their hash id."""
    #    pass


class SampleEvaluationTestLoader(Sampler):
    """Sample test loader for the provided data."""

    def __init__(self, file_path: str, training_prop: float):
        """Initialize test loader with path to dataset."""
        super().__init__()
        self.file_path = file_path
        self.raw_test_cases: list = None
        with open(file_path, 'r') as fp:
            self.raw_test_cases = json.load(fp)

        self.test_details_lst = utils.make_test_details_list(self.raw_test_cases)
        self.test_details_dict = {test_details.test_id: test_details for test_details in self.test_details_lst}
        self.training_prop: float = training_prop
        self.current_oracle_index = 0
        self.split_index = int(training_prop*len(self.raw_test_cases))
        self.current_test_index = self.split_index

    def benchmark(self) -> str:
        """Return the name of the benchmark."""
        return self.file_path

    def get_test_details_lst(self) -> list[mydata.TestDetails]:
        """Return test cases in a list."""
        return self.test_details_lst

    def get_test_details_dict(self) -> dict:
        """Return test cases in a dictionary."""
        return self.test_details_dict

    def load(self, test_id: str) -> mydata.TestDetails:
        """Return test case with a specific id."""
        return self.test_details_dict[test_id]

    def get_test_ids(self):
        """Return al test case ids."""
        return self.test_details_dict.keys()




#class SensoDatTestLoader(Sampler):
#    """Test Loader for tests stored in MongoDB."""
#
#    def __init__(self, collection: Collection):
#        """Initialize with a MongoDB collection."""
#        self.collection = collection
#
#    def sample(nr_subjects: int, subject_size: int) -> mydata.Sample:
#        return mydata.Sample()
#        pass
#
#    def benchmark(self) -> str:
#        """Return the name of the benchmark."""
#        return self.collection.name
#
#    def get_test_details_lst(self) -> list[mydata.TestDetails]:
#        """Return list of all test cases and their oracle."""
#        querry = [
#            {
#                '$project': {
#                    '_id': 0,
#                    'sim_time': {'$toDouble': '$OpenDRIVE.header.sdc_test_info.@test_duration'},
#                    'test_id': {'$toString': '$_id'},
#                    'hasFailed': {
#                        '$eq': [
#                            '$OpenDRIVE.header.sdc_test_info.@test_outcome', 'FAIL'
#                        ]
#                    },
#                    'road_points': {
#                        '$map': {
#                            'input': '$OpenDRIVE.road.planView.geometry',
#                            'as': 'road_points',
#                            'in': [
#                                {
#                                    '$toDouble': '$$road_points.@x'
#                                }, {
#                                    '$toDouble': '$$road_points.@y'
#                                }
#                            ]
#                        }
#                    }
#                }
#            }
#        ]
#
#        cursor = self.collection.aggregate(querry)
#        test_details: list = []
#        for item in cursor:
#            td = mydata.TestDetails(
#                test_id=item['test_id'],
#                hasFailed=item['hasFailed'],
#                sim_time=item['sim_time'],
#                road_points=[(pt[0], pt[1]) for pt in item['road_points']]  # convert to list of tuples
#            )
#            test_details.append(td)
#
#        return test_details
#
#    def get_test_details_dict(self) -> dict:
#        """Get test cases by their hash id."""
#        return {'0': mydata.TestDetails(test_id='0', hasFailed=True, sim_time=1.0, road_points=[(0.0, 0.0)])}


class CompetitionEvaluationSampler(Sampler):

    def __init__(self, mongo_client, sampling_strategy: subject.SubjectCreationStrategy) -> None:
        super().__init__()
        self.mongo_client = mongo_client
        self.subject_creation_strategy = sampling_strategy

    def sample(self, pg_conn: psycopg.Connection, nr_subjects: int, subject_size: int, experiment_id: int) -> mydata.Sample:

        subjects = []
        for _ in range(nr_subjects):
            subject: mydata.Subject = self.subject_creation_strategy.create(pg_conn, subject_size)
            subjects.append(subject)



        persist_sample_sql = """
        INSERT INTO samples(sampling_strategy_id, experiment_id)
        SELECT sampling_strategy_id, %s
        FROM sampling_strategies
        WHERE strategy=%s
        RETURNING sample_id;
        """

        persist_sample_subjects_sql = """
        INSERT INTO samples_subjects(sample_id, subject_id)
        VALUES (%s, %s);
        """

        sample_id = -1
        with pg_conn.cursor() as cur:
            cur.execute(persist_sample_sql, (experiment_id, self.subject_creation_strategy.name()))
            record = cur.fetchone()
            sample_id = record[0]

        if sample_id == -1:
            raise Exception()


        for subject in subjects:
            with pg_conn.cursor() as cur:
                cur.execute(persist_sample_subjects_sql, (sample_id, subject.subject_id))

        return mydata.Sample(
            sample_id=sample_id,
            subjects=subjects
        )


    def benchmark(self) -> str:
        """Return the name of the benchmark."""
        return self.subject_creation_strategy.name()


    #def get_test_details_lst(self) -> list[mydata.TestDetails]:
    #    """Return list of all test cases and their oracle."""

    #    test_details: list = []
    #    queries_per_collection = self.subject_creation_strategy.create()
    #    for collection, query in queries_per_collection:
    #        cursor = self.collection.aggregate(query)
    #        for item in cursor:
    #            td = mydata.TestDetails(
    #                test_id=item['test_id'],
    #                hasFailed=item['hasFailed'],
    #                sim_time=item['sim_time'],
    #                road_points=[(pt[0], pt[1]) for pt in item['road_points']]  # convert to list of tuples
    #            )
    #            test_details.append(td)

    #    return test_details

    def get_test_details_dict(self) -> dict:
        """Get test cases by their hash id."""
        return {'0': mydata.TestDetails(test_id='0', hasFailed=True, sim_time=1.0, road_points=[(0.0, 0.0)])}

