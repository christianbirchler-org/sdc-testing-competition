import os
from bson import timestamp
from dotenv import load_dotenv
import grpc
import psycopg
import pymongo
from pymongo.server_api import ServerApi
import sampling
import subject
import competition_2026_pb2
import competition_2026_pb2_grpc
import metrics
import sampling
import time
import mydata
import errors
import datetime
import utils


def _register_experiment_in_db(pg_conn: psycopg.Connection, tool_name: str, experiment_start_time: datetime.datetime) -> int:
    insert_sql = """
    INSERT INTO experiments(tool_name, start_utc)
    VALUES (%s, %s)
    RETURNING experiment_id;
    """

    experiment_id = -1
    data = (tool_name, experiment_start_time)
    with pg_conn.cursor() as cur:
        cur.execute(insert_sql, data)
        record = cur.fetchone()
        experiment_id = record[0]

    return experiment_id

def _persist_treatment(pg_conn: psycopg.Connection, treatment_data: mydata.TreatmentData, subject_id: int, is_successful: bool):
    insert_sql = """
    INSERT INTO treatments(
    time_to_prioritize_tests,
    time_to_first_fault,
    time_to_last_fault,
    apfd,
    apfdc,
    subject_id,
    is_successful
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    RETURNING treatment_id;
    """

    treatment_id = -1
    data = (
        treatment_data.time_to_prioritize_tests,
        treatment_data.time_to_first_fault,
        treatment_data.time_to_last_fault,
        treatment_data.apfd,
        treatment_data.apfdc,
        subject_id,
        is_successful
    )
    with pg_conn.cursor() as cur:
        cur.execute(insert_sql, data)
        record = cur.fetchone()
        treatment_id = record[0]

    return treatment_id



def _persist_experiment_end_time(pg_conn: psycopg.Connection, experiment_id: int, end_time: datetime.datetime):
    insert_sql = """
    UPDATE experiments SET end_utc=%s
    WHERE experiment_id=%s;
    """

    data = (end_time, experiment_id)
    with pg_conn.cursor() as cur:
        cur.execute(insert_sql, data)


class ExperimentalSetup:
    """Evaluator to compute evaluation metrics for test prioritizers."""

    def __init__(self, pg_conn: psycopg.Connection, mongo_client: pymongo.MongoClient, metric_evaluator: metrics.MetricEvaluator, sampler: sampling.Sampler):
        """Initialize evaluator with test loader and metric evaluator."""
        self.pg_conn = pg_conn
        self.mongo_client = mongo_client
        self.metric_evaluator = metric_evaluator
        self.sampler: sampling.Sampler = sampler

    def start_experiment(self, stub: competition_2026_pb2_grpc.CompetitionToolStub, sample_size, subject_size):
        experiment_start_time = datetime.datetime.now(datetime.timezone.utc)

        name_reply: competition_2026_pb2.NameReply = stub.Name(competition_2026_pb2.Empty())
        tool_name = name_reply.name
        experiment_id = _register_experiment_in_db(self.pg_conn, tool_name, experiment_start_time)
        print('start experiment: {}'.format(experiment_id))

        sample = self.sampler.sample(self.pg_conn, sample_size, subject_size, experiment_id)
        # initialization data is sampled separatedly with the same size as for the experiments below
        init_data = sample.subjects[0]
        evaluation_data = sample.subjects[1:]


        init_start_time = time.time()
        init_resp: competition_2026_pb2.InitializationReply = stub.Initialize(utils.init_iterator(mongo_client=self.mongo_client, init_data=init_data))
        if not init_resp.ok:
            raise errors.InitializationError()
        init_end_time = time.time()

        #treatment_ids = []
        for subject in evaluation_data:
            try:
                treatment_data: mydata.TreatmentData = self.treat_subject(stub, subject)
                _persist_treatment(self.pg_conn, treatment_data, subject.subject_id, True)
                print('subject={} sucessfully treated'.format(subject.subject_id))
            except Exception as e:
                treatment_data: mydata.TreatmentData = mydata.TreatmentData(
                    time_to_prioritize_tests=-1,
                    time_to_first_fault=-1,
                    time_to_last_fault=-1,
                    apfd=-1,
                    apfdc=-1
                )
                _persist_treatment(self.pg_conn, treatment_data, subject.subject_id, False)
                print("treatment failed for subject: {}".format(subject.subject_id))
                #print(e)

        #print('experiment id: {}\ttreatment ids: {}'.format(experiment_id, treatment_ids))

        experiment_end_time = datetime.datetime.now(datetime.timezone.utc)

        _persist_experiment_end_time(self.pg_conn, experiment_id, experiment_end_time)

        print('end experiment: {}'.format(experiment_id))

        return mydata.ExperimentalResults(
            tool_name=tool_name,
            start_utc=experiment_start_time,
            end_utc=experiment_end_time
        )


    def treat_subject(self, stub: competition_2026_pb2_grpc.CompetitionToolStub, subject: mydata.Subject) -> mydata.TreatmentData:

        prioritization_start_time = time.time()
        prioritization_iterator = stub.Prioritize(utils.test_suite_iterator(self.mongo_client, subject))

        #build the prioritized list
        prioritized_list = []
        for test_case in prioritization_iterator:
            #print(test_case)
            test_case: competition_2026_pb2.PrioritizationReply = test_case
            prioritized_list.append(test_case.testId)

        prioritization_end_time = time.time()

        # check if the prioritized list is well-formed (throws PrioritizationError on fail)
        self.metric_evaluator.check_prioritization_validity(test_suite=subject, prioritized_list=prioritized_list)

        return mydata.TreatmentData(
            time_to_prioritize_tests=(prioritization_end_time - prioritization_start_time),
            time_to_first_fault=self.metric_evaluator.compute_time_to_first_fault(test_suite=subject, prioritized_list=prioritized_list),
            time_to_last_fault=self.metric_evaluator.compute_time_to_last_fault(test_suite=subject, prioritized_list=prioritized_list),
            apfd=self.metric_evaluator.compute_apfd(test_suite=subject, prioritized_list=prioritized_list),
            apfdc=self.metric_evaluator.compute_apdfc(test_suite=subject, prioritized_list=prioritized_list)
        )


    def evaluate(self, stub: competition_2026_pb2_grpc.CompetitionToolStub) -> mydata.EvaluationReport:
        """Generate evaluation report for the given tool (stub)."""
        # get the tool name
        name_reply: competition_2026_pb2.NameReply = stub.Name(competition_2026_pb2.Empty())
        sample_for_initialization = self.sampler.sample(self.pg_conn, nr_subjects=1, subject_size=10)
        initialization_test_suite = sample_for_initialization.subjects[0]

        # initialize the tool with training data (i.e., Oracles)
        init_start_time = time.time()
        init_resp: competition_2026_pb2.InitializationReply = stub.Initialize(utils.init_iterator(self.mongo_client, initialization_test_suite))
        if not init_resp.ok:
            raise errors.InitializationError()
        init_end_time = time.time()

        # create test sample for the evaluation
        sample_for_evaluation = self.sampler.sample(self.pg_conn, nr_subjects=1, subject_size=10)
        evaluation_test_suite = sample_for_evaluation.subjects[0]

        # tool returns a prioritization of test cases
        prioritization_start_time = time.time()
        prioritization_iterator = stub.Prioritize(utils.test_suite_iterator(self.mongo_client, evaluation_test_suite))
        prioritization_end_time = time.time()

        #build the prioritized list
        prioritized_list = []
        for test_case in prioritization_iterator:
            test_case: competition_2026_pb2.PrioritizationReply = test_case
            prioritized_list.append(test_case.testId)

        # check if the prioritized list is well-formed (throws PrioritizationError on fail)
        self.metric_evaluator.check_prioritization_validity(test_suite=evaluation_test_suite, prioritized_list=prioritized_list)

        return mydata.EvaluationReport(
            test_suite_cnt=len(evaluation_test_suite.pg_test_data),
            benchmark=self.sampler.benchmark(),
            time_to_initialize=(init_end_time-init_start_time),
            time_to_prioritize_tests=(prioritization_end_time - prioritization_start_time),
            tool_name=name_reply.name,
            time_to_first_fault=self.metric_evaluator.compute_time_to_first_fault(test_suite=evaluation_test_suite, prioritized_list=prioritized_list),
            time_to_last_fault=self.metric_evaluator.compute_time_to_last_fault(test_suite=evaluation_test_suite, prioritized_list=prioritized_list),
            apfd=self.metric_evaluator.compute_apfd(test_suite=evaluation_test_suite, prioritized_list=prioritized_list),
            apfdc=self.metric_evaluator.compute_apdfc(test_suite=evaluation_test_suite, prioritized_list=prioritized_list)
        )
