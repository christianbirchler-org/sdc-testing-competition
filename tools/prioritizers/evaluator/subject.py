import abc
import mydata

import psycopg

class SubjectCreationStrategy(abc.ABC):
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @abc.abstractmethod
    def create(self, pg_conn: psycopg.Connection, size: int) -> mydata.Subject:
        pass



class OverallRandomSubjectCreationStrategy(SubjectCreationStrategy):
    def __init__(self) -> None:
        super().__init__()

    def name(self):
        return "overall"

    def create(self, pg_conn: psycopg.Connection, size: int):
        create_random_subject_sql = """
        SELECT
        test_case_id,
        sensodat_collection_id,
        object_id,
        has_passed,
        has_failed,
        risk_factor,
        oob,
        max_speed_kmh,
        is_valid,
        sensodat_file_path,
        duration_seconds
        FROM create_and_get_random_subject_overall_with_size(%s);
        """

        persist_random_subject_sql = """
        INSERT INTO subjects(test_suite_size) VALUES (%s) RETURNING subject_id;
        """

        subj = []
        with pg_conn.cursor() as cur:
            cur.execute(create_random_subject_sql, (size,))
            for record in cur:
                if record is not None:
                    pgdata = mydata.PGTestData(
                        test_case_id=record[0],
                        sensodat_collection_id=record[1],
                        object_id=record[2],
                        has_passed=record[3],
                        has_failed=record[4],
                        risk_factor=record[5],
                        oob=record[6],
                        max_speed_kmh=record[7],
                        is_valid=record[8],
                        sensodat_file_path=record[9],
                        duration_seconds=record[10],
                    )
                    subj.append(pgdata)
            cur.execute(persist_random_subject_sql, (size,))
            record = cur.fetchone()
            subject_id = record[0]

        return mydata.Subject(subject_id=subject_id, pg_test_data=subj)


    def get_sensodat_queryies_per_collection(self):
        sample_with_object_ids = self.get_random_object_ids()

        mongo_object_id_lst = []

        find = {'_id': mongo_object_id_lst}
        projection = {
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

        return []

