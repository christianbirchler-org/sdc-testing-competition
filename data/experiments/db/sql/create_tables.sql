CREATE TABLE IF NOT EXISTS campaigns(
       campaigns_id SERIAL PRIMARY KEY
);


CREATE TABLE IF NOT EXISTS experiments(
       experiment_id SERIAL PRIMARY KEY,
       tool_name TEXT,
       start_utc TIMESTAMP,
       end_utc TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sampling_strategies(
       sampling_strategy_id SERIAL PRIMARY KEY,
       strategy TEXT
);

INSERT INTO sampling_strategies(sampling_strategy_id, strategy)
VALUES
(0, 'overall'),
(1, 'within_collection');

CREATE TABLE IF NOT EXISTS samples(
       sample_id SERIAL PRIMARY KEY,
       sampling_strategy_id INT,
       experiment_id INT,

       CONSTRAINT fk_sampling_strategy_id FOREIGN KEY (sampling_strategy_id) REFERENCES sampling_strategies(sampling_strategy_id),
       CONSTRAINT fk_experiment_id FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

CREATE TABLE IF NOT EXISTS subjects(
       subject_id SERIAL PRIMARY KEY,
       test_suite_size INT
);

CREATE TABLE IF NOT EXISTS samples_subjects(
       samples_subjects_id SERIAL PRIMARY KEY,
       sample_id INT,
       subject_id INT,

       CONSTRAINT fk_sample_id FOREIGN KEY (sample_id) REFERENCES samples(sample_id),
       CONSTRAINT fk_subject_id FOREIGN KEY (subject_id) REFERENCES subjects(subject_id)
);

CREATE TABLE IF NOT EXISTS sensodat_collections(
       sensodat_collection_id INT PRIMARY KEY,
       collection_name TEXT
);

INSERT INTO sensodat_collections(sensodat_collection_id, collection_name)
VALUES
(0, 'campaign_12_freneticV'),
(1, 'campaign_2_frenetic'),
(2, 'campaign_13_frenetic_v'),
(3, 'campaign_14_frenetic'),
(4, 'campaign_6_frenetic_v'),
(5, 'campaign_9_frenetic'),
(6, 'campaign_4_frenetic_v'),
(7, 'campaign_6_ambiegen'),
(8, 'campaign_11_frenetic'),
(9, 'campaign_3_ambiegen'),
(10, 'campaign_2_ambiegen'),
(11, 'campaign_7_frenetic'),
(12, 'campaign_8_ambiegen'),
(13, 'campaign_7_frenetic_v'),
(14, 'campaign_7_ambiegen'),
(15, 'campaign_5_frenetic_v'),
(16, 'campaign_9_ambiegen'),
(17, 'campaign_11_ambiegen'),
(18, 'campaign_4_ambiegen'),
(19, 'campaign_4_frenetic'),
(20, 'campaign_13_ambiegen'),
(21, 'campaign_3_frenetic'),
(22, 'campaign_15_ambiegen'),
(23, 'campaign_15_freneticV'),
(24, 'campaign_11_frenetic_v'),
(25, 'campaign_5_frenetic'),
(26, 'campaign_14_frenetic_v'),
(27, 'campaign_12_frenetic'),
(28, 'campaign_15_frenetic'),
(29, 'campaign_14_ambiegen'),
(30, 'campaign_13_frenetic'),
(31, 'campaign_8_frenetic'),
(32, 'campaign_5_ambiegen'),
(33, 'campaign_6_frenetic'),
(34, 'campaign_2_frenetic_v'),
(35, 'campaign_10_ambiegen');



CREATE TABLE IF NOT EXISTS sensodat_test_cases(
       test_case_id SERIAL PRIMARY KEY,
       sensodat_collection_id INT,
       object_id TEXT,
       has_passed BOOL,
       has_failed BOOL,
       risk_factor FLOAT,
       oob FLOAT,
       max_speed_kmh INT,
       is_valid BOOL,
       sensodat_file_path TEXT,
       duration_seconds FLOAT,

       CONSTRAINT fk_sensodat_collection_id FOREIGN KEY (sensodat_collection_id) REFERENCES sensodat_collections(sensodat_collection_id)
);

CREATE TABLE IF NOT EXISTS subjects_sensodat_test_cases(
       subjects_sensodat_test_cases SERIAL PRIMARY KEY,
       subject_id INT,
       sensodat_test_case_id INT,

       CONSTRAINT fk_subject_id FOREIGN KEY (subject_id) REFERENCES subjects(subject_id),
       CONSTRAINT fk_sensodat_test_case_id FOREIGN KEY (sensodat_test_case_id) REFERENCES sensodat_test_cases(test_case_id)
);

CREATE TABLE IF NOT EXISTS treatments(
       treatment_id SERIAL PRIMARY KEY,
       time_to_prioritize_tests FLOAT,
       time_to_first_fault FLOAT,
       time_to_last_fault FLOAT,
       apfd FLOAT,
       apfdc FLOAT,
       subject_id INT,
       is_successful BOOL,

       CONSTRAINT fk_subject_id FOREIGN KEY (subject_id) REFERENCES subjects(subject_id)
);


CREATE OR REPLACE FUNCTION create_and_get_random_subject_within_collection_with_size(coll_id INT, subject_size INT)
RETURNS TABLE(
       test_case_id INT,
       sensodat_collection_id INT,
       object_id TEXT,
       has_passed BOOL,
       has_failed BOOL,
       risk_factor FLOAT,
       oob FLOAT,
       max_speed_kmh INT,
       is_valid BOOL,
       sensodat_file_path TEXT,
       duration_seconds FLOAT)
AS $$
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
FROM sensodat_test_cases
WHERE sensodat_collection_id=coll_id
ORDER BY random()
LIMIT subject_size;
$$
LANGUAGE SQL;


CREATE OR REPLACE FUNCTION create_and_get_random_subject_overall_with_size(subject_size INT)
RETURNS TABLE(
       test_case_id INT,
       sensodat_collection_id INT,
       object_id TEXT,
       has_passed BOOL,
       has_failed BOOL,
       risk_factor FLOAT,
       oob FLOAT,
       max_speed_kmh INT,
       is_valid BOOL,
       sensodat_file_path TEXT,
       duration_seconds FLOAT)
AS $$
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
FROM sensodat_test_cases
ORDER BY random()
LIMIT subject_size;
$$
LANGUAGE SQL;


CREATE OR REPLACE FUNCTION get_data_of_experiment(ex_id INT)
RETURNS TABLE (
treatment_id INT,
time_to_prioritize_tests FLOAT,
time_to_first_fault FLOAT,
time_to_last_fault FLOAT,
apfd FLOAT,
apfdc FLOAT,
subject_id INT,
is_successful BOOL,
samples_subjects_id INT,
sample_id INT,
sampling_strategy_id INT,
experiment_id INT
)
AS $$
SELECT
t1.treatment_id INT,
t1.time_to_prioritize_tests FLOAT,
t1.time_to_first_fault FLOAT,
t1.time_to_last_fault FLOAT,
t1.apfd FLOAT,
t1.apfdc FLOAT,
t1.subject_id INT,
t1.is_successful BOOL,
t2.samples_subjects_id INT,
t2.sample_id INT,
t3.sampling_strategy_id INT,
t3.experiment_id INT
FROM treatments t1
JOIN samples_subjects t2 ON t1.subject_id=t2.subject_id
JOIN samples t3 ON t2.sample_id=t3.sample_id
WHERE t3.experiment_id=ex_id;
$$
LANGUAGE SQL;

--CREATE OR REPLACE PROCEDURE create_subject_within_collection_with_size(coll_id INT, subject_size INT)
--AS $$
--DECLARE
--subj_id INT;
--BEGIN
--
--subj_id = INSERT INTO subjects(test_suite_size) VALUES (subject_size) RETURNING subject_id;
--
--
--INSERT INTO subjects_sensodat_test_cases (subject_id, sensodat_test_case_id)
--VALUES SELECT subject_id, sensodat_test_case_id FROM create_and_get_random_subject_within_collection_with_size(coll_id, subject_size);
--END
--$$
--LANGUAGE plpgsql;
