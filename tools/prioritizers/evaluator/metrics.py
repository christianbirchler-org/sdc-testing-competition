import mydata
import errors


class MetricEvaluator:
    def __init__(self):
        pass

    def compute_apfd(self, test_suite: mydata.Subject, prioritized_list: list[str]):
        """
        Computes Average Percentage of Fault Detected (APFD) metric
        """
        test_dict = {test.object_id: test for test in test_suite.pg_test_data}
        num_tests = len(prioritized_list)
        failed_positions = []

        for i, test_id in enumerate(prioritized_list):
            test = test_dict.get(test_id)
            if test and test.has_failed:
                failed_positions.append(i + 1)  # indexing starts at 1

        num_failed = len(failed_positions)

        if num_tests == 0 or num_failed == 0:
            return 1.0  # perfect score if no faults or no tests

        sum_failed_ranks = sum(failed_positions)
        apfd = 1 - (sum_failed_ranks / (num_tests * num_failed)) + 1 / (2 * num_tests)

        return apfd


    def compute_apdfc(self, test_suite: mydata.Subject, prioritized_list: list[str]):
        """
        Compute Cost-aware variant of Average Percentage of Fault Detected metric (APFD_C)
        """

        test_dict = {test.object_id: test for test in test_suite.pg_test_data}
        sim_times = []
        cumulative_costs_to_faults = []

        cumulative_time = 0.0
        total_cost = 0.0
        num_failed = 0

        for test_id in prioritized_list:
            test = test_dict.get(test_id)
            if test is None:
                continue
            sim_times.append(test.duration_seconds)
            cumulative_time += test.duration_seconds
            total_cost += test.duration_seconds
            if test.has_failed:
                cumulative_costs_to_faults.append(cumulative_time)
                num_failed += 1

        if num_failed == 0 or total_cost == 0:
            return 1.0  # perfect score if no faults or no cost

        sum_cfi = sum(cumulative_costs_to_faults)
        apfdc = 1 - (sum_cfi / (total_cost * num_failed)) + 1 / (2 * num_failed)

        return apfdc


    def compute_time_to_first_fault(self, test_suite: mydata.Subject, prioritized_list: list[str]):
        """
        Returns Time to first fault as a float, or None if no fault is found.
        """
        cumulative_time = 0.0
        test_dict = {test.object_id: test for test in test_suite.pg_test_data}

        for test_id in prioritized_list:
            test = test_dict.get(test_id)
            cumulative_time += test.duration_seconds
            if test.has_failed:
                return cumulative_time
        return None  # no failure found in selection


    def compute_time_to_last_fault(self, test_suite: mydata.Subject, prioritized_list: list[str]):
        """
        Returns Time to last fault as a float, or None if no fault is found.
        """
        cumulative_time = 0.0
        cumulative_time_to_last_fault = None

        test_dict = {test.object_id: test for test in test_suite.pg_test_data}

        for i, test_id in enumerate(prioritized_list):
            test = test_dict.get(test_id)
            cumulative_time += test.duration_seconds
            if test.has_failed:
                cumulative_time_to_last_fault = cumulative_time

        return cumulative_time_to_last_fault

    def check_prioritization_validity(self, test_suite: mydata.Subject, prioritized_list: list[str]):
        """
        Check if the prioritized list is well-formed:
        - All IDs in the prioritized_list exist in the test_suite
        - No duplicates
        - No missing test cases

        Raises:
        - PrioritizationError if the prioritized_list is invalid.
        """
        test_ids = {test.object_id for test in test_suite.pg_test_data}
        seen = set()

        for test_id in prioritized_list:
            if test_id not in test_ids:
                raise errors.PrioritizationError(f"Invalid test ID in prioritization: {test_id}")
            if test_id in seen:
                raise errors.PrioritizationError(f"Duplicate test ID in prioritization: {test_id}")
            seen.add(test_id)

        if len(seen) != len(test_ids):
            missing = test_ids - seen
            raise errors.PrioritizationError(f"Prioritization list does not cover all test cases. Missing: {missing}")
