import random
import argparse
from concurrent.futures import ThreadPoolExecutor
import grpc
import numpy as np
from collections import defaultdict
import logging
from typing import List, Dict, Set, Tuple
import os
from datetime import datetime

import competition_pb2_grpc
import competition_pb2
from utils import RoadAnalyzer, RoadAnalysis

class BalancedTestSelector(competition_pb2_grpc.CompetitionToolServicer):
    def __init__(self):
        self.road_analyzer = RoadAnalyzer()
        self.historical_failures: Dict[str, bool] = {}
        self.road_analyses: Dict[str, RoadAnalysis] = {}
        self.similar_groups: Dict[int, Set[str]] = defaultdict(set)
        self.selection_history: List[str] = []
        
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        
        self.selection_ratio = 0.55 
        self.min_group_selections = 2  
        self.similarity_threshold = 0.75
        self.failure_weight = 1.5
        
    def Name(self, request, context):
        return competition_pb2.NameReply(name="balanced_selector")

    def _analyze_test_case(self, test_case: competition_pb2.SDCTestCase) -> bool:
        """Basic test case analysis"""
        try:
            road_points = [(pt.x, pt.y) for pt in test_case.roadPoints]
            analysis = self.road_analyzer.analyze_road(road_points)
            self.road_analyses[test_case.testId] = analysis
            
            # Find or create group
            best_group = None
            best_similarity = 0
            
            # Compare with existing groups
            for group_id, group_tests in self.similar_groups.items():
                if not group_tests:
                    continue
                    
                # Compare with first test in group
                ref_test = next(iter(group_tests))
                if ref_test not in self.road_analyses:
                    continue
                    
                similarity = self.road_analyzer.calculate_road_similarity(
                    analysis,
                    self.road_analyses[ref_test]
                )
                
                if similarity > self.similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_group = group_id
            
            # Assign to group or create new
            if best_group is not None:
                self.similar_groups[best_group].add(test_case.testId)
            else:
                new_group = len(self.similar_groups)
                self.similar_groups[new_group].add(test_case.testId)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error analyzing test case {test_case.testId}: {str(e)}")
            return False

    def Initialize(self, request_iterator, context):
        """Process initialization data"""
        self.logger.info("Starting initialization...")
        
        try:
            test_cases = []
            failure_count = 0
            
            for oracle in request_iterator:
                self.historical_failures[oracle.testCase.testId] = oracle.hasFailed
                test_cases.append(oracle.testCase)
                if oracle.hasFailed:
                    failure_count += 1
                    self.logger.info(f"Found failed test: {oracle.testCase.testId}")
            
            # Process test cases
            for test_case in test_cases:
                success = self._analyze_test_case(test_case)
                if not success:
                    return competition_pb2.InitializationReply(ok=False)
            
            self.logger.info(f"Initialization complete. Found {failure_count} failed tests out of {len(test_cases)}")
            return competition_pb2.InitializationReply(ok=True)
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            return competition_pb2.InitializationReply(ok=False)

    def _compute_selection_score(self, test_id: str) -> float:
        """Compute balanced selection score"""
        analysis = self.road_analyses[test_id]
        
        # Basic complexity score
        complexity_score = (
            0.4 * analysis.complexity_score +
            0.3 * (analysis.max_curvature / 0.3) +
            0.2 * (analysis.turn_count / 10) +
            0.1 * (analysis.total_length / 100)
        )
        
        # Historical failure bonus
        failure_bonus = self.failure_weight if self.historical_failures.get(test_id, False) else 1.0
        
        # Group diversity
        group_penalty = 1.0
        for group_tests in self.similar_groups.values():
            if test_id in group_tests:
                selected_in_group = sum(1 for t in group_tests if t in self.selection_history)
                if selected_in_group >= self.min_group_selections:
                    group_penalty = 0.5
                break
        
        # Recently selected penalty
        recent_penalty = 0.7 if test_id in self.selection_history[-10:] else 1.0
        
        return complexity_score * failure_bonus * group_penalty * recent_penalty

    def Select(self, request_iterator, context):
        """Select test cases with balanced criteria"""
        self.logger.info("Starting test selection...")
        
        try:
            test_cases = []
            for test_case in request_iterator:
                test_cases.append(test_case)
                if test_case.testId not in self.road_analyses:
                    self._analyze_test_case(test_case)
            
            total_tests = len(test_cases)
            target_selections = int(total_tests * self.selection_ratio)
            
            # Calculate initial scores
            scores = {
                test_case.testId: self._compute_selection_score(test_case.testId)
                for test_case in test_cases
            }
            
            # Sort by score
            sorted_tests = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            # Track selections per group
            group_selections = defaultdict(int)
            selected_count = 0
            
            # First pass: select minimum per group
            for test_id, score in sorted_tests:
                if selected_count >= target_selections:
                    break
                    
                # Find test's group
                test_group = None
                for group_id, group_tests in self.similar_groups.items():
                    if test_id in group_tests:
                        test_group = group_id
                        break
                
                if test_group is not None:
                    # minimum selections per group
                    if group_selections[test_group] < self.min_group_selections:
                        self.selection_history.append(test_id)
                        group_selections[test_group] += 1
                        selected_count += 1
                        self.logger.info(f"Selected test {test_id} with score {score:.3f}")
                        yield competition_pb2.SelectionReply(testId=test_id)
            
            # Second pass: select remaining based on score
            for test_id, score in sorted_tests:
                if selected_count >= target_selections:
                    break
                    
                if test_id not in self.selection_history:  # Skip already selected
                    self.selection_history.append(test_id)
                    selected_count += 1
                    self.logger.info(f"Selected test {test_id} with score {score:.3f}")
                    yield competition_pb2.SelectionReply(testId=test_id)
            
            self.logger.info(f"Selection complete. Selected {selected_count}/{total_tests} tests")
            
        except Exception as e:
            self.logger.error(f"Error during selection: {str(e)}")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", required=True)
    args = parser.parse_args()
    
    server = grpc.server(ThreadPoolExecutor(max_workers=4))
    competition_pb2_grpc.add_CompetitionToolServicer_to_server(
        BalancedTestSelector(), server)
    
    server.add_insecure_port(f"[::]:{args.port}")
    server.start()
    print(f"Server started on port {args.port}")
    server.wait_for_termination()