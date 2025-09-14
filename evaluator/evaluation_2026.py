#!/usr/bin/env python3
"""
SDC Competition 2026 - Tool Evaluator
====================================

Evaluates both legacy (2025) and 2026 competition tools.
Supports new metrics from roadmap paper discussion with Christian.

New evaluation criteria:
- Reality gap assessment
- Cost-effectiveness analysis
- Safety evaluation with human alignment
- Oracle quality metrics
- Selection confidence and reasoning analysis

Author: Prakash Aryan, University of Bern  
Date: September 2025
"""

import argparse
import json
import os
import sys
import time
import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from sklearn.metrics.pairwise import pairwise_distances
from shapely.geometry import LineString

import grpc
import competition_pb2_grpc
import competition_pb2


@dataclass
class TestDetails:
    """Test case information and oracle data."""
    test_id: str
    hasFailed: bool
    sim_time: float
    road_points: list


@dataclass
class EvaluationReport2026:
    """2026 evaluation report with traditional and new metrics."""
    # Traditional metrics (maintain compatibility)
    tool_name: str
    benchmark: str
    test_suite_count: int
    selection_count: int
    time_to_initialize: float
    time_to_select_tests: float
    time_to_fault_ratio: float
    fault_to_selection_ratio: float
    diversity: float
    
    # 2026 metrics
    tool_version: Optional[str] = None
    has_confidence_scores: bool = False
    has_reasoning: bool = False
    average_confidence: Optional[float] = None
    supports_reality_gap: bool = False
    supports_cost_effectiveness: bool = False
    supports_safety_assessment: bool = False
    supports_oracle_quality: bool = False


class MetricEvaluator2026:
    """Evaluator with both traditional and 2026 metrics."""

    def __init__(self):
        pass

    def time_to_fault_ratio(self, test_suite: list[TestDetails], selection: list[str]) -> float:
        """Traditional metric: simulation time per detected fault."""
        detected_faults = 0
        total_sim_time = 0.0
        test_mapping = {test.test_id: test for test in test_suite}

        for test_id in selection:
            if test_id not in test_mapping:
                continue
            test = test_mapping[test_id]
            if test.hasFailed:
                detected_faults += 1
            total_sim_time += test.sim_time

        return total_sim_time / max(detected_faults, 1)

    def fault_to_selection_ratio(self, test_suite: list[TestDetails], selection: list[str]) -> float:
        """Traditional metric: proportion of selected tests that fail."""
        detected_faults = 0
        test_mapping = {test.test_id: test for test in test_suite}

        for test_id in selection:
            if test_id not in test_mapping:
                continue
            test = test_mapping[test_id]
            if test.hasFailed:
                detected_faults += 1

        return detected_faults / max(len(selection), 1)

    def diversity(self, test_suite: list[TestDetails], selection: list[str]) -> float:
        """Traditional metric: diversity based on road curvature."""
        selected_profiles = []

        for test_detail in test_suite:
            if test_detail.test_id in selection:
                profile = self._calculate_curvature_profile(test_detail)
                profile_stats = [np.mean(profile), np.std(profile)]
                selected_profiles.append(profile_stats)

        if len(selected_profiles) < 2:
            return 0.0

        distances = pairwise_distances(selected_profiles, selected_profiles)
        return float(np.mean(distances))

    def _calculate_curvature_profile(self, test_detail: TestDetails):
        """Calculate road curvature for diversity analysis."""
        if len(test_detail.road_points) < 3:
            return [0.0]

        curvatures = []
        points = test_detail.road_points
        
        for i in range(1, len(points) - 1):
            p1, p2, p3 = points[i-1], points[i], points[i+1]
            
            v1 = np.array(p2) - np.array(p1)
            v2 = np.array(p3) - np.array(p2)
            
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                curvature = 1.0 - cos_angle
                curvatures.append(curvature)
        
        return curvatures if curvatures else [0.0]


def _create_oracle_iterator(train_set: list[TestDetails]):
    """Generate Oracle messages for tool initialization."""
    for test_detail in train_set:
        road_points = [
            competition_pb2.RoadPoint(sequenceNumber=i, x=pts[0], y=pts[1]) 
            for i, pts in enumerate(test_detail.road_points)
        ]
        test_case = competition_pb2.SDCTestCase(
            testId=test_detail.test_id, 
            roadPoints=road_points
        )
        
        oracle = competition_pb2.Oracle(
            testCase=test_case, 
            hasFailed=test_detail.hasFailed
        )
        
        # Add 2026 metrics if available
        oracle.executionTime = test_detail.sim_time
        oracle.failureDetails = f"Test outcome: {'FAIL' if test_detail.hasFailed else 'PASS'}"
        
        yield oracle


def _create_test_case_iterator(test_set: list[TestDetails]):
    """Generate SDCTestCase messages for selection."""
    for test_detail in test_set:
        road_points = [
            competition_pb2.RoadPoint(sequenceNumber=i, x=pts[0], y=pts[1]) 
            for i, pts in enumerate(test_detail.road_points)
        ]
        yield competition_pb2.SDCTestCase(
            testId=test_detail.test_id,
            roadPoints=road_points
        )


class TestLoader2026:
    """Load test data from JSON files (SensoDat support removed for simplicity)."""

    def __init__(self, file_path: str, training_proportion: float = 0.8):
        """Initialize with JSON test file."""
        self.file_path = file_path
        self.training_proportion = training_proportion
        
        with open(file_path, 'r') as fp:
            raw_tests = json.load(fp)
        
        self.test_details = self._parse_test_data(raw_tests)
        split_index = int(training_proportion * len(self.test_details))
        self.training_set = self.test_details[:split_index]
        self.test_set = self.test_details[split_index:]

    def _parse_test_data(self, raw_tests) -> list[TestDetails]:
        """Convert raw JSON to TestDetails objects."""
        test_details = []
        for raw_test in raw_tests:
            test_id = raw_test['_id']['$oid']
            road_points = [(pt['x'], pt['y']) for pt in raw_test['road_points']]
            has_failed = raw_test['meta_data']['test_info']['test_outcome'] == "FAIL"
            sim_time = raw_test['meta_data']['test_info']['test_duration']
            
            test_details.append(TestDetails(
                test_id=test_id,
                hasFailed=has_failed,
                sim_time=sim_time,
                road_points=road_points
            ))
        
        return test_details

    def get_benchmark_name(self) -> str:
        """Return benchmark identifier."""
        return self.file_path


class ToolEvaluator2026:
    """Main evaluator for 2026 competition."""

    def __init__(self, test_loader: TestLoader2026):
        """Initialize with test data."""
        self.test_loader = test_loader
        self.metric_evaluator = MetricEvaluator2026()

    def evaluate(self, stub: competition_pb2_grpc.CompetitionToolStub) -> EvaluationReport2026:
        """Run complete tool evaluation."""
        print("Starting 2026 tool evaluation...")
        
        # Get tool name
        name_reply = stub.Name(competition_pb2.Empty())
        tool_name = name_reply.name
        print(f"Evaluating: {tool_name}")
        
        # Check 2026 capabilities
        tool_version = None
        capabilities = {}
        try:
            metrics_support = stub.GetSupportedMetrics(competition_pb2.Empty())
            tool_version = metrics_support.toolVersion
            capabilities = {
                'reality_gap': metrics_support.supportsRealityGap,
                'cost_effectiveness': metrics_support.supportsCostEffectiveness,
                'safety_assessment': metrics_support.supportsSafetyAssessment,
                'oracle_quality': metrics_support.supportsOracleQuality,
            }
            print(f"Tool version: {tool_version}")
            print(f"2026 capabilities: {[k for k, v in capabilities.items() if v]}")
        except Exception:
            print("Tool uses legacy interface (pre-2026)")
        
        # Initialize tool
        print("Initializing with training data...")
        start_time = time.time()
        init_response = stub.Initialize(_create_oracle_iterator(self.test_loader.training_set))
        init_time = time.time() - start_time
        
        if not init_response.ok:
            raise RuntimeError("Tool initialization failed")
        
        if init_response.HasField('message'):
            print(f"Tool message: {init_response.message}")
        
        # Get test selections
        print("Collecting test selections...")
        start_time = time.time()
        selection_iterator = stub.Select(_create_test_case_iterator(self.test_loader.test_set))
        
        selections = []
        confidence_scores = []
        reasoning_count = 0
        
        for selection_reply in selection_iterator:
            selections.append(selection_reply.testId)
            
            # Check 2026 fields
            if selection_reply.HasField('confidence'):
                confidence_scores.append(selection_reply.confidence)
                
            if selection_reply.HasField('reasoning'):
                reasoning_count += 1
                print(f"  {selection_reply.testId[:8]}: {selection_reply.reasoning}")
        
        selection_time = time.time() - start_time
        
        # Calculate metrics
        traditional_metrics = {
            'time_to_fault_ratio': self.metric_evaluator.time_to_fault_ratio(
                self.test_loader.test_set, selections
            ),
            'fault_to_selection_ratio': self.metric_evaluator.fault_to_selection_ratio(
                self.test_loader.test_set, selections
            ),
            'diversity': self.metric_evaluator.diversity(
                self.test_loader.test_set, selections
            )
        }
        
        # Create evaluation report
        report = EvaluationReport2026(
            # Traditional fields
            tool_name=tool_name,
            benchmark=self.test_loader.get_benchmark_name(),
            test_suite_count=len(self.test_loader.test_set),
            selection_count=len(selections),
            time_to_initialize=init_time,
            time_to_select_tests=selection_time,
            time_to_fault_ratio=traditional_metrics['time_to_fault_ratio'],
            fault_to_selection_ratio=traditional_metrics['fault_to_selection_ratio'],
            diversity=traditional_metrics['diversity'],
            
            # 2026 fields
            tool_version=tool_version,
            has_confidence_scores=len(confidence_scores) > 0,
            has_reasoning=reasoning_count > 0,
            average_confidence=np.mean(confidence_scores) if confidence_scores else None,
            supports_reality_gap=capabilities.get('reality_gap', False),
            supports_cost_effectiveness=capabilities.get('cost_effectiveness', False),
            supports_safety_assessment=capabilities.get('safety_assessment', False),
            supports_oracle_quality=capabilities.get('oracle_quality', False),
        )
        
        return report


def save_results(report: EvaluationReport2026, output_path: Path):
    """Save evaluation results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create header
    fields = [
        'tool_name', 'benchmark', 'test_suite_count', 'selection_count',
        'time_to_initialize', 'time_to_select_tests', 'time_to_fault_ratio',
        'fault_to_selection_ratio', 'diversity', 'tool_version',
        'has_confidence_scores', 'has_reasoning', 'average_confidence',
        'supports_reality_gap', 'supports_cost_effectiveness',
        'supports_safety_assessment', 'supports_oracle_quality'
    ]
    
    # Write results
    if not output_path.exists():
        with open(output_path, 'w') as f:
            f.write(','.join(fields) + '\n')
    
    values = [getattr(report, field, '') for field in fields]
    with open(output_path, 'a') as f:
        f.write(','.join(map(str, values)) + '\n')


def main():
    """Run the 2026 evaluation tool."""
    parser = argparse.ArgumentParser(
        description='SDC Competition 2026 Tool Evaluator',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-u", "--url", required=True, 
                        help="Tool gRPC URL (e.g., localhost:4546)")
    parser.add_argument("-t", "--tests", required=True,
                        help="Path to test data JSON file")
    parser.add_argument("-o", "--output", default="results_2026.csv",
                        help="Output CSV file (default: results_2026.csv)")
    
    args = parser.parse_args()
    
    try:
        # Load test data
        print(f"Loading test data from {args.tests}")
        test_loader = TestLoader2026(args.tests)
        print(f"Loaded {len(test_loader.training_set)} training tests, "
              f"{len(test_loader.test_set)} evaluation tests")
        
        # Connect to tool
        print(f"Connecting to tool at {args.url}")
        channel = grpc.insecure_channel(args.url)
        stub = competition_pb2_grpc.CompetitionToolStub(channel)
        
        # Run evaluation
        evaluator = ToolEvaluator2026(test_loader)
        report = evaluator.evaluate(stub)
        
        # Save and display results
        output_path = Path(args.output)
        save_results(report, output_path)
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print("="*60)
        print(f"Tool: {report.tool_name}")
        if report.tool_version:
            print(f"Version: {report.tool_version}")
        print(f"Selected: {report.selection_count}/{report.test_suite_count} tests")
        print(f"Fault detection rate: {report.fault_to_selection_ratio:.1%}")
        print(f"Time to initialize: {report.time_to_initialize:.3f}s")
        print(f"Time to select: {report.time_to_select_tests:.3f}s")
        
        if report.has_confidence_scores:
            print(f"Average confidence: {report.average_confidence:.2f}")
        if report.has_reasoning:
            print("Provides selection reasoning")
        
        capabilities = []
        if report.supports_reality_gap:
            capabilities.append("Reality Gap")
        if report.supports_cost_effectiveness:
            capabilities.append("Cost-Effectiveness") 
        if report.supports_safety_assessment:
            capabilities.append("Safety Assessment")
        if report.supports_oracle_quality:
            capabilities.append("Oracle Quality")
        
        if capabilities:
            print(f"2026 capabilities: {', '.join(capabilities)}")
        
        print(f"\nResults saved to: {output_path}")
        print("="*60)
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()