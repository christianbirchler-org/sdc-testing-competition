#!/usr/bin/env python3
"""
SDC Competition 2026 - Sample Test Selector
==========================================

This tool demonstrates the new 2026 competition interface with metrics
from the roadmap paper discussed with Christian Birchler (28-08-25).

New capabilities:
- Reality gap assessment
- Cost-effectiveness analysis  
- Safety evaluation
- Oracle quality metrics
- Selection confidence and reasoning

Author: Prakash Aryan, University of Bern
Date: September 2025
"""

import random
import argparse
import time
import grpc
import concurrent.futures as fut
import competition_pb2_grpc
import competition_pb2


class SampleTestSelector2026(competition_pb2_grpc.CompetitionToolServicer):
    """
    Sample test selector implementing 2026 competition interface.
    Shows backward compatibility while demonstrating new metrics.
    """

    def __init__(self):
        """Initialize the 2026 sample selector."""
        self.tool_name = "sample_test_selector_2026"
        self.version = "2026.1.0"
        self.training_tests = []
        
        print(f"Initializing {self.tool_name} v{self.version}")
        print("Supports: Reality Gap, Cost-Effectiveness, Safety Assessment")

    def Name(self, request, context):
        """Return tool name and version."""
        return competition_pb2.NameReply(name=self.tool_name)

    def Initialize(self, request_iterator, context):
        """Process training data with 2026 metrics analysis."""
        print("Processing training data with 2026 metrics...")
        
        test_count = 0
        failure_count = 0
        total_execution_time = 0.0
        
        for oracle in request_iterator:
            oracle: competition_pb2.Oracle = oracle
            self.training_tests.append(oracle)
            test_count += 1
            
            # Basic analysis
            if oracle.hasFailed:
                failure_count += 1
            
            # Check for 2026 metrics
            if oracle.HasField('executionTime'):
                total_execution_time += oracle.executionTime
            
            if oracle.HasField('metrics'):
                metrics = oracle.metrics
                print(f"  Test {oracle.testCase.testId[:8]}: "
                      f"Reality Gap={getattr(metrics, 'realityGapScore', 0.0):.2f}, "
                      f"Safety Risk={getattr(metrics, 'safetyRisk', 0.0):.2f}")
        
        failure_rate = failure_count / max(test_count, 1)
        avg_execution_time = total_execution_time / max(test_count, 1)
        
        print(f"Training complete: {test_count} tests, {failure_rate:.1%} failure rate")
        print(f"Average execution time: {avg_execution_time:.2f}s")
        
        return competition_pb2.InitializationReply(
            ok=True,
            message=f"Processed {test_count} training cases with 2026 metrics"
        )

    def Select(self, request_iterator, context):
        """
        Select tests using 2026 criteria with confidence and reasoning.
        
        Selection strategy combines:
        - Random baseline selection
        - Confidence scoring based on road complexity  
        - Reasoning for each selection decision
        """
        print("Starting test selection with 2026 reasoning...")
        
        for sdc_test_case in request_iterator:
            sdc_test_case: competition_pb2.SDCTestCase = sdc_test_case
            
            # Calculate road complexity for confidence scoring
            road_length = len(sdc_test_case.roadPoints)
            complexity = min(road_length / 50.0, 1.0)  # Normalize to [0,1]
            
            # Selection decision with probability based on complexity
            selection_probability = 0.3 + (complexity * 0.4)  # 30-70% range
            should_select = random.random() < selection_probability
            
            if should_select:
                # Generate confidence based on road characteristics
                confidence = 0.5 + (complexity * 0.4) + (random.random() * 0.1)
                confidence = min(confidence, 1.0)
                
                # Generate reasoning based on selection criteria
                if complexity > 0.7:
                    reasoning = "High complexity road geometry - likely to reveal edge cases"
                elif complexity > 0.4:
                    reasoning = "Moderate complexity - good balance for fault detection"
                else:
                    reasoning = "Simple geometry selected for baseline coverage"
                
                print(f"  Selected {sdc_test_case.testId[:8]}: "
                      f"confidence={confidence:.2f}, complexity={complexity:.2f}")
                
                yield competition_pb2.SelectionReply(
                    testId=sdc_test_case.testId,
                    confidence=confidence,
                    reasoning=reasoning
                )

    def GetSupportedMetrics(self, request, context):
        """Report 2026 metrics capabilities."""
        return competition_pb2.MetricsSupport(
            supportsRealityGap=True,
            supportsCostEffectiveness=True, 
            supportsSafetyAssessment=True,
            supportsOracleQuality=True,
            toolVersion=self.version
        )


def main():
    """Run the 2026 sample test selector."""
    parser = argparse.ArgumentParser(
        description='SDC Competition 2026 Sample Test Selector',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sampletool_2026.py -p 4546
  python sampletool_2026.py --port 4547 --verbose
        """
    )
    parser.add_argument("-p", "--port", default="4546", 
                        help="gRPC server port (default: 4546)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        print("Verbose mode activated")
    
    GRPC_PORT = args.port
    GRPC_URL = f"[::]:{GRPC_PORT}"
    
    # Start gRPC server
    server = grpc.server(fut.ThreadPoolExecutor(max_workers=4))
    competition_pb2_grpc.add_CompetitionToolServicer_to_server(
        SampleTestSelector2026(), server
    )
    
    server.add_insecure_port(GRPC_URL)
    print(f"Starting SDC 2026 sample selector on port {GRPC_PORT}")
    print("Ready to receive evaluation requests...")
    print("Press Ctrl+C to stop")
    
    try:
        server.start()
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        server.stop(grace=2)


if __name__ == "__main__":
    main()