"""OutRoadMB — Outlier-based Road geometry prioritizer.

gRPC Competition Tool implementing the CompetitionTool service interface.
Delegates prioritization to domain-layer strategies that rank SDC test cases
by road geometry anomaly — tests with unusual shapes (sharp turns, abnormal
distances, low safety margins) are prioritized first to detect faults earlier.

Approach:
    1. Extract 8 road geometry features per test case (distance metrics,
       angle statistics, road safety via inflection-point detection and
       Shoelace formula).
    2. Compute anomaly scores using outlier detection (Euclidean z-score
       or Mahalanobis distance in feature space).
    3. Return test cases ordered by decreasing anomaly score.

The tool is unsupervised — it does not require training data from the
Initialize phase, though it properly consumes the Oracle stream as
required by the competition protocol.

Authors: Marcello B.
"""

import os
import sys
import logging
import argparse
import time
import concurrent.futures as fut

import grpc
import competition_2026_pb2
import competition_2026_pb2_grpc

from strategies import (
    TestCaseData,
    PrioritizationStrategy,
    get_strategy,
    available_strategies,
    STRATEGY_REGISTRY,
)

# ---------------------------------------------------------------------------
#   Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("OutRoadMB")


# ---------------------------------------------------------------------------
#   Protobuf ↔ Domain Converters
# ---------------------------------------------------------------------------
def _sdc_test_case_to_domain(pb: competition_2026_pb2.SDCTestCase) -> TestCaseData:
    """Convert a gRPC SDCTestCase message to our domain TestCaseData.

    Mapping:
        pb.testId           →  TestCaseData.test_id
        pb.roadPoints[i].x  →  TestCaseData.road_points[i][0]
        pb.roadPoints[i].y  →  TestCaseData.road_points[i][1]
    """
    road_points = [(pt.x, pt.y) for pt in pb.roadPoints]
    return TestCaseData(test_id=pb.testId, road_points=road_points)


# ---------------------------------------------------------------------------
#   Oracle Store — holds initialization data from the evaluator
# ---------------------------------------------------------------------------
class OracleStore:
    """Stores historical test results received during Initialize.

    The evaluator sends Oracle messages containing:
        - testCase (SDCTestCase): road geometry
        - hasFailed (bool): whether the test detected a failure

    This data can be used by strategies that benefit from training data
    (e.g., supervised classifiers). Currently our strategies are unsupervised,
    so we store the data for future extensibility but don't require it.
    """

    def __init__(self):
        self.oracles: list[tuple[TestCaseData, bool]] = []

    def ingest(self, oracle: competition_2026_pb2.Oracle):
        tc = _sdc_test_case_to_domain(oracle.testCase)
        self.oracles.append((tc, oracle.hasFailed))

    def clear(self):
        self.oracles.clear()

    @property
    def size(self) -> int:
        return len(self.oracles)


# ---------------------------------------------------------------------------
#   gRPC Servicer — implements CompetitionToolServicer
# ---------------------------------------------------------------------------
class OutRoadMBServicer(competition_2026_pb2_grpc.CompetitionToolServicer):
    """gRPC adapter that delegates prioritization to domain strategies.

    Strategy selection:
        Configured via --strategy CLI flag (or STRATEGY env var).
        Defaults to 'mahalanobis-outlier-first'.
    """

    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.strategy: PrioritizationStrategy = get_strategy(strategy_name)
        self.oracle_store = OracleStore()
        logger.info(
            "Servicer initialized — strategy='%s' (%s)",
            strategy_name,
            type(self.strategy).__name__,
        )

    def Name(self, request, context):
        """Return tool name — used by the evaluator to identify us."""
        return competition_2026_pb2.NameReply(name=f"OutRoadMB-{self.strategy_name}")

    def Initialize(self, request_iterator, context):
        """Ingest Oracle stream (historical test results as training data).

        Required by competition protocol. We consume the full stream and
        store oracle data for potential future use by supervised strategies.
        """
        self.oracle_store.clear()
        count = 0
        for oracle in request_iterator:
            self.oracle_store.ingest(oracle)
            count += 1

        logger.info("Initialize complete — received %d oracle entries", count)
        return competition_2026_pb2.InitializationReply(ok=True)

    def Prioritize(self, request_iterator, context):
        """Prioritize a stream of SDCTestCase messages.

        Flow:
            1. Consume the full input stream → List[SDCTestCase]
            2. Convert protobuf → domain: SDCTestCase → TestCaseData
            3. Delegate to strategy: strategy.prioritize(test_cases)
            4. Yield PrioritizationReply in prioritized order
        """
        start = time.perf_counter()

        # 1. Consume stream
        pb_test_cases = list(request_iterator)

        # 2. Convert protobuf → domain objects
        domain_test_cases = [_sdc_test_case_to_domain(tc) for tc in pb_test_cases]

        logger.info("Prioritize called — %d test cases", len(domain_test_cases))

        # 3. Delegate to strategy
        prioritized_ids: list[str] = self.strategy.prioritize(domain_test_cases)

        elapsed = time.perf_counter() - start
        logger.info("Prioritization complete in %.4fs", elapsed)

        # 4. Yield results
        for test_id in prioritized_ids:
            yield competition_2026_pb2.PrioritizationReply(testId=test_id)


# ---------------------------------------------------------------------------
#   Server Bootstrap
# ---------------------------------------------------------------------------
def serve(port: str, strategy_name: str):
    server = grpc.server(fut.ThreadPoolExecutor(max_workers=4))

    servicer = OutRoadMBServicer(strategy_name)
    competition_2026_pb2_grpc.add_CompetitionToolServicer_to_server(servicer, server)

    bind_address = f"[::]:{port}"
    server.add_insecure_port(bind_address)

    logger.info("Starting gRPC server on %s", bind_address)
    logger.info("Strategy: %s", strategy_name)
    server.start()
    logger.info("Server is running")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.stop(grace=5)

    logger.info("Server terminated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OutRoadMB — SDC Test Prioritizer (gRPC Competition Tool)"
    )
    parser.add_argument(
        "-p", "--port",
        default=os.environ.get("GRPC_PORT", "50051"),
        help="Port to listen on (default: 50051 or $GRPC_PORT)",
    )
    parser.add_argument(
        "-s", "--strategy",
        default=os.environ.get("STRATEGY", "mahalanobis-outlier-first"),
        help="Prioritization strategy (default: mahalanobis-outlier-first)",
    )
    args = parser.parse_args()

    if args.strategy not in STRATEGY_REGISTRY:
        print(f"ERROR: Unknown strategy '{args.strategy}'")
        print(f"Available: {', '.join(available_strategies())}")
        sys.exit(1)

    serve(port=args.port, strategy_name=args.strategy)
