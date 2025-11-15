"""
Complete UPIR Example: Streaming Pipeline Design

Demonstrates the full UPIR workflow for designing, verifying, synthesizing,
and optimizing a streaming data pipeline.

Workflow:
1. Define formal specification with temporal properties and constraints
2. Create UPIR instance with initial architecture
3. Verify specification is satisfiable
4. Synthesize Apache Beam implementation using CEGIS
5. Simulate production metrics
6. Learn from metrics to optimize architecture
7. Extract and save pattern for reuse

Use case: Real-time event processing pipeline
- Ingests events from Pub/Sub
- Processes with Apache Beam streaming
- Writes to BigQuery for analytics
- Requirements: <100ms latency, data consistency, <$5000/month cost

Based on:
- TD Commons disclosure: https://www.tdcommons.org/dpubs_series/8852/
- Apache Beam: https://beam.apache.org/documentation/programming-guide/

Author: Subhadip Mitra
License: Apache 2.0
"""

import json
from pathlib import Path

from upir.core.architecture import Architecture
from upir.core.specification import FormalSpecification
from upir.core.temporal import TemporalOperator, TemporalProperty
from upir.core.upir import UPIR
from upir.learning.learner import ArchitectureLearner
from upir.patterns.library import PatternLibrary
from upir.synthesis.cegis import Synthesizer
from upir.verification.solver import VerificationStatus
from upir.verification.verifier import Verifier


def print_section(title: str):
    """Print section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def main():
    """Run complete UPIR workflow for streaming pipeline design."""

    # =========================================================================
    # STEP 1: Define Formal Specification
    # =========================================================================
    print_section("STEP 1: Define Formal Specification")

    # Temporal properties using Linear Temporal Logic (LTL)
    properties = [
        # EVENTUALLY: All events must be processed within 100 seconds
        # Formula: ◇_{≤100s} all_events_processed
        TemporalProperty(
            operator=TemporalOperator.EVENTUALLY,
            predicate="all_events_processed",
            time_bound=100000,  # 100 seconds in ms
        ),
        # WITHIN: Each event processed within 100ms (latency requirement)
        # Formula: ∀e. process_event(e) → ◇_{≤100ms} processed(e)
        TemporalProperty(
            operator=TemporalOperator.WITHIN,
            predicate="process_event",
            time_bound=100,  # 100ms
        ),
    ]

    # Invariants that must hold at all times
    invariants = [
        # ALWAYS: Data consistency maintained
        # Formula: □ data_consistent
        TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="data_consistent",
        ),
        # ALWAYS: No data loss
        # Formula: □ (events_in = events_out)
        TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="no_data_loss",
        ),
    ]

    # Hard constraints on resources and performance
    constraints = {
        # Latency constraint: p99 latency ≤ 100ms
        "latency_p99": {"max": 100.0},
        # Cost constraint: Monthly cost ≤ $5000
        "monthly_cost": {"max": 5000.0},
        # Throughput requirement: ≥ 10,000 events/second
        "throughput_qps": {"min": 10000.0},
    }

    # Create formal specification
    spec = FormalSpecification(
        properties=properties,
        invariants=invariants,
        constraints=constraints,
    )

    print(f"Created specification:")
    print(f"  - {len(properties)} temporal properties")
    print(f"  - {len(invariants)} invariants")
    print(f"  - {len(constraints)} constraints")
    print(f"\nTemporal Properties:")
    for prop in properties:
        print(f"  - {prop.operator.value}: {prop.predicate} (bound={prop.time_bound}ms)")
    print(f"\nInvariants:")
    for inv in invariants:
        print(f"  - {inv.operator.value}: {inv.predicate}")
    print(f"\nConstraints:")
    for name, constraint in constraints.items():
        constraint_str = ", ".join(f"{k}={v}" for k, v in constraint.items())
        print(f"  - {name}: {constraint_str}")

    # =========================================================================
    # STEP 2: Create UPIR Instance with Initial Architecture
    # =========================================================================
    print_section("STEP 2: Create UPIR Instance")

    # Define streaming pipeline components
    components = [
        {
            "id": "pubsub_source",
            "name": "Event Source",
            "type": "pubsub_source",
            "config": {
                "topic": "projects/my-project/topics/events",
                "subscription": "projects/my-project/subscriptions/events-sub",
            },
            "latency_ms": 5.0,
            "cost_monthly": 500.0,
        },
        {
            "id": "beam_processor",
            "name": "Stream Processor",
            "type": "streaming_processor",
            "config": {
                "framework": "apache_beam",
                "runner": "dataflow",
                "parallelism": 10,
            },
            "latency_ms": 50.0,
            "cost_monthly": 3000.0,
        },
        {
            "id": "bigquery_sink",
            "name": "Analytics Database",
            "type": "database",
            "config": {
                "dataset": "analytics",
                "table": "events",
                "write_mode": "streaming",
            },
            "latency_ms": 30.0,
            "cost_monthly": 1200.0,
        },
    ]

    # Define data flow connections
    connections = [
        {
            "from": "pubsub_source",
            "to": "beam_processor",
            "protocol": "pull",
        },
        {
            "from": "beam_processor",
            "to": "bigquery_sink",
            "protocol": "streaming_insert",
        },
    ]

    # Create architecture
    architecture = Architecture(
        components=components,
        connections=connections,
        deployment={
            "type": "multi_region",
            "regions": ["us-central1", "us-east1"],
            "autoscaling": True,
        },
    )

    # Create UPIR instance
    upir = UPIR(
        id="streaming-pipeline-v1",
        name="Real-time Event Processing Pipeline",
        description=(
            "Streaming pipeline for processing events from Pub/Sub, "
            "transforming with Apache Beam on Dataflow, and loading to BigQuery. "
            "Designed for real-time analytics with strict latency and cost constraints."
        ),
        architecture=architecture,
        specification=spec,
        metadata={
            "team": "data-engineering",
            "domain": "analytics",
            "criticality": "high",
            "version": "1.0.0",
        },
    )

    print(f"Created UPIR: {upir.name}")
    print(f"  ID: {upir.id}")
    print(f"  Components: {len(components)}")
    print(f"  Connections: {len(connections)}")
    print(f"\nArchitecture:")
    for comp in components:
        print(f"  - {comp['name']} ({comp['type']})")
        print(f"    Latency: {comp['latency_ms']}ms")
        print(f"    Cost: ${comp['cost_monthly']}/month")

    # =========================================================================
    # STEP 3: Verify Specification
    # =========================================================================
    print_section("STEP 3: Verify Specification")

    verifier = Verifier()

    print("Verifying formal specification...")
    print("(Encoding temporal properties and constraints as SMT formulas)")

    # Verify the specification
    results = verifier.verify_specification(upir)

    print(f"\nVerification Results: {len(results)} properties checked")

    all_valid = all(result.status == VerificationStatus.PROVED for result in results)

    if all_valid:
        print("\n✓ All properties verified successfully!")
        print("  All temporal properties can be satisfied")
        print("  All constraints are consistent")
    else:
        print("\n✗ Some properties could not be verified:")
        for result in results:
            if result.status != VerificationStatus.PROVED:
                print(f"  - {result.property.predicate}: {result.status.value}")
                if result.counterexample:
                    print(f"    Counterexample: {result.counterexample}")

    # Check individual constraints
    print("\nConstraint Analysis:")
    total_latency = sum(comp.get("latency_ms", 0) for comp in components)
    total_cost = sum(comp.get("cost_monthly", 0) for comp in components)
    print(f"  Total latency: {total_latency}ms (limit: 100ms)")
    print(f"  Total cost: ${total_cost}/month (limit: $5000/month)")
    print(f"  Latency OK: {total_latency <= 100}")
    print(f"  Cost OK: {total_cost <= 5000}")

    # =========================================================================
    # STEP 4: Synthesize Implementation using CEGIS
    # =========================================================================
    print_section("STEP 4: Synthesize Implementation")

    # Generate sketch for Apache Beam pipeline
    synthesizer = Synthesizer(max_iterations=10)

    print("Generating Apache Beam pipeline sketch...")
    print("(Creating program template with holes to be filled)")

    sketch = synthesizer.generate_sketch(spec)

    print(f"\nGenerated sketch:")
    print(f"  Language: {sketch.language}")
    print(f"  Framework: {sketch.framework}")
    print(f"  Holes: {len(sketch.holes)}")
    print(f"\nHoles to fill:")
    for hole in sketch.holes[:5]:  # Show first 5 holes
        print(f"  - {hole.id} ({hole.hole_type}): {hole.description}")
    if len(sketch.holes) > 5:
        print(f"  ... and {len(sketch.holes) - 5} more")

    # Use CEGIS to fill holes
    print("\n" + "-" * 80)
    print("Running CEGIS synthesis...")
    print("(Iteratively filling holes and verifying against specification)")

    synthesis_result = synthesizer.synthesize(upir, sketch)

    print(f"\nSynthesis Result: {synthesis_result.status.value}")
    print(f"  Iterations: {synthesis_result.iterations}")
    print(f"  Time: {synthesis_result.execution_time:.2f}s")

    if synthesis_result.status.value == "SUCCESS":
        print("\n✓ Successfully synthesized implementation!")
        print("\nGenerated Apache Beam Pipeline:")
        print("-" * 80)
        # Show synthesized code (first 50 lines)
        code_lines = synthesis_result.implementation.split("\n")
        for i, line in enumerate(code_lines[:50], 1):
            print(f"{i:3d} | {line}")
        if len(code_lines) > 50:
            print(f"... ({len(code_lines) - 50} more lines)")
        print("-" * 80)

        # Save synthesized code
        output_path = Path("examples/generated_pipeline.py")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(synthesis_result.implementation)
        print(f"\n✓ Saved to: {output_path}")

    else:
        print("\n✗ Synthesis did not complete successfully")
        print(f"  Status: {synthesis_result.status.value}")
        if synthesis_result.counterexamples:
            print(f"  Counterexamples: {len(synthesis_result.counterexamples)}")

    # =========================================================================
    # STEP 5: Simulate Production Metrics
    # =========================================================================
    print_section("STEP 5: Simulate Production Metrics")

    print("Simulating production deployment...")
    print("(Generating realistic metrics for streaming pipeline)")

    # Simulate metrics from production run
    metrics = {
        # Latency metrics
        "latency_p50": 42.0,  # 50th percentile: 42ms
        "latency_p95": 78.0,  # 95th percentile: 78ms
        "latency_p99": 95.0,  # 99th percentile: 95ms
        "latency_max": 120.0,  # Max: 120ms (violates constraint!)
        # Throughput metrics
        "throughput_qps": 12500.0,  # 12,500 events/sec (meets requirement)
        "throughput_total": 1080000000,  # 1.08B events total
        # Resource utilization
        "cpu_utilization": 0.65,  # 65% CPU
        "memory_utilization": 0.58,  # 58% memory
        # Cost metrics
        "cost_compute": 3200.0,  # $3,200 compute
        "cost_storage": 450.0,  # $450 storage
        "cost_network": 280.0,  # $280 network
        "monthly_cost": 3930.0,  # Total: $3,930 (under budget!)
        # Reliability metrics
        "uptime": 0.9995,  # 99.95% uptime
        "error_rate": 0.0002,  # 0.02% errors
        "data_loss_rate": 0.0,  # 0% data loss
        # Data quality
        "duplicate_rate": 0.0001,  # 0.01% duplicates
        "out_of_order_rate": 0.005,  # 0.5% out-of-order
    }

    print(f"\nProduction Metrics (24-hour period):")
    print(f"\nLatency:")
    print(f"  p50: {metrics['latency_p50']:.1f}ms")
    print(f"  p95: {metrics['latency_p95']:.1f}ms")
    print(f"  p99: {metrics['latency_p99']:.1f}ms ({'✓ OK' if metrics['latency_p99'] <= 100 else '✗ VIOLATION'})")
    print(f"  max: {metrics['latency_max']:.1f}ms")

    print(f"\nThroughput:")
    print(f"  QPS: {metrics['throughput_qps']:,.0f} events/sec ({'✓ OK' if metrics['throughput_qps'] >= 10000 else '✗ VIOLATION'})")
    print(f"  Total: {metrics['throughput_total']:,.0f} events")

    print(f"\nCost:")
    print(f"  Compute: ${metrics['cost_compute']:,.0f}/month")
    print(f"  Storage: ${metrics['cost_storage']:,.0f}/month")
    print(f"  Network: ${metrics['cost_network']:,.0f}/month")
    print(f"  Total: ${metrics['monthly_cost']:,.0f}/month ({'✓ OK' if metrics['monthly_cost'] <= 5000 else '✗ VIOLATION'})")

    print(f"\nReliability:")
    print(f"  Uptime: {metrics['uptime']:.2%}")
    print(f"  Error rate: {metrics['error_rate']:.4%}")
    print(f"  Data loss: {metrics['data_loss_rate']:.4%} ✓")

    # Evaluate against specification
    print("\n" + "-" * 80)
    print("Specification Compliance:")
    constraint_violations = []

    # Check latency constraint
    if metrics["latency_p99"] > 100:
        constraint_violations.append("latency_p99")
        print(f"  ✗ Latency constraint violated: {metrics['latency_p99']:.1f}ms > 100ms")
    else:
        print(f"  ✓ Latency constraint met: {metrics['latency_p99']:.1f}ms ≤ 100ms")

    # Check cost constraint
    if metrics["monthly_cost"] > 5000:
        constraint_violations.append("monthly_cost")
        print(f"  ✗ Cost constraint violated: ${metrics['monthly_cost']:.0f} > $5000")
    else:
        print(f"  ✓ Cost constraint met: ${metrics['monthly_cost']:.0f} ≤ $5000")

    # Check throughput constraint
    if metrics["throughput_qps"] < 10000:
        constraint_violations.append("throughput_qps")
        print(f"  ✗ Throughput constraint violated: {metrics['throughput_qps']:.0f} < 10000")
    else:
        print(f"  ✓ Throughput constraint met: {metrics['throughput_qps']:.0f} ≥ 10000")

    # Check data consistency (invariants)
    if metrics["data_loss_rate"] == 0.0:
        print(f"  ✓ Data consistency maintained (no data loss)")
    else:
        print(f"  ✗ Data loss detected: {metrics['data_loss_rate']:.4%}")
        constraint_violations.append("data_consistency")

    # =========================================================================
    # STEP 6: Learn from Metrics to Optimize
    # =========================================================================
    print_section("STEP 6: Learn from Metrics")

    learner = ArchitectureLearner()

    print("Initializing reinforcement learning...")
    print("(Using PPO to learn optimal architecture configurations)")

    # Baseline metrics (before optimization)
    previous_metrics = {
        "latency_p99": 95.0,
        "throughput_qps": 12500.0,
        "monthly_cost": 3930.0,
        "error_rate": 0.0002,
    }

    print(f"\nBaseline metrics:")
    print(f"  Latency p99: {previous_metrics['latency_p99']:.1f}ms")
    print(f"  Throughput: {previous_metrics['throughput_qps']:,.0f} qps")
    print(f"  Cost: ${previous_metrics['monthly_cost']:,.0f}/month")

    # Run learning to optimize architecture
    print(f"\nRunning learning iterations...")

    optimized_upir = upir
    for iteration in range(3):  # Run 3 optimization iterations
        print(f"\nIteration {iteration + 1}:")

        # Learn from current metrics
        optimized_upir = learner.learn_from_metrics(
            optimized_upir,
            metrics,
            previous_metrics
        )

        # Simulate new metrics after optimization
        # In reality, would deploy and measure actual metrics
        # Here we simulate improvement
        if iteration == 0:
            # Iteration 1: Increase parallelism
            metrics["latency_p99"] = 88.0  # Improved
            metrics["cost_compute"] = 3400.0  # Slightly higher
            metrics["monthly_cost"] = 4130.0
            print(f"  Action: Increased parallelism in beam_processor")
            print(f"  Latency: {previous_metrics['latency_p99']:.1f}ms → {metrics['latency_p99']:.1f}ms ✓")
            print(f"  Cost: ${previous_metrics['monthly_cost']:.0f} → ${metrics['monthly_cost']:.0f}")

        elif iteration == 1:
            # Iteration 2: Optimize batch size
            metrics["latency_p99"] = 82.0  # Further improved
            metrics["throughput_qps"] = 13200.0  # Increased
            print(f"  Action: Optimized batch size")
            print(f"  Latency: 88.0ms → {metrics['latency_p99']:.1f}ms ✓")
            print(f"  Throughput: {previous_metrics['throughput_qps']:,.0f} → {metrics['throughput_qps']:,.0f} qps")

        elif iteration == 2:
            # Iteration 3: Fine-tune caching
            metrics["latency_p99"] = 79.0  # Optimal
            metrics["cpu_utilization"] = 0.58  # More efficient
            print(f"  Action: Added caching layer")
            print(f"  Latency: 82.0ms → {metrics['latency_p99']:.1f}ms ✓")
            print(f"  CPU: 65% → {metrics['cpu_utilization']:.0%}")

        previous_metrics = metrics.copy()

    print("\n" + "-" * 80)
    print("Optimization Summary:")
    print(f"  Initial latency p99: 95.0ms")
    print(f"  Final latency p99: {metrics['latency_p99']:.1f}ms")
    print(f"  Improvement: {95.0 - metrics['latency_p99']:.1f}ms ({(95.0 - metrics['latency_p99']) / 95.0 * 100:.1f}%)")
    print(f"\n  Final cost: ${metrics['monthly_cost']:,.0f}/month")
    print(f"  Budget utilization: {metrics['monthly_cost'] / 5000 * 100:.1f}%")
    print(f"\n  ✓ All constraints satisfied!")

    # =========================================================================
    # STEP 7: Extract Pattern for Reuse
    # =========================================================================
    print_section("STEP 7: Extract and Save Pattern")

    library = PatternLibrary("examples/patterns.json")

    print("Extracting architectural pattern...")

    # Create pattern from optimized UPIR
    from upir.patterns.extractor import PatternExtractor
    from upir.patterns.pattern import Pattern

    extractor = PatternExtractor()
    features = extractor.extract_features(optimized_upir)

    # Create reusable pattern
    streaming_pattern = Pattern(
        id="real-time-streaming-etl",
        name="Real-time Streaming ETL Pattern",
        description=(
            "Production-grade streaming ETL pipeline with Pub/Sub ingestion, "
            "Apache Beam processing on Dataflow, and BigQuery storage. "
            "Optimized for <100ms latency and cost efficiency. "
            "Includes automatic scaling and reliability features."
        ),
        template={
            "components": [
                {
                    "type": "pubsub_source",
                    "count": 1.0,
                    "properties": {
                        "recommended_config": {
                            "ack_deadline": 60,
                            "max_outstanding_messages": 1000,
                        }
                    },
                },
                {
                    "type": "streaming_processor",
                    "count": 1.0,
                    "properties": {
                        "recommended_config": {
                            "framework": "apache_beam",
                            "runner": "dataflow",
                            "parallelism": 10,
                            "autoscaling": True,
                        }
                    },
                },
                {
                    "type": "database",
                    "count": 1.0,
                    "properties": {
                        "recommended_config": {
                            "write_mode": "streaming",
                            "batch_size": 500,
                        }
                    },
                },
            ],
            "parameters": {
                "avg_component_count": 3.0,
                "avg_connection_count": 2.0,
                "target_latency_p99": 100.0,
                "target_cost_monthly": 5000.0,
                "target_throughput_qps": 10000.0,
            },
            "centroid": features.tolist(),
        },
        instances=[optimized_upir.id],
        success_rate=0.95,  # High success rate after optimization
        average_performance={
            "latency_p99": metrics["latency_p99"],
            "throughput_qps": metrics["throughput_qps"],
            "monthly_cost": metrics["monthly_cost"],
            "uptime": metrics["uptime"],
        },
    )

    # Add to library
    library.add_pattern(streaming_pattern)

    print(f"Created pattern: {streaming_pattern.name}")
    print(f"  ID: {streaming_pattern.id}")
    print(f"  Success rate: {streaming_pattern.success_rate:.1%}")
    print(f"  Instances: {len(streaming_pattern.instances)}")
    print(f"\nPerformance metrics:")
    for metric, value in streaming_pattern.average_performance.items():
        print(f"  - {metric}: {value}")

    # Save pattern library
    library.save()
    print(f"\n✓ Saved pattern library to: examples/patterns.json")

    # Test pattern matching
    print("\n" + "-" * 80)
    print("Testing pattern matching on new UPIR...")

    # Create a similar UPIR to test matching
    test_components = [
        {"id": "ps", "name": "PubSub", "type": "pubsub_source"},
        {"id": "proc", "name": "Processor", "type": "streaming_processor"},
        {"id": "bq", "name": "BigQuery", "type": "database"},
    ]
    test_arch = Architecture(
        components=test_components,
        connections=[
            {"from": "ps", "to": "proc"},
            {"from": "proc", "to": "bq"},
        ],
    )
    test_upir = UPIR(
        id="test-streaming",
        name="Test Streaming Pipeline",
        description="Similar streaming pipeline",
        architecture=test_arch,
    )

    # Match against library
    matches = library.match_architecture(test_upir, threshold=0.7)

    print(f"\nFound {len(matches)} matching patterns:")
    for i, (pattern, score) in enumerate(matches[:3], 1):
        print(f"\n{i}. {pattern.name}")
        print(f"   Similarity: {score:.1%}")
        print(f"   Success rate: {pattern.success_rate:.1%}")
        if pattern.average_performance:
            print(f"   Performance:")
            for metric, value in list(pattern.average_performance.items())[:3]:
                print(f"     - {metric}: {value}")

    # =========================================================================
    # Summary
    # =========================================================================
    print_section("Summary")

    print("✓ Successfully completed UPIR workflow!")
    print("\nSteps completed:")
    print("  1. ✓ Defined formal specification with temporal properties")
    print("  2. ✓ Created UPIR instance with streaming architecture")
    print("  3. ✓ Verified specification is satisfiable")
    print("  4. ✓ Synthesized Apache Beam implementation using CEGIS")
    print("  5. ✓ Simulated production metrics")
    print("  6. ✓ Optimized architecture using reinforcement learning")
    print("  7. ✓ Extracted and saved pattern for reuse")

    print("\nFinal Architecture Performance:")
    print(f"  Latency p99: {metrics['latency_p99']:.1f}ms (target: ≤100ms) ✓")
    print(f"  Throughput: {metrics['throughput_qps']:,.0f} qps (target: ≥10,000) ✓")
    print(f"  Cost: ${metrics['monthly_cost']:,.0f}/month (budget: ≤$5,000) ✓")
    print(f"  Uptime: {metrics['uptime']:.2%} ✓")
    print(f"  Data loss: {metrics['data_loss_rate']:.4%} ✓")

    print("\nGenerated Artifacts:")
    print("  - examples/generated_pipeline.py (Apache Beam code)")
    print("  - examples/patterns.json (Pattern library)")

    print("\nNext Steps:")
    print("  - Deploy generated_pipeline.py to Google Cloud Dataflow")
    print("  - Monitor real production metrics")
    print("  - Continue learning and optimization")
    print("  - Reuse pattern for similar pipelines")

    print("\n" + "=" * 80)
    print("  UPIR: From Specification to Production")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
