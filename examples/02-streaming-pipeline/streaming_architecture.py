"""
Real-time Streaming Data Pipeline Example

This example demonstrates UPIR for a real-time analytics pipeline that
processes high-volume event streams with strict latency requirements.

Scenario:
- IoT devices send telemetry data (10K events/second)
- Stream processor filters, transforms, and aggregates events
- Hot storage for real-time queries
- Cold storage for batch analytics
- Monitoring dashboard for live visualization

Requirements:
- Events must be processed within 100ms (p99)
- No data loss (at-least-once delivery)
- Exactly-once semantics for aggregations
- Store all raw events for 30 days
- Handle bursts up to 50K events/second

Author: Subhadip Mitra
License: Apache 2.0
"""

from upir import (
    UPIR,
    Architecture,
    FormalSpecification,
    TemporalOperator,
    TemporalProperty,
    Verifier,
    PatternExtractor,
)


def create_streaming_specification() -> FormalSpecification:
    """
    Create formal specification for streaming pipeline.

    Returns:
        FormalSpecification with streaming requirements
    """
    # Invariants (must always hold)
    invariants = [
        TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="no_data_loss",
            parameters={
                "description": "All events must be persisted",
                "delivery_guarantee": "at-least-once",
                "critical": True
            }
        ),
        TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="exactly_once_aggregation",
            parameters={
                "description": "Aggregations must not double-count",
                "idempotent": True,
                "critical": True
            }
        ),
        TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="ordered_within_partition",
            parameters={
                "description": "Events maintain order within partition key",
                "critical": False
            }
        ),
    ]

    # Liveness properties
    properties = [
        TemporalProperty(
            operator=TemporalOperator.WITHIN,
            predicate="event_processed",
            time_bound=100,  # 100ms
            parameters={
                "percentile": 99,
                "description": "99% of events processed within 100ms"
            }
        ),
        TemporalProperty(
            operator=TemporalOperator.WITHIN,
            predicate="dashboard_updated",
            time_bound=5000,  # 5s
            parameters={
                "description": "Dashboard reflects recent data within 5s"
            }
        ),
        TemporalProperty(
            operator=TemporalOperator.EVENTUALLY,
            predicate="batch_job_complete",
            time_bound=3600000,  # 1 hour
            parameters={
                "description": "Hourly batch jobs complete within 1h"
            }
        ),
    ]

    # Resource constraints
    constraints = {
        "latency_p99": {"max": 100.0},  # 100ms for stream processing
        "latency_p50": {"max": 50.0},   # 50ms median
        "throughput_eps": {"min": 10000.0},  # 10K events/second minimum
        "burst_capacity_eps": {"min": 50000.0},  # Handle 50K bursts
        "monthly_cost": {"max": 8000.0},  # $8K/month budget
        "availability": {"min": 0.9999},  # 99.99% uptime
        "data_retention_days": {"min": 30.0},  # 30 days minimum
    }

    # Environmental assumptions
    assumptions = [
        "network_reliable",
        "iot_devices_send_valid_json",
        "clock_skew_minimal",
        "storage_eventually_consistent",
    ]

    return FormalSpecification(
        invariants=invariants,
        properties=properties,
        constraints=constraints,
        assumptions=assumptions
    )


def create_streaming_architecture() -> Architecture:
    """
    Create architecture for streaming data pipeline.

    Returns:
        Architecture with streaming components
    """
    components = [
        # IoT Event Ingestion (Pub/Sub)
        {
            "id": "event_ingestion",
            "name": "Event Ingestion Layer",
            "type": "pubsub_source",
            "latency_ms": 5.0,
            "cost_monthly": 800.0,
            "config": {
                "topics": ["telemetry", "alerts", "metrics"],
                "partitions": 100,
                "retention_hours": 24,
                "max_message_size_kb": 1024,
                "compression": "snappy"
            }
        },

        # Stream Processing (Apache Beam / Dataflow)
        {
            "id": "stream_processor",
            "name": "Stream Processor",
            "type": "streaming_processor",
            "latency_ms": 30.0,
            "cost_monthly": 3500.0,
            "config": {
                "engine": "dataflow",
                "workers": 20,
                "worker_machine_type": "n1-standard-4",
                "auto_scaling": True,
                "min_workers": 10,
                "max_workers": 50,
                "windowing": "sliding_1min",
                "watermark_delay_s": 10
            }
        },

        # Real-time Database (BigTable / Cassandra)
        {
            "id": "hot_storage",
            "name": "Hot Storage",
            "type": "nosql_database",
            "latency_ms": 10.0,
            "cost_monthly": 2000.0,
            "config": {
                "type": "bigtable",
                "nodes": 5,
                "storage_type": "ssd",
                "replication": 2,
                "ttl_days": 7,
                "read_ops_per_second": 100000,
                "write_ops_per_second": 50000
            }
        },

        # Cold Storage (BigQuery / Data Warehouse)
        {
            "id": "cold_storage",
            "name": "Cold Storage",
            "type": "data_warehouse",
            "latency_ms": 50.0,
            "cost_monthly": 1200.0,
            "config": {
                "type": "bigquery",
                "storage_tb": 10,
                "partitioning": "daily",
                "clustering": ["device_id", "event_type"],
                "retention_days": 365
            }
        },

        # Analytics Service
        {
            "id": "analytics_service",
            "name": "Analytics Service",
            "type": "service",
            "latency_ms": 20.0,
            "cost_monthly": 600.0,
            "config": {
                "replicas": 3,
                "cpu": "2 cores",
                "memory": "8GB",
                "functions": ["aggregate", "filter", "transform"]
            }
        },

        # Dashboard Service
        {
            "id": "dashboard",
            "name": "Real-time Dashboard",
            "type": "visualization",
            "latency_ms": 15.0,
            "cost_monthly": 400.0,
            "config": {
                "type": "grafana",
                "users": 100,
                "dashboards": 50,
                "refresh_rate_s": 5
            }
        },

        # Monitoring & Alerting
        {
            "id": "monitoring",
            "name": "Monitoring System",
            "type": "monitoring",
            "latency_ms": 10.0,
            "cost_monthly": 300.0,
            "config": {
                "type": "prometheus",
                "scrape_interval_s": 15,
                "retention_days": 30,
                "alert_manager": True
            }
        },
    ]

    # Connections
    connections = [
        # Ingestion to processing
        {"from": "event_ingestion", "to": "stream_processor", "latency_ms": 2.0},

        # Processing to storages
        {"from": "stream_processor", "to": "hot_storage", "latency_ms": 3.0},
        {"from": "stream_processor", "to": "cold_storage", "latency_ms": 5.0},

        # Analytics queries
        {"from": "analytics_service", "to": "hot_storage", "latency_ms": 2.0},
        {"from": "analytics_service", "to": "cold_storage", "latency_ms": 5.0},

        # Dashboard data
        {"from": "dashboard", "to": "analytics_service", "latency_ms": 2.0},
        {"from": "dashboard", "to": "hot_storage", "latency_ms": 2.0},

        # Monitoring
        {"from": "monitoring", "to": "stream_processor", "latency_ms": 1.0},
        {"from": "monitoring", "to": "hot_storage", "latency_ms": 1.0},
    ]

    # Deployment
    deployment = {
        "regions": ["us-central1", "europe-west1"],
        "strategy": "canary",
        "disaster_recovery": {
            "enabled": True,
            "rpo_minutes": 5,
            "rto_minutes": 15
        },
        "monitoring": "stackdriver",
    }

    # Patterns
    patterns = [
        "lambda_architecture",
        "event_sourcing",
        "stream_processing",
        "polyglot_persistence",
    ]

    return Architecture(
        components=components,
        connections=connections,
        deployment=deployment,
        patterns=patterns
    )


def main():
    """
    Run the streaming pipeline example.

    Demonstrates:
    1. Streaming-specific specifications
    2. Lambda architecture pattern
    3. Hot/cold storage separation
    4. Pattern extraction from architecture
    """
    print("=" * 80)
    print("UPIR Example: Real-time Streaming Data Pipeline")
    print("=" * 80)
    print()

    # Step 1: Create specification
    print("Step 1: Creating streaming specification...")
    spec = create_streaming_specification()
    print(f"  ✓ Streaming requirements defined:")
    print(f"    - {len(spec.invariants)} invariants (safety)")
    print(f"    - {len(spec.properties)} temporal properties")
    print(f"    - Throughput: {spec.constraints['throughput_eps']['min']:.0f} events/sec")
    print(f"    - Burst capacity: {spec.constraints['burst_capacity_eps']['min']:.0f} events/sec")
    print(f"    - Latency p99: {spec.constraints['latency_p99']['max']:.0f}ms")
    print()

    # Step 2: Create architecture
    print("Step 2: Creating streaming architecture...")
    arch = create_streaming_architecture()
    print(f"  ✓ Pipeline architecture created:")
    print(f"    - {len(arch.components)} components")
    print(f"    - {len(arch.connections)} data flows")
    print(f"    - Patterns: {', '.join(arch.patterns)}")
    print()

    # Component breakdown
    print("  Pipeline Components:")
    for comp in arch.components:
        comp_type = comp['type'].replace('_', ' ').title()
        print(f"    • {comp['name']} ({comp_type})")
        print(f"      Latency: {comp['latency_ms']}ms, Cost: ${comp['cost_monthly']}/mo")
    print()

    # Step 3: Create UPIR
    print("Step 3: Creating UPIR instance...")
    upir = UPIR(
        specification=spec,
        architecture=arch,
        name="IoT Analytics Pipeline",
        description="Real-time streaming pipeline for IoT telemetry data"
    )
    upir.validate()
    print("  ✓ UPIR created and validated")
    print()

    # Step 4: Verify properties
    print("Step 4: Verifying streaming properties...")
    verifier = Verifier(timeout=30000, enable_cache=True)
    results = verifier.verify_specification(upir)
    print(f"  ✓ Verified {len(results)} properties")
    print()

    # Step 5: Results
    print("Step 5: Verification Results")
    print("-" * 80)

    for i, result in enumerate(results, 1):
        status_symbol = "✓" if result.status.value == "PROVED" else "✗"
        print(f"{i}. {status_symbol} {result.property.predicate}")
        print(f"   {result.property.parameters.get('description', 'N/A')}")
        print(f"   Status: {result.status.value}")
        if result.property.time_bound:
            print(f"   Time bound: {result.property.time_bound}ms")
        print()

    # Step 6: Extract patterns
    print("Step 6: Pattern Analysis")
    print("-" * 80)
    print("Detected architectural patterns:")
    for pattern in arch.patterns:
        print(f"  • {pattern.replace('_', ' ').title()}")
    print()

    # Step 7: Performance analysis
    print("Step 7: Performance Analysis")
    print("-" * 80)
    print(f"End-to-end latency: {arch.total_latency_ms:.1f}ms")
    print(f"  Components:  {sum(c.get('latency_ms', 0) for c in arch.components):.1f}ms")
    print(f"  Network:     {sum(c.get('latency_ms', 0) for c in arch.connections):.1f}ms")
    print()
    print(f"Monthly operating cost: ${arch.total_cost:.2f}")
    print()

    # Constraint compliance
    latency_ok = arch.total_latency_ms <= spec.constraints["latency_p99"]["max"]
    cost_ok = arch.total_cost <= spec.constraints["monthly_cost"]["max"]

    print("Constraint Compliance:")
    print(f"  Latency (p99): {'✓ PASS' if latency_ok else '✗ FAIL'}")
    print(f"    Actual: {arch.total_latency_ms:.1f}ms")
    print(f"    Required: ≤ {spec.constraints['latency_p99']['max']}ms")
    print()
    print(f"  Monthly cost: {'✓ PASS' if cost_ok else '✗ FAIL'}")
    print(f"    Actual: ${arch.total_cost:.2f}")
    print(f"    Budget: ≤ ${spec.constraints['monthly_cost']['max']:.2f}")
    print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)

    proved = sum(1 for r in results if r.status.value == "PROVED")
    total = len(results)

    print(f"Verification: {proved}/{total} properties proved")
    print(f"Performance:  {'Within' if latency_ok else 'Exceeds'} latency budget")
    print(f"Cost:         {'Within' if cost_ok else 'Exceeds'} cost budget")
    print()

    if latency_ok and cost_ok and proved == total:
        print("✓ Architecture meets all requirements!")
    else:
        print("Recommendations:")
        if not latency_ok:
            print("  • Reduce processing latency with faster workers")
            print("  • Optimize window aggregation functions")
            print("  • Add caching layer for hot queries")
        if not cost_ok:
            print("  • Use smaller worker machine types")
            print("  • Reduce storage retention periods")
            print("  • Optimize data partitioning")
        if proved < total:
            print("  • Review unproved properties")
            print("  • Add redundancy for reliability")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
