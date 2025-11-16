# Real-time Streaming Data Pipeline Example

This example demonstrates UPIR for a high-throughput streaming analytics pipeline processing IoT telemetry data.

## Scenario

An IoT analytics platform that:

- **Ingests** 10K+ events/second from IoT devices
- **Processes** events in real-time with Apache Beam/Dataflow
- **Stores** recent data in hot storage (Bigtable) for fast queries
- **Archives** historical data in cold storage (BigQuery) for batch analytics
- **Visualizes** real-time metrics on monitoring dashboards
- **Handles** burst traffic up to 50K events/second

## Architecture Pattern: Lambda Architecture

This example implements the **Lambda Architecture** pattern:

- **Speed Layer**: Real-time stream processing → hot storage
- **Batch Layer**: Historical data → cold storage → batch analytics
- **Serving Layer**: Unified query interface across both layers

## Requirements

### Invariants (Safety)
- No data loss (at-least-once delivery guarantee)
- Exactly-once semantics for aggregations
- Event ordering preserved within partitions

### Performance
- p99 latency: ≤ 100ms for event processing
- Dashboard updates within 5 seconds
- Batch jobs complete within 1 hour

### Scale
- Minimum throughput: 10,000 events/second
- Burst capacity: 50,000 events/second
- Data retention: 30 days minimum

### Operations
- 99.99% availability
- Monthly cost: ≤ $8,000
- 5 minute RPO, 15 minute RTO for disaster recovery

## Running the Example

```bash
# From the repository root
PYTHONPATH=. python examples/02-streaming-pipeline/streaming_architecture.py
```

## What It Demonstrates

1. **Streaming-Specific Specifications**
   - Throughput requirements (events/second)
   - Burst capacity constraints
   - Data retention policies
   - Exactly-once processing semantics

2. **Lambda Architecture Components**
   - Event ingestion (Pub/Sub)
   - Stream processing (Dataflow)
   - Hot path (Bigtable for real-time queries)
   - Cold path (BigQuery for batch analytics)
   - Unified serving layer

3. **Performance Analysis**
   - End-to-end latency breakdown
   - Cost optimization opportunities
   - Scalability validation

4. **Pattern Recognition**
   - Automatically detects architectural patterns
   - Validates pattern application
   - Suggests improvements

## Expected Output

The example will:
1. Define streaming requirements with temporal logic
2. Create pipeline architecture with 7 components
3. Verify all safety and liveness properties
4. Analyze performance characteristics
5. Check constraint compliance
6. Extract and report architectural patterns
7. Provide optimization recommendations

## Key Concepts Demonstrated

### Temporal Properties
```python
# Events must be processed within 100ms (p99)
TemporalProperty(
    operator=TemporalOperator.WITHIN,
    predicate="event_processed",
    time_bound=100
)

# No data loss invariant
TemporalProperty(
    operator=TemporalOperator.ALWAYS,
    predicate="no_data_loss"
)
```

### Resource Constraints
```python
constraints = {
    "throughput_eps": {"min": 10000.0},  # Events per second
    "burst_capacity_eps": {"min": 50000.0},  # Peak load
    "latency_p99": {"max": 100.0},  # Milliseconds
}
```

## Extending This Example

Try modifications like:
- Adding ML inference to the stream processor
- Implementing different windowing strategies
- Adding data quality validation
- Implementing backpressure handling
- Adding anomaly detection alerts

## Learn More

- [Stream Processing Patterns](../../docs/guide/patterns.md)
- [Performance Specifications](../../docs/guide/specifications.md)
- [Verification Guide](../../docs/guide/verification.md)
