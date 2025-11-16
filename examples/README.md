# UPIR Examples

This directory contains complete, runnable examples demonstrating UPIR's capabilities for real-world distributed system architectures.

## Overview

Each example showcases different aspects of UPIR:

| Example | Focus | Patterns | Complexity |
|---------|-------|----------|------------|
| [01-ecommerce-microservices](01-ecommerce-microservices/) | Microservices architecture | Event-driven, CQRS, Database-per-service | ⭐⭐⭐ |
| [02-streaming-pipeline](02-streaming-pipeline/) | Real-time data processing | Lambda architecture, Stream processing | ⭐⭐⭐⭐ |
| [03-high-availability-api](03-high-availability-api/) | Reliability & fault tolerance | Active-active, Circuit breaker, Multi-region | ⭐⭐⭐⭐⭐ |

## Quick Start

```bash
# Clone the repository
git clone https://github.com/bassrehab/upir.git
cd upir

# Install UPIR
pip install -e .

# Run any example
python examples/01-ecommerce-microservices/ecommerce_architecture.py
python examples/02-streaming-pipeline/streaming_architecture.py
python examples/03-high-availability-api/ha_architecture.py
```

## Example 1: E-commerce Microservices

**Scenario**: High-throughput e-commerce platform with order processing, payments, inventory management, and notifications.

**Key Features**:
- 10 microservices with separate databases
- Message queue for async communication
- Redis caching layer
- API gateway with rate limiting

**Demonstrates**:
- Formal specification with temporal logic
- Microservices patterns (database-per-service, event-driven)
- Cost and latency constraint checking
- Verification of safety and liveness properties

**Requirements**:
- Process 1,000 orders/second
- 99% of orders complete within 5 seconds
- No double-charging (safety invariant)
- Monthly cost under $5,000

[→ See full example](01-ecommerce-microservices/)

## Example 2: Streaming Data Pipeline

**Scenario**: Real-time IoT analytics processing 10K+ events/second with Lambda architecture pattern.

**Key Features**:
- Pub/Sub event ingestion
- Apache Beam stream processing
- Hot storage (Bigtable) for real-time queries
- Cold storage (BigQuery) for batch analytics
- Real-time dashboards

**Demonstrates**:
- Streaming-specific specifications (throughput, burst capacity)
- Lambda architecture (speed + batch layers)
- Exactly-once processing semantics
- Pattern extraction and analysis

**Requirements**:
- Process 10,000 events/second sustained
- Handle bursts up to 50,000 events/second
- p99 latency under 100ms
- No data loss (at-least-once delivery)
- 30 days data retention

[→ See full example](02-streaming-pipeline/)

## Example 3: High-Availability API

**Scenario**: Mission-critical global API with 99.99% uptime SLA and multi-region deployment.

**Key Features**:
- Active-active deployment across 3 regions (US, EU, APAC)
- Global load balancing with geo-routing
- Automatic failover within 30 seconds
- Cross-region database replication
- CDN integration

**Demonstrates**:
- Strict SLA requirements (99.99% availability)
- Multi-region architecture patterns
- Automatic failover verification
- Reliability patterns (circuit breaker, bulkhead, retry)
- Disaster recovery planning

**Requirements**:
- 99.99% availability (< 53 min downtime/year)
- p99 latency < 200ms globally
- Support 10,000 requests/second
- Automatic failover in 30s
- MTTR < 5 minutes

[→ See full example](03-high-availability-api/)

## What You'll Learn

### Formal Specifications
```python
# Define requirements with temporal logic
spec = FormalSpecification(
    invariants=[
        # Safety: must ALWAYS hold
        TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="no_data_loss"
        )
    ],
    properties=[
        # Liveness: must EVENTUALLY hold
        TemporalProperty(
            operator=TemporalOperator.WITHIN,
            predicate="request_completed",
            time_bound=100  # milliseconds
        )
    ],
    constraints={
        "latency_p99": {"max": 100.0},
        "monthly_cost": {"max": 5000.0}
    }
)
```

### Architecture Modeling
```python
# Define distributed system components
arch = Architecture(
    components=[
        {
            "id": "api_service",
            "type": "service",
            "latency_ms": 20.0,
            "cost_monthly": 800.0,
            "config": {
                "replicas": 3,
                "auto_scaling": True
            }
        },
        # ... more components
    ],
    connections=[
        {"from": "api_gateway", "to": "api_service", "latency_ms": 2.0}
    ],
    patterns=["microservices", "event-driven"]
)
```

### Automated Verification
```python
# Create UPIR and verify
upir = UPIR(specification=spec, architecture=arch)
verifier = Verifier()
results = verifier.verify_specification(upir)

# Check results
for result in results:
    print(f"{result.property.predicate}: {result.status}")
    # Output: no_data_loss: PROVED
```

## Common Patterns Demonstrated

### Architectural Patterns
- **Microservices**: Decompose into independent services
- **Event-Driven**: Async communication via message queues
- **CQRS**: Separate read and write paths
- **Lambda Architecture**: Hot path + cold path for analytics
- **Active-Active**: Multi-region with all regions serving traffic

### Reliability Patterns
- **Circuit Breaker**: Prevent cascade failures
- **Bulkhead**: Isolate failure domains
- **Retry with Backoff**: Handle transient failures
- **Health Checks**: Continuous monitoring
- **Database per Service**: Data ownership boundaries

### Performance Patterns
- **Caching**: Redis for sub-millisecond reads
- **CDN**: Edge caching for global distribution
- **Partitioning**: Horizontal scaling
- **Async Processing**: Decouple for better throughput

## Running Examples

### Prerequisites
```bash
# Install UPIR
pip install upir

# Or from source
git clone https://github.com/bassrehab/upir.git
cd upir
pip install -e .
```

### Run Individual Examples
```bash
# From repository root
python examples/01-ecommerce-microservices/ecommerce_architecture.py
python examples/02-streaming-pipeline/streaming_architecture.py
python examples/03-high-availability-api/ha_architecture.py
```

### Expected Output
Each example will:
1. Create formal specification
2. Define architecture
3. Verify all properties
4. Report verification results
5. Check constraint compliance
6. Provide recommendations

## Modifying Examples

All examples are designed to be educational and modifiable:

1. **Change Requirements**: Edit constraints in `create_*_specification()`
2. **Modify Architecture**: Add/remove components in `create_*_architecture()`
3. **Add Properties**: Define new temporal properties
4. **Experiment**: See how changes affect verification results

## Next Steps

- Read the [Quick Start Guide](../docs/getting-started/quickstart.md)
- Explore [API Reference](../docs/api/core/upir.md)
- Learn about [Temporal Logic](../docs/guide/specifications.md)
- Study [Verification Techniques](../docs/guide/verification.md)
- Understand [Pattern Library](../docs/api/patterns/library.md)

## Contributing Examples

Have an interesting use case? We welcome contributions!

See [Contributing Guide](../docs/contributing/setup.md) for details.

## Questions?

- [GitHub Issues](https://github.com/bassrehab/upir/issues)
- [Documentation](https://upir.subhadipmitra.com)

---

**Author**: Subhadip Mitra
**License**: Apache 2.0
