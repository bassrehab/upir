# High-Availability API Service Example

This example demonstrates UPIR for a mission-critical API service with strict reliability and availability requirements.

## Scenario

A global API service that must maintain 99.99% uptime (< 53 minutes downtime per year) with:

- **Multi-region deployment** across US-East, EU-West, and APAC
- **Active-active** configuration for load distribution
- **Automatic failover** within 30 seconds of regional failure
- **Global load balancing** with geo-routing
- **Cross-region replication** for data durability
- **Comprehensive monitoring** and alerting

## SLA Requirements

### Availability
- **99.99% uptime** (four nines)
- Maximum 52.6 minutes downtime per year
- Automatic failover within 30 seconds
- MTTR (Mean Time To Recovery) ≤ 5 minutes

### Performance
- p99 latency: ≤ 200ms globally
- p50 latency: ≤ 100ms
- Throughput: ≥ 10,000 requests/second
- Error rate: ≤ 0.01%

### Operations
- Monthly cost: ≤ $12,000
- Support region failures gracefully
- Zero-downtime deployments

## Architecture Patterns

This example implements multiple reliability patterns:

1. **Active-Active**: All regions serve traffic simultaneously
2. **Circuit Breaker**: Prevent cascade failures
3. **Bulkhead**: Isolate failure domains
4. **Retry with Backoff**: Handle transient failures
5. **Health Checks**: Continuous monitoring
6. **Geo-Distribution**: Serve users from nearest region

## Running the Example

```bash
# From the repository root
PYTHONPATH=. python examples/03-high-availability-api/ha_architecture.py
```

## What It Demonstrates

1. **Multi-Region Architecture**
   - Three independent regions (US, EU, APAC)
   - Each region has complete infrastructure stack
   - Global load balancer distributes traffic
   - Cross-region database replication

2. **Formal HA Requirements**
   ```python
   # At least one region must always be healthy
   TemporalProperty(
       operator=TemporalOperator.ALWAYS,
       predicate="at_least_one_region_healthy"
   )

   # Failover must complete within 30s
   TemporalProperty(
       operator=TemporalOperator.WITHIN,
       predicate="failover_complete",
       time_bound=30000
   )
   ```

3. **SLA Verification**
   - Automatically verifies availability requirements
   - Checks latency budgets
   - Validates failover capabilities
   - Ensures cost compliance

4. **Reliability Patterns**
   - Circuit breakers to prevent cascades
   - Bulkhead isolation between regions
   - Health checks with automatic recovery
   - Retry logic with exponential backoff

## Regional Deployment

Each region contains:
- **API Servers**: 5 replicas with auto-scaling (3-20 instances)
- **Database**: PostgreSQL with 3 replicas and automatic failover
- **Cache**: Redis with 2 replicas for sub-millisecond reads
- **Monitoring**: Continuous health checks and metrics

## Failover Mechanism

1. **Health Checks**: Every 10 seconds
2. **Failure Detection**: 3 consecutive failures trigger alert
3. **Automatic Failover**: Traffic reroutes to healthy regions
4. **Database Promotion**: Replica promoted to primary
5. **Recovery**: 2 consecutive successful checks restore traffic

## Expected Output

The example will:
1. Define HA requirements with 99.99% SLA
2. Create multi-region architecture with 12 components
3. Verify all reliability properties
4. Analyze failover capabilities
5. Check SLA compliance
6. Report on cost and performance
7. Provide optimization recommendations

## Key Metrics

- **Availability**: 99.99% (4 nines)
- **RTO** (Recovery Time Objective): 30 seconds
- **RPO** (Recovery Point Objective): 60 seconds
- **MTTR** (Mean Time To Recovery): 5 minutes
- **Global Latency**: < 200ms p99

## Extending This Example

Try modifications like:
- Adding more regions (e.g., South America, Middle East)
- Implementing active-passive instead of active-active
- Adding disaster recovery automation
- Testing chaos engineering scenarios
- Implementing gradual rollouts (canary deployments)

## Learn More

- [High Availability Patterns](../../docs/guide/patterns.md)
- [Formal Specifications](../../docs/guide/specifications.md)
- [Architecture Verification](../../docs/guide/verification.md)
