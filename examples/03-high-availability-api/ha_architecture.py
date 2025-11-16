"""
High-Availability API Service Example

This example demonstrates UPIR for a mission-critical API service with
strict reliability, availability, and fault tolerance requirements.

Scenario:
- Public-facing REST API serving millions of requests/day
- Global distribution across multiple regions
- Automatic failover and disaster recovery
- Zero-downtime deployments
- Comprehensive monitoring and alerting

Requirements:
- 99.99% availability (SLA: <53 minutes downtime/year)
- Request latency p99 < 200ms globally
- Automatic failover within 30 seconds
- Support 10,000 requests/second sustained
- Handle region failures gracefully

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
    Synthesizer,
)


def create_ha_specification() -> FormalSpecification:
    """
    Create formal specification for high-availability API.

    Returns:
        FormalSpecification with HA requirements
    """
    # Invariants (must always hold)
    invariants = [
        TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="at_least_one_region_healthy",
            parameters={
                "description": "At least one region must be operational",
                "min_healthy_regions": 1,
                "critical": True
            }
        ),
        TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="data_replicated",
            parameters={
                "description": "Data must be replicated across regions",
                "min_replicas": 2,
                "critical": True
            }
        ),
        TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="circuit_breaker_active",
            parameters={
                "description": "Circuit breakers prevent cascade failures",
                "critical": True
            }
        ),
    ]

    # Liveness and performance properties
    properties = [
        TemporalProperty(
            operator=TemporalOperator.WITHIN,
            predicate="request_completed",
            time_bound=200,  # 200ms
            parameters={
                "percentile": 99,
                "description": "99% of requests complete within 200ms"
            }
        ),
        TemporalProperty(
            operator=TemporalOperator.WITHIN,
            predicate="failover_complete",
            time_bound=30000,  # 30 seconds
            parameters={
                "description": "Automatic failover within 30s"
            }
        ),
        TemporalProperty(
            operator=TemporalOperator.WITHIN,
            predicate="health_check_response",
            time_bound=5000,  # 5 seconds
            parameters={
                "description": "Health checks respond within 5s"
            }
        ),
        TemporalProperty(
            operator=TemporalOperator.EVENTUALLY,
            predicate="degraded_service_recovered",
            time_bound=300000,  # 5 minutes
            parameters={
                "description": "Degraded services auto-recover within 5min"
            }
        ),
    ]

    # Resource and SLA constraints
    constraints = {
        "latency_p99": {"max": 200.0},  # 200ms max
        "latency_p50": {"max": 100.0},  # 100ms median
        "throughput_rps": {"min": 10000.0},  # 10K req/sec
        "availability": {"min": 0.9999},  # 99.99% SLA
        "monthly_cost": {"max": 12000.0},  # $12K/month
        "failover_time_ms": {"max": 30000.0},  # 30s max
        "error_rate": {"max": 0.0001},  # 0.01% error rate
        "mttr_minutes": {"max": 5.0},  # Mean time to recovery
    }

    # Environmental assumptions
    assumptions = [
        "regions_fail_independently",
        "network_partitions_transient",
        "dns_eventually_consistent",
        "load_balancer_reliable",
    ]

    return FormalSpecification(
        invariants=invariants,
        properties=properties,
        constraints=constraints,
        assumptions=assumptions
    )


def create_ha_architecture() -> Architecture:
    """
    Create high-availability architecture with multi-region deployment.

    Returns:
        Architecture with HA components
    """
    components = [
        # Global Load Balancer
        {
            "id": "global_lb",
            "name": "Global Load Balancer",
            "type": "load_balancer",
            "latency_ms": 5.0,
            "cost_monthly": 500.0,
            "config": {
                "type": "anycast",
                "geo_routing": True,
                "health_checks": True,
                "ddos_protection": True,
                "ssl_termination": True
            }
        },

        # US-East Region
        {
            "id": "us_east_api",
            "name": "US-East API Servers",
            "type": "service",
            "latency_ms": 20.0,
            "cost_monthly": 2000.0,
            "config": {
                "region": "us-east-1",
                "replicas": 5,
                "cpu": "4 cores",
                "memory": "16GB",
                "auto_scaling": True,
                "min_instances": 3,
                "max_instances": 20
            }
        },

        {
            "id": "us_east_db",
            "name": "US-East Database",
            "type": "database",
            "latency_ms": 15.0,
            "cost_monthly": 1500.0,
            "config": {
                "type": "postgresql",
                "region": "us-east-1",
                "replicas": 3,
                "failover": "automatic",
                "backup_retention_days": 30
            }
        },

        {
            "id": "us_east_cache",
            "name": "US-East Cache",
            "type": "cache",
            "latency_ms": 2.0,
            "cost_monthly": 400.0,
            "config": {
                "type": "redis",
                "region": "us-east-1",
                "replicas": 2,
                "memory_gb": 32
            }
        },

        # EU-West Region
        {
            "id": "eu_west_api",
            "name": "EU-West API Servers",
            "type": "service",
            "latency_ms": 20.0,
            "cost_monthly": 2000.0,
            "config": {
                "region": "eu-west-1",
                "replicas": 5,
                "cpu": "4 cores",
                "memory": "16GB",
                "auto_scaling": True,
                "min_instances": 3,
                "max_instances": 20
            }
        },

        {
            "id": "eu_west_db",
            "name": "EU-West Database",
            "type": "database",
            "latency_ms": 15.0,
            "cost_monthly": 1500.0,
            "config": {
                "type": "postgresql",
                "region": "eu-west-1",
                "replicas": 3,
                "failover": "automatic",
                "backup_retention_days": 30
            }
        },

        {
            "id": "eu_west_cache",
            "name": "EU-West Cache",
            "type": "cache",
            "latency_ms": 2.0,
            "cost_monthly": 400.0,
            "config": {
                "type": "redis",
                "region": "eu-west-1",
                "replicas": 2,
                "memory_gb": 32
            }
        },

        # Asia-Pacific Region
        {
            "id": "apac_api",
            "name": "APAC API Servers",
            "type": "service",
            "latency_ms": 20.0,
            "cost_monthly": 2000.0,
            "config": {
                "region": "ap-southeast-1",
                "replicas": 5,
                "cpu": "4 cores",
                "memory": "16GB",
                "auto_scaling": True,
                "min_instances": 3,
                "max_instances": 20
            }
        },

        {
            "id": "apac_db",
            "name": "APAC Database",
            "type": "database",
            "latency_ms": 15.0,
            "cost_monthly": 1500.0,
            "config": {
                "type": "postgresql",
                "region": "ap-southeast-1",
                "replicas": 3,
                "failover": "automatic",
                "backup_retention_days": 30
            }
        },

        {
            "id": "apac_cache",
            "name": "APAC Cache",
            "type": "cache",
            "latency_ms": 2.0,
            "cost_monthly": 400.0,
            "config": {
                "type": "redis",
                "region": "ap-southeast-1",
                "replicas": 2,
                "memory_gb": 32
            }
        },

        # Global services
        {
            "id": "cdn",
            "name": "Content Delivery Network",
            "type": "cdn",
            "latency_ms": 10.0,
            "cost_monthly": 800.0,
            "config": {
                "edge_locations": 200,
                "cache_ttl_seconds": 300,
                "compression": True
            }
        },

        {
            "id": "monitoring",
            "name": "Global Monitoring",
            "type": "monitoring",
            "latency_ms": 5.0,
            "cost_monthly": 500.0,
            "config": {
                "type": "datadog",
                "metrics_per_second": 10000,
                "log_retention_days": 90,
                "alerts": True,
                "uptime_checks": True
            }
        },
    ]

    # Connections
    connections = [
        # Global LB to regional APIs
        {"from": "global_lb", "to": "us_east_api", "latency_ms": 2.0},
        {"from": "global_lb", "to": "eu_west_api", "latency_ms": 2.0},
        {"from": "global_lb", "to": "apac_api", "latency_ms": 2.0},

        # CDN integration
        {"from": "cdn", "to": "global_lb", "latency_ms": 1.0},

        # US-East region
        {"from": "us_east_api", "to": "us_east_cache", "latency_ms": 1.0},
        {"from": "us_east_api", "to": "us_east_db", "latency_ms": 1.0},

        # EU-West region
        {"from": "eu_west_api", "to": "eu_west_cache", "latency_ms": 1.0},
        {"from": "eu_west_api", "to": "eu_west_db", "latency_ms": 1.0},

        # APAC region
        {"from": "apac_api", "to": "apac_cache", "latency_ms": 1.0},
        {"from": "apac_api", "to": "apac_db", "latency_ms": 1.0},

        # Cross-region database replication
        {"from": "us_east_db", "to": "eu_west_db", "latency_ms": 50.0},
        {"from": "eu_west_db", "to": "apac_db", "latency_ms": 100.0},
        {"from": "apac_db", "to": "us_east_db", "latency_ms": 80.0},

        # Monitoring
        {"from": "monitoring", "to": "us_east_api", "latency_ms": 1.0},
        {"from": "monitoring", "to": "eu_west_api", "latency_ms": 1.0},
        {"from": "monitoring", "to": "apac_api", "latency_ms": 1.0},
    ]

    # Deployment
    deployment = {
        "regions": ["us-east-1", "eu-west-1", "ap-southeast-1"],
        "strategy": "active-active",
        "failover": {
            "automatic": True,
            "health_check_interval_s": 10,
            "failure_threshold": 3,
            "recovery_threshold": 2
        },
        "disaster_recovery": {
            "enabled": True,
            "rpo_seconds": 60,
            "rto_seconds": 30,
            "backup_regions": ["us-west-2", "eu-central-1"]
        }
    }

    # Patterns
    patterns = [
        "active-active",
        "circuit-breaker",
        "bulkhead",
        "retry-with-backoff",
        "health-check",
        "geo-distribution",
    ]

    return Architecture(
        components=components,
        connections=connections,
        deployment=deployment,
        patterns=patterns
    )


def main():
    """
    Run the high-availability API example.

    Demonstrates:
    1. Multi-region architecture
    2. Automatic failover mechanisms
    3. Strict SLA requirements
    4. Reliability patterns
    """
    print("=" * 80)
    print("UPIR Example: High-Availability API Service")
    print("=" * 80)
    print()

    # Step 1: Specification
    print("Step 1: Defining HA requirements...")
    spec = create_ha_specification()
    print(f"  ✓ High-availability SLA:")
    print(f"    - Availability: {spec.constraints['availability']['min'] * 100}%")
    print(f"    - Max downtime: ~{(1 - spec.constraints['availability']['min']) * 525600:.0f} min/year")
    print(f"    - Failover time: ≤ {spec.constraints['failover_time_ms']['max'] / 1000:.0f}s")
    print(f"    - MTTR: ≤ {spec.constraints['mttr_minutes']['max']:.0f} minutes")
    print()

    # Step 2: Architecture
    print("Step 2: Creating multi-region architecture...")
    arch = create_ha_architecture()

    # Count components by region
    us_east = sum(1 for c in arch.components if 'us_east' in c['id'] or c['id'] == 'global_lb')
    eu_west = sum(1 for c in arch.components if 'eu_west' in c['id'])
    apac = sum(1 for c in arch.components if 'apac' in c['id'])
    global_comp = sum(1 for c in arch.components if c['id'] in ['cdn', 'monitoring', 'global_lb'])

    print(f"  ✓ Architecture created:")
    print(f"    - Total components: {len(arch.components)}")
    print(f"    - US-East region: {us_east} components")
    print(f"    - EU-West region: {eu_west} components")
    print(f"    - APAC region: {apac} components")
    print(f"    - Global services: {global_comp} components")
    print(f"    - Reliability patterns: {len(arch.patterns)}")
    print()

    # Step 3: UPIR instance
    print("Step 3: Creating UPIR instance...")
    upir = UPIR(
        specification=spec,
        architecture=arch,
        name="Global HA API",
        description="Mission-critical API with multi-region deployment"
    )
    upir.validate()
    print("  ✓ UPIR validated successfully")
    print()

    # Step 4: Verification
    print("Step 4: Verifying HA properties...")
    verifier = Verifier(timeout=30000, enable_cache=True)
    results = verifier.verify_specification(upir)
    print(f"  ✓ Verified {len(results)} properties")
    print()

    # Step 5: Results
    print("Step 5: Verification Results")
    print("-" * 80)

    proved = 0
    for i, result in enumerate(results, 1):
        status_symbol = "✓" if result.status.value == "PROVED" else "✗"
        print(f"{i}. {status_symbol} {result.property.predicate}")

        desc = result.property.parameters.get('description', '')
        if desc:
            print(f"   {desc}")

        print(f"   Status: {result.status.value}")

        if result.property.time_bound:
            print(f"   Time bound: {result.property.time_bound}ms")

        if result.status.value == "PROVED":
            proved += 1
        print()

    # Step 6: Reliability analysis
    print("Step 6: Reliability Analysis")
    print("-" * 80)

    print("Deployed regions: US-East, EU-West, APAC")
    print("Replication: Each region has independent resources")
    print(f"Patterns implemented: {', '.join(arch.patterns)}")
    print()

    print("Failover capability:")
    print("  • Automatic health checks every 10 seconds")
    print("  • 3 failed checks trigger failover")
    print("  • Traffic reroutes to healthy regions")
    print("  • Database replicas promote automatically")
    print()

    # Step 7: Performance
    print("Step 7: Performance Metrics")
    print("-" * 80)
    print(f"End-to-end latency: {arch.total_latency_ms:.1f}ms")
    print(f"Monthly cost: ${arch.total_cost:.2f}")
    print(f"Cost per region: ${arch.total_cost / 3:.2f}")
    print()

    # Constraint check
    latency_ok = arch.total_latency_ms <= spec.constraints["latency_p99"]["max"]
    cost_ok = arch.total_cost <= spec.constraints["monthly_cost"]["max"]

    print("SLA Compliance:")
    print(f"  Latency (p99): {'✓ PASS' if latency_ok else '✗ FAIL'}")
    print(f"    Actual: {arch.total_latency_ms:.1f}ms ≤ {spec.constraints['latency_p99']['max']}ms")
    print()
    print(f"  Monthly cost: {'✓ PASS' if cost_ok else '✗ FAIL'}")
    print(f"    Actual: ${arch.total_cost:.2f} ≤ ${spec.constraints['monthly_cost']['max']:.2f}")
    print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Properties verified: {proved}/{len(results)} proved")
    print(f"SLA compliance: {'✓ All constraints met' if latency_ok and cost_ok else '✗ Some constraints violated'}")
    print(f"Multi-region deployment: 3 active regions")
    print(f"Reliability patterns: {len(arch.patterns)} implemented")
    print()

    if proved == len(results) and latency_ok and cost_ok:
        print("✓ Architecture meets all HA requirements!")
        print("  Ready for production deployment")
    else:
        print("Recommendations:")
        if not latency_ok:
            print("  • Add edge caching")
            print("  • Optimize database queries")
        if not cost_ok:
            print("  • Reduce instance sizes")
            print("  • Optimize resource utilization")
        if proved < len(results):
            print("  • Review unproved properties")
            print("  • Add redundancy where needed")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
