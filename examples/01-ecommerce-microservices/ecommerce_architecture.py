"""
E-commerce Microservices Architecture Example

This example demonstrates UPIR's capabilities for a real-world e-commerce system
with multiple microservices, databases, and message queues.

Scenario:
- Order service receives customer orders
- Payment service processes payments
- Inventory service manages stock
- Notification service sends confirmations
- All services communicate via message queue

Requirements:
- Orders must be processed within 5 seconds (99th percentile)
- Payment data must always be consistent
- System must handle 1000 orders/second
- Total cost must stay under $5000/month
- All services must be eventually consistent

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
    PatternLibrary,
)


def create_ecommerce_specification() -> FormalSpecification:
    """
    Create formal specification for e-commerce system.

    Returns:
        FormalSpecification with all requirements
    """
    # Define invariants (must always hold)
    invariants = [
        TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="payment_data_consistent",
            parameters={
                "description": "Payment records must never be corrupted",
                "critical": True
            }
        ),
        TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="no_double_charge",
            parameters={
                "description": "Customers must never be charged twice for same order",
                "critical": True
            }
        ),
        TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="inventory_non_negative",
            parameters={
                "description": "Stock levels cannot go below zero",
                "critical": True
            }
        ),
    ]

    # Define liveness properties (eventually hold)
    properties = [
        TemporalProperty(
            operator=TemporalOperator.WITHIN,
            predicate="order_processed",
            time_bound=5000,  # 5 seconds
            parameters={
                "percentile": 99,
                "description": "99% of orders processed within 5s"
            }
        ),
        TemporalProperty(
            operator=TemporalOperator.EVENTUALLY,
            predicate="order_confirmed",
            time_bound=30000,  # 30 seconds
            parameters={
                "description": "All orders eventually receive confirmation"
            }
        ),
        TemporalProperty(
            operator=TemporalOperator.EVENTUALLY,
            predicate="inventory_synced",
            time_bound=10000,  # 10 seconds
            parameters={
                "description": "Inventory updates propagate within 10s"
            }
        ),
    ]

    # Define resource constraints
    constraints = {
        "latency_p99": {"max": 5000.0},  # 5s max for 99th percentile
        "latency_p50": {"max": 2000.0},  # 2s max for median
        "throughput_qps": {"min": 1000.0},  # 1000 orders/second minimum
        "monthly_cost": {"max": 5000.0},  # $5000/month budget
        "availability": {"min": 0.999},  # 99.9% uptime
        "error_rate": {"max": 0.001},  # 0.1% error rate max
    }

    # Environmental assumptions
    assumptions = [
        "network_partitions_rare",
        "databases_eventually_consistent",
        "message_queue_ordered_delivery",
        "payment_gateway_available",
    ]

    return FormalSpecification(
        invariants=invariants,
        properties=properties,
        constraints=constraints,
        assumptions=assumptions
    )


def create_ecommerce_architecture() -> Architecture:
    """
    Create architecture for e-commerce microservices.

    Returns:
        Architecture with all components and connections
    """
    # Define microservices
    components = [
        # API Gateway
        {
            "id": "api_gateway",
            "name": "API Gateway",
            "type": "api_gateway",
            "latency_ms": 10.0,
            "cost_monthly": 200.0,
            "config": {
                "max_connections": 10000,
                "rate_limit": 5000,
                "protocol": "https"
            }
        },

        # Order Service
        {
            "id": "order_service",
            "name": "Order Service",
            "type": "service",
            "latency_ms": 50.0,
            "cost_monthly": 800.0,
            "config": {
                "replicas": 3,
                "cpu": "2 cores",
                "memory": "4GB",
                "auto_scaling": True
            }
        },

        # Payment Service
        {
            "id": "payment_service",
            "name": "Payment Service",
            "type": "service",
            "latency_ms": 100.0,
            "cost_monthly": 1000.0,
            "config": {
                "replicas": 5,
                "cpu": "4 cores",
                "memory": "8GB",
                "encryption": "AES-256",
                "pci_compliant": True
            }
        },

        # Inventory Service
        {
            "id": "inventory_service",
            "name": "Inventory Service",
            "type": "service",
            "latency_ms": 40.0,
            "cost_monthly": 600.0,
            "config": {
                "replicas": 2,
                "cpu": "2 cores",
                "memory": "4GB"
            }
        },

        # Notification Service
        {
            "id": "notification_service",
            "name": "Notification Service",
            "type": "service",
            "latency_ms": 30.0,
            "cost_monthly": 400.0,
            "config": {
                "replicas": 2,
                "cpu": "1 core",
                "memory": "2GB",
                "channels": ["email", "sms", "push"]
            }
        },

        # Message Queue (Pub/Sub)
        {
            "id": "message_queue",
            "name": "Message Queue",
            "type": "message_queue",
            "latency_ms": 5.0,
            "cost_monthly": 500.0,
            "config": {
                "type": "pubsub",
                "topics": ["orders", "payments", "inventory", "notifications"],
                "guaranteed_delivery": True
            }
        },

        # Orders Database
        {
            "id": "orders_db",
            "name": "Orders Database",
            "type": "database",
            "latency_ms": 20.0,
            "cost_monthly": 600.0,
            "config": {
                "type": "postgresql",
                "replicas": 3,
                "storage_gb": 500,
                "backup_enabled": True
            }
        },

        # Payments Database
        {
            "id": "payments_db",
            "name": "Payments Database",
            "type": "database",
            "latency_ms": 15.0,
            "cost_monthly": 800.0,
            "config": {
                "type": "postgresql",
                "replicas": 5,
                "storage_gb": 200,
                "encryption_at_rest": True,
                "compliance": "PCI-DSS"
            }
        },

        # Inventory Database
        {
            "id": "inventory_db",
            "name": "Inventory Database",
            "type": "database",
            "latency_ms": 25.0,
            "cost_monthly": 400.0,
            "config": {
                "type": "postgresql",
                "replicas": 2,
                "storage_gb": 100
            }
        },

        # Redis Cache
        {
            "id": "cache",
            "name": "Redis Cache",
            "type": "cache",
            "latency_ms": 2.0,
            "cost_monthly": 300.0,
            "config": {
                "type": "redis",
                "memory_gb": 16,
                "eviction_policy": "lru"
            }
        },
    ]

    # Define connections between components
    connections = [
        # API Gateway to services
        {"from": "api_gateway", "to": "order_service", "latency_ms": 3.0},
        {"from": "api_gateway", "to": "cache", "latency_ms": 1.0},

        # Order service connections
        {"from": "order_service", "to": "orders_db", "latency_ms": 2.0},
        {"from": "order_service", "to": "message_queue", "latency_ms": 1.0},
        {"from": "order_service", "to": "cache", "latency_ms": 1.0},

        # Payment service connections
        {"from": "payment_service", "to": "payments_db", "latency_ms": 2.0},
        {"from": "payment_service", "to": "message_queue", "latency_ms": 1.0},
        {"from": "message_queue", "to": "payment_service", "latency_ms": 1.0},

        # Inventory service connections
        {"from": "inventory_service", "to": "inventory_db", "latency_ms": 2.0},
        {"from": "inventory_service", "to": "message_queue", "latency_ms": 1.0},
        {"from": "message_queue", "to": "inventory_service", "latency_ms": 1.0},

        # Notification service connections
        {"from": "message_queue", "to": "notification_service", "latency_ms": 1.0},
    ]

    # Deployment configuration
    deployment = {
        "regions": ["us-west-2", "us-east-1"],
        "strategy": "blue-green",
        "load_balancing": "round-robin",
        "health_checks": True,
        "monitoring": "prometheus",
    }

    # Applied patterns
    patterns = [
        "microservices",
        "event-driven",
        "cqrs",
        "api-gateway",
        "database-per-service",
    ]

    return Architecture(
        components=components,
        connections=connections,
        deployment=deployment,
        patterns=patterns
    )


def main():
    """
    Run the e-commerce microservices example.

    Demonstrates:
    1. Creating formal specification
    2. Defining architecture
    3. Creating UPIR instance
    4. Verifying properties
    5. Analyzing results
    """
    print("=" * 80)
    print("UPIR Example: E-commerce Microservices Architecture")
    print("=" * 80)
    print()

    # Step 1: Create specification
    print("Step 1: Creating formal specification...")
    spec = create_ecommerce_specification()
    print(f"  ✓ Created specification with:")
    print(f"    - {len(spec.invariants)} invariants (must always hold)")
    print(f"    - {len(spec.properties)} properties (liveness goals)")
    print(f"    - {len(spec.constraints)} resource constraints")
    print(f"    - {len(spec.assumptions)} environmental assumptions")
    print()

    # Step 2: Create architecture
    print("Step 2: Creating microservices architecture...")
    arch = create_ecommerce_architecture()
    print(f"  ✓ Created architecture with:")
    print(f"    - {len(arch.components)} components")
    print(f"    - {len(arch.connections)} connections")
    print(f"    - {len(arch.patterns)} architectural patterns")
    print(f"    - Total latency: {arch.total_latency_ms:.1f}ms")
    print(f"    - Total monthly cost: ${arch.total_cost:.2f}")
    print()

    # Step 3: Create UPIR instance
    print("Step 3: Creating UPIR instance...")
    upir = UPIR(
        specification=spec,
        architecture=arch,
        name="E-commerce Platform",
        description="High-throughput e-commerce system with microservices"
    )

    # Validate UPIR
    if upir.validate():
        print("  ✓ UPIR validation successful")
    print()

    # Step 4: Verify architecture
    print("Step 4: Verifying architecture against specification...")
    verifier = Verifier(timeout=30000, enable_cache=True)
    results = verifier.verify_specification(upir)

    print(f"  Verification completed: {len(results)} properties checked")
    print()

    # Step 5: Analyze results
    print("Step 5: Verification Results")
    print("-" * 80)

    proved_count = 0
    failed_count = 0
    unknown_count = 0

    for i, result in enumerate(results, 1):
        status_symbol = {
            "PROVED": "✓",
            "DISPROVED": "✗",
            "UNKNOWN": "?",
            "TIMEOUT": "⏱",
            "ERROR": "⚠"
        }.get(result.status.value, "?")

        print(f"{i}. {status_symbol} {result.property.predicate}")
        print(f"   Status: {result.status.value}")
        print(f"   Operator: {result.property.operator.value}")

        if result.property.time_bound:
            print(f"   Time bound: {result.property.time_bound}ms")

        if result.status.value == "PROVED":
            proved_count += 1
        elif result.status.value == "DISPROVED":
            failed_count += 1
        else:
            unknown_count += 1

        print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Properties Proved: {proved_count}/{len(results)}")
    print(f"Properties Failed: {failed_count}/{len(results)}")
    print(f"Unknown/Timeout:   {unknown_count}/{len(results)}")
    print()

    # Architecture metrics
    print("Architecture Metrics:")
    print(f"  Total Latency:     {arch.total_latency_ms:.1f}ms")
    print(f"  Total Cost:        ${arch.total_cost:.2f}/month")
    print(f"  Components:        {len(arch.components)}")
    print(f"  Connections:       {len(arch.connections)}")
    print()

    # Check constraints
    print("Constraint Compliance:")
    latency_ok = arch.total_latency_ms <= spec.constraints["latency_p99"]["max"]
    cost_ok = arch.total_cost <= spec.constraints["monthly_cost"]["max"]

    print(f"  Latency constraint: {'✓ PASS' if latency_ok else '✗ FAIL'}")
    print(f"    Actual: {arch.total_latency_ms:.1f}ms, "
          f"Max: {spec.constraints['latency_p99']['max']}ms")

    print(f"  Cost constraint: {'✓ PASS' if cost_ok else '✗ FAIL'}")
    print(f"    Actual: ${arch.total_cost:.2f}/month, "
          f"Max: ${spec.constraints['monthly_cost']['max']}/month")
    print()

    # Recommendations
    if not latency_ok or not cost_ok:
        print("Recommendations:")
        if not latency_ok:
            print("  • Consider adding more caching layers")
            print("  • Optimize database queries")
            print("  • Use async processing for non-critical paths")
        if not cost_ok:
            print("  • Reduce number of replicas")
            print("  • Use smaller instance types")
            print("  • Optimize database storage")
        print()

    print("=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
