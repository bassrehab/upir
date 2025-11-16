# E-commerce Microservices Architecture Example

This example demonstrates how to use UPIR to formally specify, verify, and analyze a real-world e-commerce microservices architecture.

## Scenario

A high-throughput e-commerce platform with:

- **Order Service**: Receives and processes customer orders
- **Payment Service**: Handles payment processing securely
- **Inventory Service**: Manages product stock levels
- **Notification Service**: Sends order confirmations
- **Message Queue**: Enables async communication between services
- **Databases**: Separate databases per service (database-per-service pattern)
- **Redis Cache**: High-performance caching layer
- **API Gateway**: Entry point with rate limiting and load balancing

## Requirements

The system must satisfy:

### Invariants (Safety Properties)
- Payment data must always be consistent
- No customer can be double-charged
- Inventory cannot go negative

### Liveness Properties
- 99% of orders processed within 5 seconds
- All orders eventually receive confirmation (within 30s)
- Inventory updates propagate within 10 seconds

### Resource Constraints
- Maximum latency (p99): 5000ms
- Minimum throughput: 1000 orders/second
- Maximum monthly cost: $5000
- Minimum availability: 99.9%
- Maximum error rate: 0.1%

## Running the Example

```bash
# From the repository root
python examples/01-ecommerce-microservices/ecommerce_architecture.py
```

## What It Demonstrates

1. **Formal Specification**: Using temporal logic to specify requirements
   - `ALWAYS` for invariants that must never be violated
   - `WITHIN` for bounded-time properties
   - `EVENTUALLY` for liveness properties

2. **Architecture Modeling**: Defining a complex distributed system
   - Multiple microservices with specific resource requirements
   - Message-based async communication
   - Database-per-service pattern
   - Caching strategy

3. **Verification**: Automatically checking if architecture satisfies specification
   - Uses Z3 SMT solver under the hood
   - Provides concrete counterexamples when properties fail
   - Caches verification results for performance

4. **Constraint Checking**: Validating resource budgets
   - Total latency calculation across the call chain
   - Monthly cost aggregation
   - Compliance reporting

## Expected Output

The example will:
1. Create the formal specification
2. Define the architecture
3. Verify all properties
4. Report which properties are proved/failed
5. Check constraint compliance
6. Provide recommendations if constraints are violated

## Extending This Example

Try modifying the architecture to:
- Add more replicas to reduce latency
- Remove components to reduce cost
- Add monitoring services
- Implement different messaging patterns
- Change database configurations

Then re-run verification to see how changes affect property satisfaction.

## Learn More

- See [Quick Start Guide](../../docs/getting-started/quickstart.md) for UPIR basics
- Read [Specifications Guide](../../docs/guide/specifications.md) for temporal logic details
- Check [API Reference](../../docs/api/core/upir.md) for complete API documentation
