# Formal Specifications

Learn how to write formal specifications using temporal logic and constraints.

---

## Overview

Formal specifications define **what** your system must do, without specifying **how** it does it. UPIR uses:

- **Temporal Logic**: Express properties that hold over time
- **Constraints**: Hard bounds on resources and performance

---

## Temporal Properties

### Linear Temporal Logic (LTL)

UPIR uses LTL operators to express properties:

| Operator | Symbol | Meaning | Example |
|----------|--------|---------|---------|
| ALWAYS | □ | Holds at all times | `□ data_consistent` |
| EVENTUALLY | ◇ | Eventually holds | `◇ task_complete` |
| WITHIN | ◇_{≤t} | Holds within time t | `◇_{≤100ms} respond` |
| UNTIL | U | P until Q | `processing U complete` |

### Creating Temporal Properties

```python
from upir.core.temporal import TemporalOperator, TemporalProperty

# Safety property: Data must always be consistent
safety = TemporalProperty(
    operator=TemporalOperator.ALWAYS,
    predicate="data_consistent"
)

# Liveness property: All tasks eventually complete
liveness = TemporalProperty(
    operator=TemporalOperator.EVENTUALLY,
    predicate="all_tasks_complete",
    time_bound=60000  # 60 seconds
)

# Performance property: Low latency
performance = TemporalProperty(
    operator=TemporalOperator.WITHIN,
    predicate="respond",
    time_bound=100  # 100ms
)
```

---

## Constraints

Hard constraints define acceptable bounds:

```python
constraints = {
    # Latency constraints
    "latency_p99": {"max": 100.0},  # p99 ≤ 100ms
    "latency_p50": {"max": 50.0},   # median ≤ 50ms

    # Cost constraints
    "monthly_cost": {"max": 5000.0},  # ≤ $5000/month

    # Throughput constraints
    "throughput_qps": {"min": 10000.0},  # ≥ 10k queries/sec

    # Availability constraints
    "availability": {"min": 0.999}  # ≥ 99.9% uptime
}
```

---

## Complete Specification

Combine properties and constraints:

```python
from upir.core.specification import FormalSpecification

spec = FormalSpecification(
    properties=[safety, liveness, performance],
    invariants=[data_integrity, no_data_loss],
    constraints=constraints
)
```

---

## Common Patterns

### High-Availability System

```python
spec = FormalSpecification(
    properties=[
        TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="system_available"
        )
    ],
    constraints={
        "availability": {"min": 0.9999},  # 99.99% uptime
        "failover_time_ms": {"max": 1000}  # Failover in 1s
    }
)
```

### Low-Latency API

```python
spec = FormalSpecification(
    properties=[
        TemporalProperty(
            operator=TemporalOperator.WITHIN,
            predicate="respond",
            time_bound=100  # 100ms
        )
    ],
    constraints={
        "latency_p99": {"max": 100.0},
        "latency_p50": {"max": 50.0}
    }
)
```

### Data Processing Pipeline

```python
spec = FormalSpecification(
    properties=[
        TemporalProperty(
            operator=TemporalOperator.EVENTUALLY,
            predicate="all_events_processed",
            time_bound=300000  # 5 minutes
        ),
        TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="no_data_loss"
        )
    ],
    constraints={
        "throughput_events_per_sec": {"min": 10000},
        "monthly_cost": {"max": 10000}
    }
)
```

---

## Best Practices

### 1. Start Simple

Begin with essential properties:

```python
# Good: Simple, focused
spec = FormalSpecification(
    properties=[
        TemporalProperty(TemporalOperator.ALWAYS, "data_consistent")
    ],
    constraints={"latency_p99": {"max": 100.0}}
)

# Avoid: Too complex initially
spec = FormalSpecification(
    properties=[...20 properties...],
    constraints={...15 constraints...}
)
```

### 2. Use Meaningful Predicates

```python
# Good: Clear, descriptive
predicate="data_consistent"
predicate="all_events_processed"
predicate="payment_confirmed"

# Avoid: Vague
predicate="ready"
predicate="done"
predicate="ok"
```

### 3. Set Realistic Time Bounds

```python
# Good: Based on requirements
time_bound=100  # 100ms for API response
time_bound=60000  # 60s for batch job

# Avoid: Arbitrary
time_bound=1  # 1ms (unrealistic)
time_bound=999999999  # Effectively unbounded
```

### 4. Combine Safety and Liveness

```python
# Good: Both safety and liveness
spec = FormalSpecification(
    properties=[
        # Safety: Nothing bad happens
        TemporalProperty(TemporalOperator.ALWAYS, "no_data_corruption"),
        # Liveness: Something good eventually happens
        TemporalProperty(TemporalOperator.EVENTUALLY, "task_complete")
    ]
)
```

---

## See Also

- [Verification Guide](verification.md) - Verify specifications
- [API Reference](../api/core/specification.md) - Specification API
- [Temporal Logic API](../api/core/temporal.md) - Temporal properties
