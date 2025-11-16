# FormalSpecification

Formal specification with temporal properties and constraints.

---

## Overview

The `FormalSpecification` class defines what a system must do:

- **Properties**: Temporal properties (liveness, safety)
- **Invariants**: Properties that must always hold
- **Constraints**: Hard constraints on resources and performance

---

## Class Documentation

::: upir.core.specification.FormalSpecification
    options:
      show_source: true
      show_root_heading: true
      show_category_heading: true

---

## Usage Example

```python
from upir.core.specification import FormalSpecification
from upir.core.temporal import TemporalProperty, TemporalOperator

# Define temporal properties
properties = [
    TemporalProperty(
        operator=TemporalOperator.EVENTUALLY,
        predicate="task_complete",
        time_bound=60000  # 60 seconds
    ),
    TemporalProperty(
        operator=TemporalOperator.WITHIN,
        predicate="respond",
        time_bound=100  # 100ms
    )
]

# Define invariants
invariants = [
    TemporalProperty(
        operator=TemporalOperator.ALWAYS,
        predicate="data_consistent"
    ),
    TemporalProperty(
        operator=TemporalOperator.ALWAYS,
        predicate="no_data_loss"
    )
]

# Define constraints
constraints = {
    "latency_p99": {"max": 100.0},
    "monthly_cost": {"max": 5000.0},
    "throughput_qps": {"min": 10000.0}
}

# Create specification
spec = FormalSpecification(
    properties=properties,
    invariants=invariants,
    constraints=constraints
)

# Serialize
spec_json = spec.to_json()
```

---

## Constraint Schema

Constraints are dictionaries with min/max bounds:

```python
constraints = {
    "metric_name": {
        "min": 100.0,  # Minimum value (optional)
        "max": 1000.0  # Maximum value (optional)
    }
}
```

---

## See Also

- [TemporalProperty](temporal.md) - Temporal logic properties
- [UPIR](upir.md) - Main UPIR class
- [Verifier](../verification/verifier.md) - Verify specifications
