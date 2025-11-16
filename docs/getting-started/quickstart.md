# Quick Start

Get up and running with UPIR in minutes. This guide walks through creating, verifying, and optimizing a simple distributed system architecture.

---

## Your First UPIR Program

Let's verify a simple distributed system architecture:

```python
from upir import (
    UPIR,
    Architecture,
    FormalSpecification,
    TemporalProperty,
    TemporalOperator,
    Verifier
)
from upir.verification.solver import VerificationStatus

# Step 1: Define temporal properties
properties = [
    # Data must always be consistent
    TemporalProperty(
        operator=TemporalOperator.ALWAYS,
        predicate="data_consistent"
    ),
    # Respond within 100ms
    TemporalProperty(
        operator=TemporalOperator.WITHIN,
        predicate="respond",
        time_bound=100  # milliseconds
    )
]

# Step 2: Create formal specification
spec = FormalSpecification(
    properties=properties,
    constraints={
        "latency_p99": {"max": 100.0},  # 100ms max
        "monthly_cost": {"max": 1000.0}  # $1000/month max
    }
)

# Step 3: Define architecture components
components = [
    {
        "id": "api_gateway",
        "name": "API Gateway",
        "type": "api_gateway",
        "latency_ms": 10.0,
        "cost_monthly": 300.0
    },
    {
        "id": "database",
        "name": "Database",
        "type": "database",
        "latency_ms": 50.0,
        "cost_monthly": 500.0
    }
]

connections = [
    {"from": "api_gateway", "to": "database", "latency_ms": 5.0}
]

arch = Architecture(
    components=components,
    connections=connections
)

# Step 4: Create UPIR instance
upir = UPIR(
    specification=spec,
    architecture=arch
)

# Step 5: Verify the architecture
verifier = Verifier()
results = verifier.verify_specification(upir)

# Step 6: Check results
if results.status == VerificationStatus.PROVED:
    print("✓ Architecture verified successfully!")
    print(f"  - {len(results.proved_properties)} properties proved")
    print(f"  - Verification time: {results.total_time_ms:.2f}ms")
else:
    print("✗ Verification failed")
    for failure in results.counterexamples:
        print(f"  - {failure}")
```

Expected output:
```
✓ Architecture verified successfully!
  - 2 properties proved
  - Verification time: 45.23ms
```

---

## Step-by-Step Explanation

### 1. Define Temporal Properties

UPIR uses Linear Temporal Logic (LTL) to express requirements:

```python
# ALWAYS operator: Property must hold at all times
always_consistent = TemporalProperty(
    operator=TemporalOperator.ALWAYS,
    predicate="data_consistent"
)

# WITHIN operator: Property must occur within time bound
low_latency = TemporalProperty(
    operator=TemporalOperator.WITHIN,
    predicate="respond",
    time_bound=100  # milliseconds
)

# EVENTUALLY operator: Property must eventually occur
eventually_complete = TemporalProperty(
    operator=TemporalOperator.EVENTUALLY,
    predicate="all_tasks_complete",
    time_bound=60000  # 1 minute
)
```

### 2. Create Formal Specification

Combine properties and constraints:

```python
spec = FormalSpecification(
    properties=[always_consistent, low_latency],
    invariants=[eventually_complete],
    constraints={
        "latency_p99": {"max": 100.0},
        "monthly_cost": {"max": 5000.0},
        "throughput_qps": {"min": 10000.0}
    }
)
```

### 3. Define Architecture

Specify components and connections:

```python
components = [
    {
        "id": "component_1",
        "name": "Component Name",
        "type": "component_type",
        "latency_ms": 10.0,
        "cost_monthly": 100.0,
        "config": {/* component-specific config */}
    }
]

connections = [
    {
        "from": "component_1",
        "to": "component_2",
        "latency_ms": 5.0
    }
]

arch = Architecture(components=components, connections=connections)
```

### 4. Create UPIR Instance

Combine specification and architecture:

```python
upir = UPIR(
    specification=spec,
    architecture=arch
)
```

### 5. Verify

Use the SMT-based verifier:

```python
verifier = Verifier()
results = verifier.verify_specification(upir)

# Check verification status
if results.status == VerificationStatus.PROVED:
    print(f"✓ Verified! {len(results.proved_properties)} properties proved")
elif results.status == VerificationStatus.FAILED:
    print(f"✗ Failed: {results.counterexamples}")
elif results.status == VerificationStatus.UNKNOWN:
    print("? Unknown (solver timeout or incomplete)")
```

---

## Next Steps

### Synthesis

Generate code from specifications using CEGIS:

```python
from upir.synthesis.cegis import Synthesizer

synthesizer = Synthesizer(max_iterations=10)
result = synthesizer.synthesize(upir, sketch)

if result.status.value == "SUCCESS":
    print(f"Generated implementation:\n{result.implementation}")
```

### Optimization

Learn from production metrics:

```python
from upir.learning.learner import ArchitectureLearner

learner = ArchitectureLearner(upir)
optimized_upir = learner.learn(metrics, episodes=100)
```

### Patterns

Use the pattern library:

```python
from upir.patterns.library import PatternLibrary

library = PatternLibrary()
matches = library.match_architecture(upir, threshold=0.8)

for pattern, similarity in matches:
    print(f"{pattern.name}: {similarity:.2%} match")
```

---

## Complete Example

For a full end-to-end example including synthesis, optimization, and pattern extraction, see:

- [Streaming Pipeline Example](../examples/streaming.md)
- [Batch Processing Example](../examples/batch.md)

---

## Learn More

- [Core Concepts](concepts.md) - Understand UPIR's architecture
- [Formal Specifications Guide](../guide/specifications.md) - Deep dive into specs
- [Verification Guide](../guide/verification.md) - Learn about SMT solving
- [API Reference](../api/core/upir.md) - Complete API documentation
