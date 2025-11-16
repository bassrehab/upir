# Streaming Pipeline Example

Complete example of designing a streaming data pipeline with UPIR.

---

## Overview

This example demonstrates the full UPIR workflow:

1. Define formal specification
2. Create architecture
3. Verify specification
4. Synthesize implementation
5. Simulate production metrics
6. Optimize with RL
7. Extract and save pattern

---

## Complete Code

See [examples/streaming_example.py](../../examples/streaming_example.py) for the complete working example.

---

## Step-by-Step

### 1. Define Specification

```python
from upir.core.specification import FormalSpecification
from upir.core.temporal import TemporalProperty, TemporalOperator

spec = FormalSpecification(
    properties=[
        TemporalProperty(
            operator=TemporalOperator.EVENTUALLY,
            predicate="all_events_processed",
            time_bound=100000  # 100 seconds
        ),
        TemporalProperty(
            operator=TemporalOperator.WITHIN,
            predicate="process_event",
            time_bound=100  # 100ms
        )
    ],
    invariants=[
        TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="data_consistent"
        )
    ],
    constraints={
        "latency_p99": {"max": 100.0},
        "monthly_cost": {"max": 5000.0},
        "throughput_qps": {"min": 10000.0}
    }
)
```

### 2. Create Architecture

```python
from upir.core.architecture import Architecture

components = [
    {
        "id": "pubsub_source",
        "type": "pubsub_source",
        "latency_ms": 5.0,
        "cost_monthly": 500.0
    },
    {
        "id": "beam_processor",
        "type": "streaming_processor",
        "latency_ms": 50.0,
        "cost_monthly": 3000.0
    },
    {
        "id": "bigquery_sink",
        "type": "database",
        "latency_ms": 30.0,
        "cost_monthly": 1200.0
    }
]

connections = [
    {"from": "pubsub_source", "to": "beam_processor", "latency_ms": 2.0},
    {"from": "beam_processor", "to": "bigquery_sink", "latency_ms": 3.0}
]

arch = Architecture(components=components, connections=connections)
```

### 3. Verify

```python
from upir import UPIR
from upir.verification.verifier import Verifier

upir = UPIR(specification=spec, architecture=arch)
verifier = Verifier()
results = verifier.verify_specification(upir)

print(f"✓ {len(results.proved_properties)} properties verified")
```

---

## See Also

- [Quick Start Guide](../getting-started/quickstart.md)
- [Specifications Guide](../guide/specifications.md)
