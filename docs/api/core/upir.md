# UPIR

The main `UPIR` class that combines specifications and architectures.

---

## Overview

The `UPIR` class is the core abstraction that ties together:

- Formal specifications (temporal properties and constraints)
- System architecture (components and connections)
- Metadata and configuration

---

## Class Documentation

::: upir.core.upir.UPIR
    options:
      show_source: true
      members:
        - __init__
        - validate
        - to_json
        - from_json
        - to_dict
        - from_dict
      show_root_heading: true
      show_category_heading: true

---

## Usage Example

```python
from upir import UPIR, FormalSpecification, Architecture
from upir.core.temporal import TemporalProperty, TemporalOperator

# Create specification
spec = FormalSpecification(
    properties=[
        TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="data_consistent"
        )
    ],
    constraints={"latency_p99": {"max": 100.0}}
)

# Create architecture
arch = Architecture(
    components=[
        {"id": "api", "type": "api_gateway", "latency_ms": 10}
    ],
    connections=[]
)

# Create UPIR instance
upir = UPIR(specification=spec, architecture=arch)

# Validate
is_valid = upir.validate()

# Serialize
upir_json = upir.to_json()
```

---

## See Also

- [Architecture](architecture.md) - Architecture representation
- [FormalSpecification](specification.md) - Specification representation
- [Verifier](../verification/verifier.md) - Verify UPIR instances
