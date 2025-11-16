# Architecture

System architecture representation with components and connections.

---

## Overview

The `Architecture` class represents the structure of a distributed system:

- **Components**: Individual services, databases, queues, etc.
- **Connections**: Network links between components
- **Metrics**: Total latency, cost, and other aggregate metrics

---

## Class Documentation

::: upir.core.architecture.Architecture
    options:
      show_source: true
      show_root_heading: true
      show_category_heading: true

---

## Usage Example

```python
from upir.core.architecture import Architecture

# Define components
components = [
    {
        "id": "api_gateway",
        "name": "API Gateway",
        "type": "api_gateway",
        "latency_ms": 10.0,
        "cost_monthly": 300.0,
        "config": {
            "max_connections": 10000
        }
    },
    {
        "id": "database",
        "name": "PostgreSQL Database",
        "type": "database",
        "latency_ms": 50.0,
        "cost_monthly": 500.0,
        "config": {
            "instance_type": "db.m5.large"
        }
    }
]

# Define connections
connections = [
    {
        "from": "api_gateway",
        "to": "database",
        "latency_ms": 5.0
    }
]

# Create architecture
arch = Architecture(
    components=components,
    connections=connections
)

# Access metrics
print(f"Total latency: {arch.total_latency_ms}ms")
print(f"Total cost: ${arch.total_cost}/month")
print(f"Components: {len(arch.components)}")

# Serialize
arch_json = arch.to_json()
```

---

## Component Schema

Each component must have:

- `id` (str): Unique identifier
- `type` (str): Component type (e.g., "database", "api_gateway")
- `latency_ms` (float, optional): Component latency in milliseconds
- `cost_monthly` (float, optional): Monthly cost in USD
- `name` (str, optional): Human-readable name
- `config` (dict, optional): Component-specific configuration

---

## Connection Schema

Each connection must have:

- `from` (str): Source component ID
- `to` (str): Destination component ID
- `latency_ms` (float, optional): Network latency in milliseconds

---

## See Also

- [UPIR](upir.md) - Main UPIR class
- [Specification](specification.md) - Formal specifications
