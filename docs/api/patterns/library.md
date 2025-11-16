# Pattern Library

Store, search, and match architectural patterns.

---

## Overview

The `PatternLibrary` manages a collection of architectural patterns:

- Store patterns with success rates
- Search by name or description
- Match architectures using cosine similarity
- Update success rates with Bayesian updates
- Persist to JSON

Includes 10 built-in distributed system patterns.

---

## Class Documentation

::: upir.patterns.library.PatternLibrary
    options:
      show_source: true
      show_root_heading: true
      show_category_heading: true

---

## Usage Example

```python
from upir import UPIR
from upir.patterns.library import PatternLibrary

# Create library
library = PatternLibrary()

# Add custom pattern
library.add_pattern(my_pattern)

# Search by name
results = library.search_patterns("streaming")
for pattern in results:
    print(f"- {pattern.name}")

# Match architecture
matches = library.match_architecture(upir, threshold=0.8)
for pattern, similarity in matches:
    print(f"{pattern.name}: {similarity:.2%} match")

# Update success rate
library.update_success_rate(pattern_id, success=True)

# Save library
library.save("my_patterns.json")
```

---

## Built-in Patterns

The library includes 10 common distributed system patterns:

1. **Streaming ETL** - Pub/Sub → Beam → BigQuery
2. **Batch Processing** - Storage → Dataflow → BigQuery
3. **Request-Response API** - Load Balancer → API → Database
4. **Event-Driven Microservices** - Pub/Sub → Services → Pub/Sub
5. **Lambda Architecture** - Batch + Streaming paths
6. **Kappa Architecture** - Pure streaming
7. **CQRS** - Separate read/write paths
8. **Event Sourcing** - Event log as source of truth
9. **Pub/Sub Fanout** - One-to-many message distribution
10. **MapReduce** - Distributed data processing

---

## See Also

- [Pattern](pattern.md) - Pattern representation
- [Pattern Extractor](extractor.md) - Extract patterns
