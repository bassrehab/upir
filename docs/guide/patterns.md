# Pattern Library

Extract, store, and reuse proven architectural patterns.

---

## Overview

UPIR's pattern library helps you:

- Extract patterns from verified architectures
- Store patterns with success rates
- Match new architectures to existing patterns
- Reuse proven solutions

---

## Quick Start

```python
from upir.patterns.library import PatternLibrary

# Create library
library = PatternLibrary()

# Match architecture to patterns
matches = library.match_architecture(upir, threshold=0.8)

for pattern, similarity in matches:
    print(f"{pattern.name}: {similarity:.2%} match")
```

---

## Built-in Patterns

The library includes 10 common distributed system patterns:

1. Streaming ETL
2. Batch Processing
3. Request-Response API
4. Event-Driven Microservices
5. Lambda Architecture
6. Kappa Architecture
7. CQRS
8. Event Sourcing
9. Pub/Sub Fanout
10. MapReduce

---

## See Also

- [Pattern Library API](../api/patterns/library.md) - Complete API reference
- [Pattern Extractor API](../api/patterns/extractor.md) - Extract patterns
