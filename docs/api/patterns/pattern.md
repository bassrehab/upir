# Pattern

Architectural pattern representation.

---

## Overview

The `Pattern` class represents reusable architectural patterns with proven success rates.

---

## Class Documentation

::: upir.patterns.pattern.Pattern
    options:
      show_source: true
      show_root_heading: true
      show_category_heading: true

---

## Usage Example

```python
from upir.patterns.pattern import Pattern
from datetime import datetime

# Create pattern
pattern = Pattern(
    id="streaming-etl-001",
    name="Streaming ETL Pattern",
    description="Real-time data pipeline with Pub/Sub -> Beam -> BigQuery",
    template={
        "components": [...],
        "connections": [...],
        "centroid": [...]  # Feature vector
    },
    instances=[],
    success_rate=0.95,
    created_at=datetime.now(),
    updated_at=datetime.now()
)

# Serialize
pattern_json = pattern.to_json()
```

---

## See Also

- [Pattern Extractor](extractor.md) - Extract patterns from architectures
- [Pattern Library](library.md) - Store and retrieve patterns
