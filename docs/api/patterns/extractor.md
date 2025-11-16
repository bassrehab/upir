# Pattern Extractor

Extract architectural patterns from UPIR instances.

---

## Overview

The `PatternExtractor` extracts reusable patterns from verified architectures using feature extraction and clustering.

---

## Class Documentation

::: upir.patterns.extractor.PatternExtractor
    options:
      show_source: true
      show_root_heading: true
      show_category_heading: true

---

## Usage Example

```python
from upir import UPIR
from upir.patterns.extractor import PatternExtractor

# Create extractor
extractor = PatternExtractor(feature_dim=32)

# Extract pattern from UPIR
pattern = extractor.extract(upir)

print(f"Pattern: {pattern.name}")
print(f"Success rate: {pattern.success_rate:.2%}")
```

---

## See Also

- [Pattern](pattern.md) - Pattern representation
- [Pattern Library](library.md) - Store patterns
