# Synthesis

Generate implementation code from specifications using CEGIS.

---

## Overview

UPIR uses CEGIS (Counterexample-Guided Inductive Synthesis) to automatically generate implementation code from formal specifications.

---

## Quick Start

```python
from upir.synthesis.cegis import Synthesizer

synthesizer = Synthesizer(max_iterations=10)
sketch = synthesizer.generate_sketch(upir.specification)
result = synthesizer.synthesize(upir, sketch)

if result.status.value == "SUCCESS":
    print(f"Generated code:\n{result.implementation}")
```

---

## See Also

- [CEGIS API](../api/synthesis/cegis.md) - Complete API reference
- [Sketch API](../api/synthesis/sketch.md) - Program sketches
