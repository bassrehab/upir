# CEGIS

Counterexample-Guided Inductive Synthesis.

---

## Overview

The `Synthesizer` class implements CEGIS (Counterexample-Guided Inductive Synthesis) to generate implementation code from specifications:

1. Start with a program sketch (template with holes)
2. Synthesize candidate by filling holes
3. Verify candidate satisfies specification
4. If invalid, use counterexample to refine
5. Iterate until correct or max iterations

---

## Class Documentation

::: upir.synthesis.cegis.Synthesizer
    options:
      show_source: true
      show_root_heading: true
      show_category_heading: true

::: upir.synthesis.cegis.CEGISResult
    options:
      show_source: true
      show_root_heading: true

::: upir.synthesis.cegis.SynthesisStatus
    options:
      show_source: true
      show_root_heading: true

---

## Usage Example

```python
from upir import UPIR
from upir.synthesis.cegis import Synthesizer

# Create synthesizer
synthesizer = Synthesizer(max_iterations=10, timeout_ms=60000)

# Generate sketch from specification
sketch = synthesizer.generate_sketch(upir.specification)

# Synthesize implementation
result = synthesizer.synthesize(upir, sketch)

# Check result
if result.status.value == "SUCCESS":
    print(f"✓ Synthesis successful!")
    print(f"Iterations: {result.iterations}")
    print(f"Time: {result.execution_time:.2f}ms")
    print(f"\nGenerated code:\n{result.implementation}")
elif result.status.value == "FAILED":
    print(f"✗ Synthesis failed: {result.error_message}")
elif result.status.value == "TIMEOUT":
    print("⏱ Synthesis timed out")
```

---

## See Also

- [Sketch](sketch.md) - Program sketches
- [Verifier](../verification/verifier.md) - Verify synthesized code
