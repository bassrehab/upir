# Verifier

SMT-based formal verification of specifications.

---

## Overview

The `Verifier` class uses Z3 SMT solver to prove that architectures satisfy specifications:

- Encodes temporal properties as SMT formulas
- Checks satisfiability using Z3
- Caches proofs for incremental verification
- Returns verification results with counterexamples

---

## Class Documentation

::: upir.verification.verifier.Verifier
    options:
      show_source: true
      show_root_heading: true
      show_category_heading: true

---

## Usage Example

```python
from upir import UPIR, FormalSpecification, Architecture
from upir.verification.verifier import Verifier
from upir.verification.solver import VerificationStatus

# Create UPIR instance
upir = UPIR(specification=spec, architecture=arch)

# Create verifier
verifier = Verifier(timeout_ms=10000)

# Verify specification
results = verifier.verify_specification(upir)

# Check results
if results.status == VerificationStatus.PROVED:
    print(f"✓ Verified! {len(results.proved_properties)} properties proved")
    print(f"Time: {results.total_time_ms:.2f}ms")
elif results.status == VerificationStatus.FAILED:
    print(f"✗ Failed!")
    for counterexample in results.counterexamples:
        print(f"  Counterexample: {counterexample}")
elif results.status == VerificationStatus.UNKNOWN:
    print("? Unknown (timeout or incomplete)")
```

---

## Incremental Verification

The verifier caches proven properties:

```python
# First verification
results1 = verifier.verify_specification(upir)  # ~100ms

# Modify architecture slightly
upir.architecture.components[0]["latency_ms"] += 1

# Second verification (uses cached proofs)
results2 = verifier.verify_specification(upir)  # ~10ms
```

---

## See Also

- [SMT Solver](solver.md) - Low-level SMT solving
- [FormalSpecification](../core/specification.md) - Create specifications
