# Verification

Learn how to verify architectures using SMT solving.

---

## Overview

UPIR uses Z3, a state-of-the-art SMT (Satisfiability Modulo Theories) solver, to formally verify that architectures satisfy specifications.

---

## Quick Start

```python
from upir import UPIR
from upir.verification.verifier import Verifier
from upir.verification.solver import VerificationStatus

# Create verifier
verifier = Verifier(timeout_ms=10000)

# Verify UPIR instance
results = verifier.verify_specification(upir)

# Check results
if results.status == VerificationStatus.PROVED:
    print(f"✓ Verified! {len(results.proved_properties)} properties proved")
```

---

## See Also

- [Verifier API](../api/verification/verifier.md) - Complete API reference
- [Specifications Guide](specifications.md) - Write specifications
