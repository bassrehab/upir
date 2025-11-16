# Temporal Logic

Temporal properties using Linear Temporal Logic (LTL).

---

## Overview

UPIR uses Linear Temporal Logic to express properties that must hold over time:

- **TemporalOperator**: Enum of LTL operators
- **TemporalProperty**: Property with operator and predicate

---

## TemporalOperator

::: upir.core.temporal.TemporalOperator
    options:
      show_source: true
      show_root_heading: true

---

## TemporalProperty

::: upir.core.temporal.TemporalProperty
    options:
      show_source: true
      show_root_heading: true
      show_category_heading: true

---

## Usage Examples

### ALWAYS Operator

Property must hold at all times:

```python
from upir.core.temporal import TemporalOperator, TemporalProperty

always_consistent = TemporalProperty(
    operator=TemporalOperator.ALWAYS,
    predicate="data_consistent"
)

# SMT encoding: ∀t. data_consistent(t)
```

### EVENTUALLY Operator

Property must eventually hold:

```python
eventually_complete = TemporalProperty(
    operator=TemporalOperator.EVENTUALLY,
    predicate="all_tasks_complete",
    time_bound=60000  # within 60 seconds
)

# SMT encoding: ∃t. (t ≤ 60000) ∧ all_tasks_complete(t)
```

### WITHIN Operator

Property must occur within time bound:

```python
within_100ms = TemporalProperty(
    operator=TemporalOperator.WITHIN,
    predicate="respond",
    time_bound=100
)

# SMT encoding: ∃t. (t ≤ 100) ∧ respond(t)
```

### UNTIL Operator

Property P holds until Q becomes true:

```python
until_complete = TemporalProperty(
    operator=TemporalOperator.UNTIL,
    predicate="processing",
    time_bound=30000  # max 30 seconds
)

# SMT encoding: ∃t. (t ≤ 30000) ∧ (∀s < t. processing(s)) ∧ complete(t)
```

---

## SMT Encoding

Convert temporal properties to SMT formulas for verification:

```python
property = TemporalProperty(
    operator=TemporalOperator.ALWAYS,
    predicate="data_consistent"
)

# Get SMT encoding
smt_formula = property.to_smt()
print(smt_formula)
# Output: "(forall ((t Int)) (>= t 0) (data_consistent t))"
```

---

## See Also

- [FormalSpecification](specification.md) - Combine properties into specs
- [Verifier](../verification/verifier.md) - Verify temporal properties
