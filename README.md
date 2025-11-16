# UPIR: Universal Plan Intermediate Representation

**Formal verification, automatic synthesis, and continuous optimization of distributed system architectures.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Table of Contents

- [Overview](#overview)
- [Attribution](#attribution)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Usage Examples](#usage-examples)
- [Architecture](#architecture)
- [API Documentation](#api-documentation)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

---

## Overview

UPIR (Universal Plan Intermediate Representation) is an open-source framework for **formally specifying, verifying, synthesizing, and optimizing distributed system architectures**. It bridges the gap between high-level architectural requirements and production-ready implementations.

### What UPIR Does

- **Formal Verification**: Prove that your architecture satisfies correctness properties using SMT solvers
- **Automatic Synthesis**: Generate implementation code from architectural specifications using CEGIS
- **Continuous Optimization**: Learn from production metrics to improve architectures using reinforcement learning
- **Pattern Management**: Extract and reuse proven architectural patterns
- **Incremental Verification**: Cache proofs for faster iteration

### Why UPIR?

Designing distributed systems is hard. Traditional approaches rely on:
- Manual design prone to errors
- Ad-hoc validation that misses edge cases
- Trial-and-error optimization that wastes resources
- Reinventing patterns instead of reusing proven solutions

UPIR automates these processes using **formal methods**, **program synthesis**, and **machine learning**.

---

## Attribution

This is a **clean-room implementation** based solely on public sources:

**Primary Source**: [Automated Synthesis and Verification of Distributed Systems Using UPIR](https://www.tdcommons.org/dpubs_series/8852/) by Subhadip Mitra, published at TD Commons under CC BY 4.0 license.

Additional references listed in [SOURCES.md](SOURCES.md).

**Author**: Subhadip Mitra

**License**: Apache 2.0

**Project Status**: Personal open source project, no affiliations

---

## Key Features

### 1. Formal Specifications with Temporal Logic

Define requirements using Linear Temporal Logic (LTL):

```python
from upir.core.temporal import TemporalOperator, TemporalProperty

# ALWAYS: Data consistency must always hold
always_consistent = TemporalProperty(
    operator=TemporalOperator.ALWAYS,
    predicate="data_consistent"
)

# WITHIN: Respond within 100ms
low_latency = TemporalProperty(
    operator=TemporalOperator.WITHIN,
    predicate="respond",
    time_bound=100  # milliseconds
)
```

### 2. SMT-Based Verification

Verify architectures satisfy specifications:

```python
from upir.verification.verifier import Verifier
from upir.verification.solver import VerificationStatus

verifier = Verifier()
results = verifier.verify_specification(upir)

for result in results:
    if result.status == VerificationStatus.PROVED:
        print(f"✓ Verified: {result.property.predicate}")
    else:
        print(f"✗ Failed: {result.property.predicate}")
        print(f"  Counterexample: {result.counterexample}")
```

### 3. Program Synthesis with CEGIS

Generate code from specifications:

```python
from upir.synthesis.cegis import Synthesizer

synthesizer = Synthesizer(max_iterations=10)
sketch = synthesizer.generate_sketch(specification)
result = synthesizer.synthesize(upir, sketch)

if result.status.value == "SUCCESS":
    print(result.implementation)  # Generated code!
```

### 4. Reinforcement Learning Optimization

Optimize architectures based on production metrics:

```python
from upir.learning.learner import ArchitectureLearner

learner = ArchitectureLearner()
metrics = {"latency_p99": 150, "throughput_qps": 5000}
optimized_upir = learner.learn_from_metrics(upir, metrics)
```

### 5. Pattern Library

Discover and reuse architectural patterns:

```python
from upir.patterns.library import PatternLibrary

library = PatternLibrary()
matches = library.match_architecture(upir, threshold=0.8)

for pattern, score in matches:
    print(f"{pattern.name}: {score:.1%} match")
```

---

## Installation

### Prerequisites

- Python 3.11 or higher
- pip or poetry for package management

### From Source

```bash
# Clone the repository
git clone https://github.com/bassrehab/upir.git
cd upir

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"
```

### Using pip (when published)

```bash
pip install upir
```

### Verify Installation

```bash
python -c "import upir; print('UPIR installed successfully')"
```

---

## Quick Start

Here's a complete example of designing a streaming pipeline with UPIR:

```python
from upir.core.architecture import Architecture
from upir.core.specification import FormalSpecification
from upir.core.temporal import TemporalOperator, TemporalProperty
from upir.core.upir import UPIR
from upir.verification.verifier import Verifier
from upir.verification.solver import VerificationStatus

# 1. Define formal specification
spec = FormalSpecification(
    properties=[
        TemporalProperty(
            operator=TemporalOperator.WITHIN,
            predicate="process_event",
            time_bound=100  # 100ms latency requirement
        )
    ],
    invariants=[
        TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="data_consistent"
        )
    ],
    constraints={
        "latency_p99": {"max": 100.0},
        "monthly_cost": {"max": 5000.0}
    }
)

# 2. Create architecture
components = [
    {"id": "pubsub", "name": "Event Source", "type": "pubsub_source"},
    {"id": "beam", "name": "Processor", "type": "streaming_processor"},
    {"id": "bq", "name": "Database", "type": "database"}
]

connections = [
    {"from": "pubsub", "to": "beam"},
    {"from": "beam", "to": "bq"}
]

arch = Architecture(components=components, connections=connections)

# 3. Create UPIR instance
upir = UPIR(
    id="streaming-pipeline",
    name="Real-time Event Processing",
    description="Streaming pipeline for event analytics",
    architecture=arch,
    specification=spec
)

# 4. Verify specification
verifier = Verifier()
results = verifier.verify_specification(upir)

print(f"Verified {len(results)} properties")
for result in results:
    print(f"  {result.property.predicate}: {result.status.value}")
```

**Output:**
```
Verified 2 properties
  process_event: PROVED
  data_consistent: PROVED
```

See [examples/streaming_example.py](examples/streaming_example.py) for a complete workflow including synthesis, optimization, and pattern extraction.

---

## Core Concepts

### 1. UPIR Instance

A UPIR represents a complete distributed system design:

```
UPIR
├── Architecture (components + connections)
├── Specification (properties + constraints)
├── Evidence (supporting documentation)
├── Reasoning (design decisions)
└── Metadata (versioning, ownership)
```

**Key components:**

- **Architecture**: Physical structure (microservices, databases, queues)
- **Specification**: Formal requirements (latency, consistency, availability)
- **Evidence**: Documentation, test results, performance data
- **Reasoning**: Design rationale and trade-offs

### 2. Temporal Properties

Express requirements using Linear Temporal Logic (LTL):

| Operator | Meaning | Example |
|----------|---------|---------|
| `ALWAYS` | Property holds at all times | `□ data_consistent` |
| `EVENTUALLY` | Property holds at some future time | `◇ all_events_processed` |
| `WITHIN` | Property holds within time bound | `◇_{≤100ms} respond` |
| `UNTIL` | Property holds until another holds | `processing □ complete` |

**Implementation:**

```python
# Safety property: Always maintain consistency
TemporalProperty(
    operator=TemporalOperator.ALWAYS,
    predicate="data_consistent"
)

# Liveness property: Eventually complete processing
TemporalProperty(
    operator=TemporalOperator.EVENTUALLY,
    predicate="processing_complete",
    time_bound=60000  # Within 60 seconds
)
```

### 3. Formal Verification

UPIR uses SMT (Satisfiability Modulo Theories) solvers to prove correctness:

1. **Encode** architecture and specification as logical formulas
2. **Solve** using Z3 SMT solver
3. **Verify** if properties hold for all possible executions
4. **Generate** counterexamples if verification fails

**Verification results:**
- `PROVED`: Property definitely holds ✓
- `DISPROVED`: Counterexample found ✗
- `UNKNOWN`: Solver couldn't determine
- `TIMEOUT`: Exceeded time limit

### 4. Program Synthesis (CEGIS)

Counterexample-Guided Inductive Synthesis:

```
1. Generate program sketch with holes
   └─> def process(data): return data.window(size=???)

2. Fill holes with candidate values
   └─> def process(data): return data.window(size=60)

3. Verify against specification
   └─> Verify latency ≤ 100ms

4. If verification fails, use counterexample to refine
   └─> Counterexample: size=60 causes 150ms latency

5. Repeat until successful or max iterations
   └─> Try size=30 → Verified! ✓
```

### 5. Reinforcement Learning Optimization

Use Proximal Policy Optimization (PPO) to learn optimal architectures:

```
State: Current architecture features
Action: Modify component (parallelism, type, connections)
Reward: Performance improvement + constraint satisfaction
Policy: Learn which modifications work best
```

**Optimization loop:**

```python
for iteration in range(num_iterations):
    # 1. Encode current architecture
    state = learner.encode_state(upir)

    # 2. Select optimization action (via PPO)
    action, _, _ = learner.ppo.select_action(state)

    # 3. Apply modification
    optimized_upir = learner.decode_action(action, upir)

    # 4. Measure new metrics
    new_metrics = simulate_deployment(optimized_upir)

    # 5. Compute reward
    reward = learner.compute_reward(new_metrics, spec)

    # 6. Update policy
    learner._update_policy()
```

### 6. Pattern Library

Discover, store, and reuse architectural patterns:

```python
library = PatternLibrary()

# Built-in patterns:
# - Streaming ETL Pipeline
# - Batch Processing Pipeline
# - Request-Response API
# - Event-Driven Microservices
# - Lambda Architecture
# - CQRS, Event Sourcing, etc.

# Match your architecture
matches = library.match_architecture(upir, threshold=0.8)

# Use best match as starting point
best_pattern, score = matches[0]
template = best_pattern.template
```

---

## Usage Examples

### Example 1: Verify Latency Requirements

```python
from upir.core.specification import FormalSpecification
from upir.core.temporal import TemporalOperator, TemporalProperty

# Define latency requirement
spec = FormalSpecification(
    properties=[
        TemporalProperty(
            operator=TemporalOperator.WITHIN,
            predicate="api_response",
            time_bound=50  # 50ms
        )
    ]
)

# Create architecture with API gateway
components = [
    {"id": "lb", "type": "load_balancer", "latency_ms": 5},
    {"id": "api", "type": "api_server", "latency_ms": 20},
    {"id": "cache", "type": "cache", "latency_ms": 2},
    {"id": "db", "type": "database", "latency_ms": 15}
]

# Total latency: 5 + 20 + 2 + 15 = 42ms < 50ms ✓
```

### Example 2: Synthesize Batch Processing Pipeline

```python
from upir.synthesis.cegis import Synthesizer

# Specification for batch job
spec = FormalSpecification(
    properties=[
        TemporalProperty(
            operator=TemporalOperator.EVENTUALLY,
            predicate="batch_complete",
            time_bound=3600000  # 1 hour
        )
    ],
    constraints={
        "throughput_records_per_sec": {"min": 10000}
    }
)

# Synthesize implementation
synthesizer = Synthesizer()
sketch = synthesizer.generate_sketch(spec)
result = synthesizer.synthesize(upir, sketch)

# Get generated code
if result.status.value == "SUCCESS":
    with open("batch_pipeline.py", "w") as f:
        f.write(result.implementation)
```

### Example 3: Optimize for Cost

```python
from upir.learning.learner import ArchitectureLearner

learner = ArchitectureLearner()

# Current metrics
current_metrics = {
    "latency_p99": 45,
    "throughput_qps": 15000,
    "monthly_cost": 8000  # Over budget!
}

# Optimize
for i in range(10):  # 10 optimization iterations
    optimized_upir = learner.learn_from_metrics(
        upir,
        current_metrics,
        previous_metrics
    )

    # Simulate new metrics
    current_metrics = simulate(optimized_upir)

    if current_metrics["monthly_cost"] <= 5000:
        print(f"✓ Optimized in {i+1} iterations")
        break
```

### Example 4: Pattern Matching

```python
from upir.patterns.library import PatternLibrary

library = PatternLibrary()

# Find patterns for your architecture
matches = library.match_architecture(upir, threshold=0.7)

print("Similar patterns:")
for pattern, score in matches[:5]:
    print(f"\n{pattern.name} ({score:.1%} match)")
    print(f"  Success rate: {pattern.success_rate:.1%}")
    print(f"  Avg latency: {pattern.average_performance.get('latency_p99', 'N/A')}ms")
    print(f"  Instances: {len(pattern.instances)}")

# Search patterns by criteria
streaming_patterns = library.search_patterns({
    "component_types": ["streaming_processor"],
    "min_success_rate": 0.85
})
```

---

## Architecture

UPIR is organized into five main modules:

```
upir/
├── core/           # Core data structures
│   ├── upir.py           # Main UPIR class
│   ├── architecture.py   # Component & connection definitions
│   ├── specification.py  # Formal specifications
│   ├── temporal.py       # Temporal logic operators
│   ├── evidence.py       # Evidence & reasoning
│   └── ...
├── verification/   # Formal verification
│   ├── verifier.py       # Main verifier (incremental)
│   └── solver.py         # SMT solver interface (Z3)
├── synthesis/      # Program synthesis
│   ├── cegis.py          # CEGIS synthesizer
│   └── sketch.py         # Program sketches
├── learning/       # RL optimization
│   ├── learner.py        # Architecture learner
│   └── ppo.py            # PPO algorithm
└── patterns/       # Pattern management
    ├── pattern.py        # Pattern dataclass
    ├── extractor.py      # Pattern extraction
    └── library.py        # Pattern library
```

**Module Dependencies:**

```
core ←─── verification
  ↑         ↑
  │         │
  └─── synthesis
  │         ↑
  │         │
  └─── learning
  │         ↑
  │         │
  └─── patterns
```

---

## API Documentation

### Core Classes

#### `UPIR`

Main class representing a complete system design.

```python
class UPIR:
    """Universal Plan Intermediate Representation."""

    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        architecture: Optional[Architecture] = None,
        specification: Optional[FormalSpecification] = None,
        **kwargs
    ):
        """Create a UPIR instance."""
```

**Key Methods:**
- `to_dict()`: Serialize to dictionary
- `from_dict(data)`: Deserialize from dictionary
- `compute_hash()`: Compute content hash
- `validate()`: Validate consistency

#### `FormalSpecification`

Captures requirements.

```python
class FormalSpecification:
    """Formal specification with temporal properties."""

    properties: List[TemporalProperty]    # Liveness properties
    invariants: List[TemporalProperty]    # Safety properties
    constraints: Dict[str, Dict[str, Any]]  # Resource constraints
    assumptions: List[str]                # Environmental assumptions
```

### Verification API

#### `Verifier`

Main verification interface.

```python
class Verifier:
    """SMT-based verifier with incremental verification."""

    def verify_specification(
        self,
        upir: UPIR
    ) -> List[VerificationResult]:
        """Verify all properties in specification."""

    def verify_property(
        self,
        property: TemporalProperty,
        architecture: Architecture
    ) -> VerificationResult:
        """Verify single property."""
```

### Synthesis API

#### `Synthesizer`

CEGIS-based synthesizer.

```python
class Synthesizer:
    """Counterexample-guided inductive synthesis."""

    def generate_sketch(
        self,
        spec: FormalSpecification
    ) -> ProgramSketch:
        """Generate program sketch with holes."""

    def synthesize(
        self,
        upir: UPIR,
        sketch: ProgramSketch
    ) -> CEGISResult:
        """Synthesize implementation from sketch."""
```

### Learning API

#### `ArchitectureLearner`

RL-based optimizer.

```python
class ArchitectureLearner:
    """Learn optimal architectures using PPO."""

    def learn_from_metrics(
        self,
        upir: UPIR,
        metrics: Dict[str, float],
        previous_metrics: Optional[Dict[str, float]] = None
    ) -> UPIR:
        """Optimize architecture based on metrics."""
```

### Patterns API

#### `PatternLibrary`

Pattern storage and matching.

```python
class PatternLibrary:
    """Library of architectural patterns."""

    def match_architecture(
        self,
        upir: UPIR,
        threshold: float = 0.8
    ) -> List[Tuple[Pattern, float]]:
        """Find patterns matching architecture."""

    def search_patterns(
        self,
        query: Dict[str, Any]
    ) -> List[Pattern]:
        """Search by component types, success rate, etc."""
```

---

## Examples

The `examples/` directory contains complete demonstrations:

### `streaming_example.py`

Complete workflow for real-time event processing pipeline:

1. Define formal specification with temporal properties
2. Create UPIR with Pub/Sub → Beam → BigQuery architecture
3. Verify specification (all properties proved)
4. Synthesize Apache Beam implementation
5. Simulate production metrics
6. Optimize using RL (3 iterations: 95ms → 79ms latency)
7. Extract and save pattern for reuse

**Run:**
```bash
PYTHONPATH=. python examples/streaming_example.py
```

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone repository
git clone https://github.com/bassrehab/upir.git
cd upir

# Create development environment
python -m venv venv
source venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Check coverage
pytest tests/ --cov=upir --cov-report=html

# Format code
black upir/ tests/
isort upir/ tests/
```

---

## License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

### Summary

- **Free to use** for commercial and non-commercial purposes
- **Modify and distribute** with attribution
- ⚠**No warranty** provided

---

## References

### Primary Source

**Mitra, Subhadip.** "Automated Synthesis and Verification of Distributed Systems Using UPIR."
*TD Commons*, Defensive Publications Series, 8852 (2025).
https://www.tdcommons.org/dpubs_series/8852/
Licensed under CC BY 4.0

### Academic Papers

**Verification & Synthesis:**
- Solar-Lezama, Armando, et al. "Program synthesis by sketching." (2008)
- Pnueli, Amir. "The temporal logic of programs." (1977)
- De Moura, Leonardo, and Nikolaj Bjørner. "Z3: An efficient SMT solver." (2008)

**Reinforcement Learning:**
- Schulman, John, et al. "Proximal policy optimization algorithms." (2017)

**Distributed Systems:**
- Lamport, Leslie. "The part-time parliament." (1998) - Paxos
- Chang, Fay, et al. "Bigtable: A distributed storage system." (2008)
- Dean, Jeffrey, and Sanjay Ghemawat. "MapReduce: simplified data processing." (2008)

### Tools & Frameworks

- **Z3 Theorem Prover**: https://github.com/Z3Prover/z3
- **Apache Beam**: https://beam.apache.org/
- **scikit-learn**: https://scikit-learn.org/
- **Python dataclasses**: https://docs.python.org/3/library/dataclasses.html

---

## Contact & Support

- **Author**: Subhadip Mitra
- **Issues**: [GitHub Issues](https://github.com/bassrehab/upir/issues)
- **Discussions**: [GitHub Discussions](https://github.com/bassrehab/upir/discussions)

---

## Acknowledgments

- **TD Commons** for publishing the foundational disclosure
- **Apache Software Foundation** for Beam and other tools
- **Z3 Team** at Microsoft Research for the SMT solver
- **Open source community** for libraries and inspiration

---

<div align="center">

**Built with ❤️ for the distributed systems community**

[⭐ Star on GitHub](https://github.com/bassrehab/upir) | [🐛 Report Bug](https://github.com/bassrehab/upir/issues) | [💡 Request Feature](https://github.com/bassrehab/upir/issues)

</div>
