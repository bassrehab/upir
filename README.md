# UPIR - Universal Plan Intermediate Representation

> Formal verification, automatic synthesis, and continuous optimization for distributed systems

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## What is UPIR?

UPIR is a breakthrough system that bridges the gap between architectural intent and implementation reality through mathematical guarantees. It combines three powerful techniques:

- **Formal Verification**: Mathematically prove that architectures meet specifications
- **Automatic Code Synthesis**: Generate correct implementations from formal specifications
- **Continuous Optimization**: Learn from production metrics while maintaining formal invariants

## Key Features

- **SMT-based Verification** using Z3 theorem prover
- **Incremental Verification** with proof caching (274x speedup)
- **CEGIS Synthesis** for distributed systems
- **RL-based Optimization** using Proximal Policy Optimization (PPO)
- **Pattern Extraction** via clustering (89.9% reuse potential)
- **Cryptographic Proof Certificates** for audit trails

## Installation

```bash
pip install upir
```

From source:

```bash
git clone https://github.com/yourusername/upir.git
cd upir
pip install -e .
```

## Quick Start

```python
from upir import UPIR, FormalSpecification, TemporalProperty, TemporalOperator

# Define formal specification
spec = FormalSpecification(
    invariants=[
        TemporalProperty(
            operator=TemporalOperator.EVENTUALLY,
            predicate="all_events_processed",
            time_bound=100.0
        ),
        TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="data_consistency"
        )
    ],
    constraints={
        "latency": {"max": 100},
        "cost": {"max": 5000},
        "availability": {"min": 0.999}
    }
)

# Create UPIR instance
upir = UPIR(
    name="Streaming Pipeline",
    specification=spec
)

# Verify specification
from upir.verification import Verifier
verifier = Verifier()
results = verifier.verify_specification(upir)

for result in results:
    if result.verified:
        print(f"✓ Proved: {result.property.predicate}")
    else:
        print(f"✗ Failed: {result.property.predicate}")
        print(f"  Counterexample: {result.counterexample}")

# Synthesize implementation
from upir.synthesis import Synthesizer
synthesizer = Synthesizer()
synthesis_result = synthesizer.synthesize(upir)

if synthesis_result.status == "success":
    print(f"Generated code:\n{synthesis_result.implementation.code}")

# Optimize from production metrics
from upir.learning import ArchitectureLearner
learner = ArchitectureLearner(state_dim=64, action_dim=40)

metrics = {
    "latency": 95,
    "throughput": 1200,
    "cost": 4800
}

optimized_upir = learner.learn_from_metrics(upir, metrics)
```

## Core Concepts

### Temporal Properties

Express time-bounded requirements using temporal logic:

```python
# Property must ALWAYS hold
TemporalProperty(
    operator=TemporalOperator.ALWAYS,
    predicate="data_consistency"
)

# Property must EVENTUALLY hold within 100 seconds
TemporalProperty(
    operator=TemporalOperator.EVENTUALLY,
    predicate="all_events_processed",
    time_bound=100.0
)
```

### Formal Specification

Combine properties with resource constraints:

```python
spec = FormalSpecification(
    invariants=[...],        # Must always hold
    properties=[...],        # Desired properties
    constraints={            # Resource bounds
        "latency": {"max": 100},
        "throughput": {"min": 1000},
        "cost": {"max": 5000}
    },
    assumptions=[            # Environmental assumptions
        "network_reliable",
        "storage_available"
    ]
)
```

### Evidence and Reasoning

Track decisions with Bayesian confidence:

```python
from upir.core import Evidence, ReasoningNode

# Add evidence
evidence = Evidence(
    source="load_test",
    type="benchmark",
    data={"p99_latency": 95},
    confidence=0.9
)
evidence_id = upir.add_evidence(evidence)

# Document reasoning
node = ReasoningNode(
    decision="Use Pub/Sub for event ingestion",
    rationale="Low latency, exactly-once semantics",
    evidence_ids=[evidence_id],
    alternatives=[
        {"option": "Kafka", "rejected_because": "Higher operational overhead"},
        {"option": "RabbitMQ", "rejected_because": "Lower throughput"}
    ]
)
upir.add_reasoning(node)

# Compute overall confidence
confidence = upir.compute_overall_confidence()
print(f"Architecture confidence: {confidence:.2%}")
```

## Examples

### Streaming Pipeline

See [examples/streaming_pipeline.py](examples/streaming_pipeline.py) for a complete streaming ETL pipeline with:
- Formal specification of latency and consistency requirements
- Automatic synthesis of Apache Beam code
- RL-based optimization of parallelism and window size

### Batch Processing

See [examples/batch_processing.py](examples/batch_processing.py) for batch job optimization.

### Microservices API

See [examples/api_service.py](examples/api_service.py) for API timeout synthesis.

## Performance

Based on benchmarks from the [TD Commons disclosure](https://www.tdcommons.org/dpubs_series/8852/):

- **274x verification speedup** for 64-component systems (vs. monolithic verification)
- **1.97ms average synthesis time** with 43-75% success rates
- **60.1% latency reduction** and 194.5% throughput improvement via RL optimization
- **89.9% pattern reuse potential** identified through clustering
- **Convergence in 45 episodes** for constrained RL optimization

## 🏗️ Architecture

```
upir/
├── core/              # Data model (UPIR, FormalSpecification, Evidence)
├── verification/      # Z3-based formal verification
├── synthesis/         # CEGIS code synthesis
├── learning/          # PPO-based optimization
└── patterns/          # Pattern extraction and library
```

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md).

This is a clean room implementation based solely on public sources. All contributions must:
- Be based on public documentation
- Include proper source attribution
- Pass tests and type checking
- Follow the code style guide

## Documentation

- [User Guide](docs/user_guide.md)
- [API Reference](docs/api_reference.md)
- [Architecture](docs/architecture.md)
- [Examples](examples/)

## References

### Primary Source

This implementation is based on concepts from:

**"Automated Synthesis and Verification of Distributed Systems Using UPIR"**
Author: Subhadip Mitra
Published: TD Commons, November 2025
License: CC BY 4.0
URL: https://www.tdcommons.org/dpubs_series/8852/

### Academic Papers

- **CEGIS**: [Solar-Lezama et al. (2008)](https://people.csail.mit.edu/asolar/papers/Solar-Lezama08.pdf)
- **PPO**: [Schulman et al. (2017)](https://arxiv.org/abs/1707.06347)
- **Temporal Logic**: [Pnueli (1977)](https://www.cs.toronto.edu/~hehner/FMCO/Pnueli.pdf)

### Tools

- **Z3 Theorem Prover**: https://github.com/Z3Prover/z3
- **Apache Beam**: https://beam.apache.org/
- **Google Cloud Platform**: https://cloud.google.com/

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

This is an independent implementation created as a personal open source project by Subhadip Mitra. Not affiliated with or endorsed by Google LLC or any other organization.

## Author

**Subhadip Mitra**

- Personal project
- Based on TD Commons disclosure (CC BY 4.0)
- Implemented using public sources only

## Star History

If you find UPIR useful, please give it a star! ⭐

## Contact

- Issues: [GitHub Issues](https://github.com/yourusername/upir/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/upir/discussions)
