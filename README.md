# UPIR - Universal Plan Intermediate Representation

**An experimental system for automated synthesis and verification of distributed systems**

## Overview

UPIR (Universal Plan Intermediate Representation) automatically generates correct, optimized distributed system implementations from high-level specifications. It combines formal verification, program synthesis, and constraint-based optimization to produce production-ready code.

**Key Features:**
- 🚀 **274× faster** synthesis than traditional approaches  
- ⚡ **1.71ms** average code generation time
- ✅ **Formally verified** correctness guarantees
- 🎯 **Auto-optimized** parameters using Z3 SMT solver
- 🔄 **Multi-language** support (Python, Go, JavaScript)

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository>
cd upir

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

Create a UPIR specification file (`system.upir`):

```upir
system PaymentProcessor {
  components {
    rate_limiter: RateLimiter {
      pattern: "rate_limiter"
      requirements {
        requests_per_second: 1000
        burst_size: 100
      }
    }
    queue: QueueWorker {
      pattern: "queue_worker"
      requirements {
        batch_size: "${optimize}"
        workers: "${optimize}"
      }
    }
  }
  properties {
    safety: "G(payment => F(processed))"
  }
}
```

Generate implementation:

```python
from upir import UPIR, CodeGenerator

# Load and verify specification
spec = UPIR.load("system.upir")
if spec.verify():
    print("✓ Specification verified")
    
    # Generate optimized code
    generator = CodeGenerator()
    code = generator.generate(spec, language="python")
    print(code)
```

## Project Structure

```
upir/
├── upir/                    # Core library
│   ├── synthesis/          # Program synthesis engine
│   ├── verification/       # Formal verification
│   ├── optimization/       # Z3-based optimization
│   └── codegen/           # Code generation
├── examples/               # Example specifications
├── tests/                  # Test suite
├── experiments/            # Experimental results
├── paper/                  # Research paper and documentation
└── webapp/                 # Web interface demo
```

## Web Interface

Run the interactive demo:

```bash
cd webapp
source venv/bin/activate
python main.py
# Visit http://localhost:8080
```

## Documentation

- [Examples](examples/) - Sample UPIR specifications
- [Web Interface](webapp/) - Interactive demo and API documentation
- [Experiment Report](experiments/20250811_105911/EXPERIMENT_REPORT.md) - Performance validation

## Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_synthesis.py

# Check coverage
python -m pytest --cov=upir tests/
```

## Performance

| Metric | Value |
|--------|-------|
| Synthesis Speed | 274× faster than baseline |
| Code Generation | 1.71ms average |
| Pattern Reuse | 89.9% across systems |
| Verification Time | O(1) incremental |

## Use Cases

- **GenAI Agent Orchestration** - Deadlock-free multi-agent systems
- **Data Pipelines** - ETL with exactly-once guarantees
- **Microservices** - Verified service mesh configurations
- **Stream Processing** - Optimized event processing systems

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Support

For questions and issues:
- Contact: subhadipmitra@google.com
- Internal: http://go/upir

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## Disclaimer

This is an experimental research project and not an official Google product.

## Links

- Code: http://go/upir:code
- Paper: http://go/upir:paper
- Demo: http://localhost:8080 (when running locally)