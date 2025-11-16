# Installation

This guide will help you install UPIR on your system.

---

## Prerequisites

### Required

- **Python 3.11+** - UPIR requires Python 3.11 or later
- **pip** - Python package installer

### Recommended

- **Virtual environment** - Keep UPIR isolated from other projects
- **Git** - For installing from source

---

## Installation Methods

### From PyPI (Recommended)

Install the latest stable release:

```bash
pip install upir
```

### From Source

For the latest development version:

```bash
git clone https://github.com/bassrehab/upir.git
cd upir
pip install -e .
```

---

## Optional Dependencies

UPIR has several optional dependency groups:

### Google Cloud Platform Integration

For GCP-specific components (BigQuery, Pub/Sub, Cloud Storage):

```bash
pip install upir[gcp]
```

### Development Tools

For contributing to UPIR (testing, linting, formatting):

```bash
pip install upir[dev]
```

### Documentation

For building documentation:

```bash
pip install upir[docs]
```

### All Optional Dependencies

Install everything:

```bash
pip install upir[gcp,dev,docs]
```

---

## Core Dependencies

UPIR automatically installs these core dependencies:

- **z3-solver** (≥4.12.2) - SMT solving for formal verification
- **numpy** (≥1.24.3, <2.0.0) - Numerical computing for RL
- **scikit-learn** (≥1.3.0, <2.0.0) - Clustering for pattern extraction

---

## Verify Installation

After installation, verify UPIR is working:

```python
import upir
print(upir.__version__)  # Should print: 0.1.0
```

Test core functionality:

```python
from upir import UPIR, FormalSpecification, TemporalProperty, TemporalOperator

# Create a simple temporal property
prop = TemporalProperty(
    operator=TemporalOperator.ALWAYS,
    predicate="test_predicate"
)
print(f"✓ UPIR installed successfully!")
```

---

## Troubleshooting

### Python Version Issues

If you get version errors, check your Python version:

```bash
python --version  # Should be 3.11 or higher
```

If you have multiple Python versions, use:

```bash
python3.11 -m pip install upir
```

### Z3 Solver Issues

If Z3 installation fails, try installing it separately:

```bash
pip install z3-solver
```

### Permission Errors

On Linux/macOS, you might need:

```bash
pip install --user upir
```

Or use a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install upir
```

---

## Next Steps

Once installed, proceed to:

- [Quick Start Guide](quickstart.md) - Your first UPIR program
- [Core Concepts](concepts.md) - Understand key ideas
- [Examples](../examples/streaming.md) - Real-world use cases
