# UPIR Architecture Specification

## Purpose

This document specifies the architecture for the clean room implementation of UPIR based on the TD Commons disclosure and standard software engineering practices.

## Design Principles

1. **Modularity**: Clear separation of concerns (verification, synthesis, learning, patterns)
2. **Extensibility**: Easy to add new system types, patterns, and optimization strategies
3. **Type Safety**: Comprehensive type hints for all public APIs
4. **Testability**: Each component testable in isolation
5. **Documentation**: Self-documenting code with comprehensive docstrings

## Project Structure

```
upir/
├── __init__.py                 # Package exports
├── core/                       # Core data model
│   ├── __init__.py
│   ├── temporal.py            # Temporal logic (TemporalProperty, TemporalOperator)
│   ├── specification.py       # FormalSpecification
│   ├── evidence.py            # Evidence, ReasoningNode
│   ├── architecture.py        # Architecture
│   └── upir.py                # Main UPIR class
├── verification/               # Formal verification
│   ├── __init__.py
│   ├── solver.py              # ProofCertificate, VerificationResult
│   ├── verifier.py            # Verifier, ProofCache
│   └── encoder.py             # SMT encoding utilities
├── synthesis/                  # Code synthesis
│   ├── __init__.py
│   ├── sketch.py              # Hole, ProgramSketch
│   ├── cegis.py               # Synthesizer, CEGIS loop
│   └── templates/             # Code templates
│       ├── streaming.py       # Streaming pipeline templates
│       ├── batch.py           # Batch processing templates
│       └── api.py             # API service templates
├── learning/                   # RL optimization
│   ├── __init__.py
│   ├── ppo.py                 # PPO algorithm
│   └── learner.py             # ArchitectureLearner
├── patterns/                   # Pattern extraction
│   ├── __init__.py
│   ├── extractor.py           # PatternExtractor
│   ├── library.py             # PatternLibrary
│   └── pattern.py             # Pattern dataclass
└── utils/                      # Shared utilities
    ├── __init__.py
    ├── logging.py             # Logging configuration
    └── serialization.py       # JSON/serialization helpers

tests/                          # Test suite
├── core/
├── verification/
├── synthesis/
├── learning/
└── patterns/

examples/                       # Usage examples
├── streaming_pipeline.py
├── batch_processing.py
└── api_service.py

docs/                           # Documentation
├── user_guide.md
├── api_reference.md
├── architecture.md
└── examples.md
```

## Module Specifications

### Core Package (upir.core)

#### temporal.py

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any

class TemporalOperator(Enum):
    """Temporal logic operators."""
    ALWAYS = "always"        # □P - must always hold
    EVENTUALLY = "eventually"  # ◇P - must hold at some point
    WITHIN = "within"        # ◇≤tP - must hold within time bound
    UNTIL = "until"          # P U Q - P holds until Q
    SINCE = "since"          # P S Q - P has held since Q

@dataclass
class TemporalProperty:
    """A temporal property with formal semantics."""
    operator: TemporalOperator
    predicate: str
    time_bound: Optional[float] = None
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_smt(self) -> str:
        """Convert to SMT-LIB format."""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemporalProperty':
        """Deserialize from dictionary."""
```

#### specification.py

```python
@dataclass
class FormalSpecification:
    """Formal specification of system requirements."""
    invariants: List[TemporalProperty]
    properties: List[TemporalProperty]
    constraints: Dict[str, Dict[str, Any]]
    assumptions: List[str] = field(default_factory=list)

    def validate(self) -> bool:
        """Validate specification consistency."""

    def hash(self) -> str:
        """Generate SHA-256 hash."""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize."""
```

#### evidence.py

```python
@dataclass
class Evidence:
    """Evidence supporting decisions."""
    source: str
    type: str  # "benchmark", "test", "production", "formal_proof"
    data: Dict[str, Any]
    confidence: float  # [0, 1]
    timestamp: datetime

    def update_confidence(self, observation: bool, weight: float = 0.1):
        """Bayesian confidence update."""

@dataclass
class ReasoningNode:
    """Node in reasoning DAG."""
    id: str
    decision: str
    rationale: str
    evidence_ids: List[str]
    parent_ids: List[str]
    alternatives: List[Dict[str, Any]]
    confidence: float

    def compute_confidence(self, evidence_map: Dict[str, Evidence]) -> float:
        """Aggregate evidence using geometric mean."""
```

#### upir.py

```python
@dataclass
class UPIR:
    """Main UPIR class - ties everything together."""
    id: str = field(default_factory=uuid4)
    name: str = ""
    specification: Optional[FormalSpecification] = None
    architecture: Optional[Architecture] = None
    evidence: Dict[str, Evidence] = field(default_factory=dict)
    reasoning: Dict[str, ReasoningNode] = field(default_factory=dict)
    implementation: Optional[Implementation] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_evidence(self, evidence: Evidence) -> str:
        """Add evidence, return ID."""

    def add_reasoning(self, node: ReasoningNode) -> str:
        """Add reasoning node."""

    def compute_overall_confidence(self) -> float:
        """Compute overall architecture confidence."""

    def validate(self) -> bool:
        """Validate UPIR consistency."""

    def generate_signature(self) -> str:
        """Cryptographic signature."""

    def to_json(self) -> str:
        """Serialize to JSON."""
```

### Verification Package (upir.verification)

#### verifier.py

```python
class Verifier:
    """Main verification engine using Z3."""

    def __init__(self, timeout: int = 30000, enable_cache: bool = True):
        """Initialize verifier."""

    def verify_specification(self, upir: UPIR) -> List[VerificationResult]:
        """Verify all properties in specification."""

    def verify_property(
        self,
        property: TemporalProperty,
        architecture: Architecture,
        assumptions: List[str] = None
    ) -> VerificationResult:
        """Verify single property."""

    def verify_incremental(
        self,
        upir: UPIR,
        changed_properties: Set[str] = None
    ) -> List[VerificationResult]:
        """Incremental verification with caching."""

class ProofCache:
    """Cache for verification proofs."""

    def get(self, property, architecture) -> Optional[VerificationResult]:
        """Retrieve cached proof."""

    def put(self, property, architecture, result: VerificationResult):
        """Store proof."""

    def invalidate(self, architecture: Architecture):
        """Invalidate cached proofs."""
```

### Synthesis Package (upir.synthesis)

#### cegis.py

```python
class Synthesizer:
    """CEGIS-based synthesis engine."""

    def __init__(self, max_iterations: int = 100, timeout: int = 60000):
        """Initialize synthesizer."""

    def synthesize(
        self,
        upir: UPIR,
        examples: List[SynthesisExample] = None
    ) -> CEGISResult:
        """Main synthesis entry point."""

    def generate_sketch(self, spec: FormalSpecification) -> ProgramSketch:
        """Generate program sketch from spec."""

    def synthesize_holes(
        self,
        sketch: ProgramSketch,
        spec: FormalSpecification,
        examples: List[SynthesisExample],
        counterexamples: List[Dict]
    ) -> bool:
        """Use SMT to fill holes."""

    def verify_synthesis(
        self,
        implementation: Implementation,
        spec: FormalSpecification
    ) -> Dict[str, Any]:
        """Verify synthesized code."""
```

### Learning Package (upir.learning)

#### ppo.py

```python
@dataclass
class PPOConfig:
    """PPO hyperparameters."""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01

class PPO:
    """Proximal Policy Optimization."""

    def __init__(self, state_dim: int, action_dim: int, config: PPOConfig):
        """Initialize PPO."""

    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Select action using policy."""

    def update(
        self,
        states, actions, old_log_probs, returns, advantages
    ) -> Dict[str, float]:
        """Update policy and value networks."""

    def compute_gae(
        self,
        rewards, values, dones, lambda_: float = 0.95
    ) -> np.ndarray:
        """Compute Generalized Advantage Estimation."""
```

#### learner.py

```python
class ArchitectureLearner:
    """Learn to optimize architectures."""

    def __init__(self, state_dim: int, action_dim: int):
        """Initialize learner."""

    def encode_state(self, upir: UPIR) -> np.ndarray:
        """Encode architecture as state vector."""

    def decode_action(self, action: int, upir: UPIR) -> UPIR:
        """Apply action to architecture."""

    def compute_reward(
        self,
        metrics: Dict[str, float],
        spec: FormalSpecification
    ) -> float:
        """Compute reward from metrics."""

    def learn_from_metrics(
        self,
        upir: UPIR,
        metrics: Dict[str, float]
    ) -> UPIR:
        """Main learning entry point."""
```

### Patterns Package (upir.patterns)

#### extractor.py

```python
class PatternExtractor:
    """Extract patterns via clustering."""

    def __init__(self, n_clusters: int = 10):
        """Initialize extractor."""

    def extract_features(self, upir: UPIR) -> np.ndarray:
        """Extract feature vector."""

    def cluster_architectures(
        self,
        upirs: List[UPIR]
    ) -> Dict[int, List[UPIR]]:
        """Cluster similar architectures."""

    def extract_pattern(self, cluster: List[UPIR]) -> Pattern:
        """Extract pattern from cluster."""

    def discover_patterns(self, upirs: List[UPIR]) -> List[Pattern]:
        """Main pattern discovery entry point."""
```

## Data Flow

### 1. Specification → Verification

```
User creates FormalSpecification
    ↓
UPIR instance created with spec
    ↓
Verifier.verify_specification(upir)
    ↓
For each property:
    Check cache → Encode → Z3 solve → Generate certificate
    ↓
Return list of VerificationResults
```

### 2. Specification → Synthesis

```
UPIR with verified specification
    ↓
Synthesizer.synthesize(upir)
    ↓
Generate sketch based on system type
    ↓
CEGIS loop:
    SMT solve holes → Instantiate → Verify → Refine
    ↓
Return Implementation with SynthesisProof
```

### 3. Metrics → Optimization

```
UPIR with implementation
    ↓
Collect production metrics
    ↓
ArchitectureLearner.learn_from_metrics(upir, metrics)
    ↓
Encode state → PPO select action → Decode action
    ↓
Verify properties still hold
    ↓
Update policy, return optimized UPIR
```

### 4. UPIRs → Patterns

```
Collection of UPIR instances
    ↓
PatternExtractor.discover_patterns(upirs)
    ↓
Extract features → Cluster → Abstract patterns
    ↓
PatternLibrary.add_pattern(pattern)
    ↓
Future UPIRs can match against library
```

## Error Handling Strategy

### Verification Errors
- **Z3 timeout**: Return VerificationStatus.TIMEOUT
- **Invalid SMT encoding**: Return VerificationStatus.ERROR with details
- **Z3 not available**: Graceful degradation, return UNKNOWN

### Synthesis Errors
- **Max iterations**: Return SynthesisStatus.PARTIAL with best attempt
- **Timeout**: Return SynthesisStatus.TIMEOUT
- **Invalid spec**: Return SynthesisStatus.INVALID_SPEC
- **Verification failure**: Continue CEGIS loop with counterexample

### Learning Errors
- **Invalid state**: Raise ValueError with helpful message
- **Property violation**: Reject action, return previous state
- **Metrics unavailable**: Use default/heuristic reward

## Performance Targets

Based on TD Commons benchmarks:

| Component | Target | Measurement |
|-----------|--------|-------------|
| Verification (cached) | O(1) lookup | <1ms for cache hit |
| Verification (uncached) | <100ms | For typical property |
| Incremental verification | 100x+ speedup | vs full reverification |
| Synthesis | <10ms average | Per iteration |
| CEGIS convergence | <100 iterations | For typical system |
| RL convergence | <50 episodes | For parameter tuning |
| Pattern matching | <100ms | Against library of 1000 |

## Testing Strategy

### Unit Tests
- Test each class in isolation
- Mock external dependencies (Z3, etc.)
- Test edge cases and error conditions
- Aim for >90% coverage

### Integration Tests
- Test component interactions
- End-to-end flows (spec → verify → synthesize → optimize)
- Test with real Z3 solver

### Performance Tests
- Benchmark cache hit rates
- Measure verification time vs component count
- Validate incremental verification speedup

### Property-Based Tests
- Use hypothesis for property testing
- Test invariants (e.g., verification is deterministic)

## Extensibility Points

### Adding New System Types
1. Add template in synthesis/templates/
2. Update Synthesizer._infer_system_type()
3. Implement _generate_X_sketch()

### Adding New Patterns
1. Define pattern in patterns/library.py
2. Add to built-in patterns list
3. Document pattern applicability

### Adding New Optimizers
1. Implement optimizer in learning/
2. Inherit from base optimizer interface
3. Register with ArchitectureLearner

## Dependencies Management

### Core Dependencies (Required)
- z3-solver: SMT solving
- numpy: Numerical operations
- scikit-learn: Clustering

### Optional Dependencies
- google-cloud-*: GCP integration
- pytest: Testing
- mypy: Type checking

### Dependency Injection
- Verifier accepts solver parameter (default Z3)
- Learner accepts optimizer parameter (default PPO)
- PatternExtractor accepts clusterer parameter (default KMeans)

## Configuration

### Environment Variables
- `UPIR_CACHE_SIZE`: Proof cache size (default 1000)
- `UPIR_LOG_LEVEL`: Logging level (default INFO)
- `UPIR_Z3_TIMEOUT`: Z3 timeout in ms (default 30000)

### Configuration Files
- `upir.toml`: Optional configuration file
- Follows XDG base directory spec

## Logging

### Log Levels
- DEBUG: Detailed SMT formulas, cache operations
- INFO: Verification results, synthesis progress
- WARNING: Cache misses, heuristic fallbacks
- ERROR: Verification failures, invalid specs

### Log Format
```
[2025-11-15 10:30:45] INFO upir.verification: Proved property 'data_consistency' in 45ms (cached: false)
```

## Security Considerations

### Input Validation
- Validate all user inputs (specs, metrics)
- Sanitize predicates before SMT encoding
- Limit resource consumption (cache size, timeout)

### Code Generation Safety
- Validate synthesized code syntax
- Sandbox code execution during verification
- Include warnings in generated code

### Cryptographic Integrity
- Use SHA-256 for all hashing
- Sign proof certificates
- Validate signatures on load

## Documentation Standards

### Docstrings
- Google-style docstrings
- Include examples for public APIs
- Document complexity and performance

### Type Hints
- All public functions fully typed
- Use typing.* for complex types
- Enable strict mypy checking

### Comments
- Explain "why", not "what"
- Document design decisions
- Reference papers for algorithms

## Release Strategy

### Versioning
- Follow Semantic Versioning 2.0
- Major: Breaking API changes
- Minor: New features, backward compatible
- Patch: Bug fixes

### Release Checklist
- [ ] All tests passing
- [ ] Type checking clean
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped
- [ ] Git tag created
- [ ] PyPI package published

---

**This architecture provides a solid foundation for clean room implementation while maintaining flexibility for future enhancements.**
