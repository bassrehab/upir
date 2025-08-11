# UPIR - Universal Plan Intermediate Representation

## What is UPIR?

UPIR is an **intermediate representation** (IR) that sits between high-level system specifications and low-level code generation. Like LLVM IR for compilers, UPIR provides a standardized format for:

1. **Specifying** distributed system components
2. **Synthesizing** optimal parameters 
3. **Verifying** system properties
4. **Generating** multi-language code

## UPIR File Format

UPIR files (`.upir` extension) contain:

```
system SystemName {
  components { ... }      # Component specifications
  connections { ... }     # Data flow graph
  properties { ... }      # Formal properties to verify
  verification { ... }    # Verification strategy
  targets { ... }        # Code generation targets
  optimization { ... }    # Learning parameters
}

types { ... }            # Type definitions
```

## Example UPIR Files

### Simple Rate Limiter
```upir
system SimpleRateLimiter {
  components {
    limiter: RateLimiter {
      pattern: "rate_limiter"
      requirements {
        requests_per_second: 1000
      }
    }
  }
  
  targets {
    language: ["python"]
  }
}
```

### Complex Pipeline
See `examples/payment_system.upir` for a complete example.

## UPIR Compilation Process

```
.upir file
    ↓
[Parser] → AST
    ↓
[Analyzer] → Validated IR
    ↓
[Synthesizer] → IR with optimal parameters
    ↓
[Verifier] → IR with proofs
    ↓
[Code Generator] → Python/Go/JS code
```

## Key Features

1. **Declarative**: Specify what, not how
2. **Formal**: Properties expressed in temporal logic
3. **Optimizable**: Parameters marked with `${optimize}`
4. **Composable**: Components connected via edges
5. **Verifiable**: Built-in verification strategies

## UPIR vs Other IRs

| Feature | UPIR | LLVM IR | WebAssembly | Protocol Buffers |
|---------|------|---------|-------------|------------------|
| Purpose | System design | Compilation | Execution | Serialization |
| Level | High | Low | Low | Data only |
| Verification | ✓ | ✗ | ✗ | ✗ |
| Synthesis | ✓ | ✗ | ✗ | ✗ |
| Multi-language | ✓ | ✓ | ✓ | ✓ |

## Using UPIR

```python
# Parse UPIR file
from upir.parser import parse_upir

ir = parse_upir("payment_system.upir")

# Synthesize parameters
from upir.synthesis import synthesize_parameters

optimized_ir = synthesize_parameters(ir)

# Verify properties
from upir.verification import verify_system

proofs = verify_system(optimized_ir)

# Generate code
from upir.codegen import generate_code

code = generate_code(optimized_ir, language="python")
```

## UPIR Grammar (Simplified)

```
system       ::= 'system' ID '{' system_body '}'
system_body  ::= components connections properties verification targets
components   ::= 'components' '{' component* '}'
component    ::= ID ':' pattern '{' component_spec '}'
pattern      ::= STRING
component_spec ::= requirements properties synthesis?
requirements ::= 'requirements' '{' (ID ':' value)* '}'
properties   ::= 'properties' '{' property* '}'
property     ::= property_type ':' formula
connections  ::= 'connections' '{' flow edge* '}'
edge         ::= 'edge' ID '->' ID '{' edge_spec '}'
```

## File Extensions

- `.upir` - UPIR intermediate representation
- `.upir.json` - JSON serialization of UPIR
- `.upir.yaml` - YAML serialization of UPIR
- `.upir.compiled` - Compiled binary format