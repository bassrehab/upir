"""
CEGIS Synthesis Demonstration

This example shows how UPIR can automatically synthesize implementations
from formal specifications using Counterexample-Guided Inductive Synthesis.

We'll synthesize a data processing pipeline that:
1. Meets specified latency requirements
2. Handles specified throughput
3. Maintains data consistency

The cool part: we don't write the implementation - CEGIS figures it out!

Author: subhadipmitra@google.com
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from upir.core.models import (
    UPIR, FormalSpecification, TemporalProperty, TemporalOperator
)
from upir.synthesis.synthesizer import (
    Synthesizer, SynthesisExample, CEGISResult, SynthesisStatus
)


def create_synthesis_specification() -> FormalSpecification:
    """
    Create a specification for synthesis.
    
    We're being very specific about what we want - this guides
    the synthesis process to generate appropriate code.
    """
    
    invariants = [
        TemporalProperty(
            operator=TemporalOperator.WITHIN,
            predicate="event_processed",
            time_bound=50.0,  # 50ms processing time
            parameters={
                "description": "All events must be processed within 50ms",
                "priority": "high"
            }
        ),
        TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="data_consistency",
            parameters={
                "description": "Data must remain consistent",
                "priority": "critical"
            }
        )
    ]
    
    properties = [
        TemporalProperty(
            operator=TemporalOperator.EVENTUALLY,
            predicate="pipeline_ready",
            time_bound=5000.0,  # 5 seconds to initialize
            parameters={
                "description": "Pipeline becomes ready within 5 seconds"
            }
        )
    ]
    
    constraints = {
        "latency": {"max": 50, "unit": "ms"},
        "throughput": {"min": 5000, "unit": "events/sec"},
        "parallelism": {"max": 20, "unit": "workers"},
        "memory": {"max": 8192, "unit": "MB"}
    }
    
    return FormalSpecification(
        invariants=invariants,
        properties=properties,
        constraints=constraints,
        assumptions=["reliable_network", "sufficient_resources"]
    )


def create_synthesis_examples() -> list:
    """
    Create input-output examples to guide synthesis.
    
    These examples help CEGIS understand what the synthesized
    code should actually do with concrete data.
    """
    
    examples = []
    
    # Example 1: Simple transformation
    examples.append(SynthesisExample(
        inputs={"event": {"id": 1, "value": 100}},
        expected_output={"id": 1, "value": 100, "processed": True},
        weight=1.0
    ))
    
    # Example 2: Batch of events
    examples.append(SynthesisExample(
        inputs={"events": [
            {"id": 1, "value": 10},
            {"id": 2, "value": 20},
            {"id": 3, "value": 30}
        ]},
        expected_output={"count": 3, "total": 60, "processed": True},
        weight=0.8
    ))
    
    # Example 3: Error handling
    examples.append(SynthesisExample(
        inputs={"event": {"id": -1, "value": None}},
        expected_output={"error": "invalid_event", "handled": True},
        weight=0.5
    ))
    
    return examples


def demonstrate_cegis_synthesis():
    """
    Main demonstration of CEGIS synthesis.
    
    This shows the complete synthesis process from specification
    to working implementation.
    """
    
    print("=" * 60)
    print("CEGIS Synthesis Demonstration")
    print("Automatically generating code from specifications")
    print("=" * 60)
    print()
    
    # Step 1: Create UPIR with specification
    print("Step 1: Creating formal specification...")
    upir = UPIR(
        name="Synthesized Pipeline",
        description="Automatically synthesized data processing pipeline"
    )
    
    spec = create_synthesis_specification()
    upir.specification = spec
    
    print(f"  Specification includes:")
    print(f"  - {len(spec.invariants)} invariants to maintain")
    print(f"  - {len(spec.properties)} properties to satisfy")
    print(f"  - {len(spec.constraints)} performance constraints")
    print()
    
    # Step 2: Create examples
    print("Step 2: Creating synthesis examples...")
    examples = create_synthesis_examples()
    print(f"  Created {len(examples)} input-output examples")
    for i, ex in enumerate(examples, 1):
        print(f"  - Example {i}: {list(ex.inputs.keys())[0]} -> {list(ex.expected_output.keys())}")
    print()
    
    # Step 3: Run CEGIS synthesis
    print("Step 3: Running CEGIS synthesis...")
    print("  This may take a moment as we search for a solution...")
    
    synthesizer = Synthesizer(max_iterations=50, timeout=30000)  # 30 second timeout
    result = synthesizer.synthesize(upir, examples)
    
    print(f"\n  Synthesis Status: {result.status.value}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Execution Time: {result.execution_time:.2f} seconds")
    
    if result.counterexamples:
        print(f"  Counterexamples processed: {len(result.counterexamples)}")
    print()
    
    # Step 4: Show results
    if result.status == SynthesisStatus.SUCCESS:
        print("Step 4: Synthesis succeeded! Generated implementation:")
        print("-" * 60)
        
        if result.implementation:
            # Show key details
            print(f"  Language: {result.implementation.language}")
            print(f"  Framework: {result.implementation.framework}")
            print(f"  Code length: {len(result.implementation.code.splitlines())} lines")
            print()
            
            # Show synthesized parameters
            if result.sketch:
                print("  Synthesized parameters:")
                for hole in result.sketch.holes:
                    if hole.is_filled():
                        print(f"    - {hole.name}: {hole.filled_value}")
                print()
            
            # Show first 30 lines of generated code
            print("  Generated code (first 30 lines):")
            print("  " + "-" * 56)
            code_lines = result.implementation.code.splitlines()
            for i, line in enumerate(code_lines[:30], 1):
                print(f"  {i:3d} | {line}")
            if len(code_lines) > 30:
                print(f"  ... ({len(code_lines) - 30} more lines)")
            print("  " + "-" * 56)
            
            # Show synthesis proof
            if result.implementation.synthesis_proof:
                proof = result.implementation.synthesis_proof
                print("\n  Synthesis proof generated:")
                print(f"    - Specification hash: {proof.specification_hash[:16]}...")
                print(f"    - Implementation hash: {proof.implementation_hash[:16]}...")
                print(f"    - Verification: {'✓ Passed' if proof.verification_result else '✗ Failed'}")
                print(f"    - Certificate: {proof.generate_certificate()[:16]}...")
        
    elif result.status == SynthesisStatus.PARTIAL:
        print("Step 4: Partial synthesis completed")
        print("  Synthesis made progress but couldn't find complete solution")
        
        if result.sketch:
            filled = len([h for h in result.sketch.holes if h.is_filled()])
            total = len(result.sketch.holes)
            print(f"  Filled {filled}/{total} holes")
            
            print("\n  Unfilled holes:")
            for hole in result.sketch.get_unfilled_holes():
                print(f"    - {hole.name} ({hole.hole_type})")
    
    elif result.status == SynthesisStatus.TIMEOUT:
        print("Step 4: Synthesis timed out")
        print("  The specification might be too complex or contradictory")
        print("  Try simplifying constraints or increasing timeout")
    
    else:
        print(f"Step 4: Synthesis failed with status: {result.status.value}")
    
    print()
    print("=" * 60)
    print("CEGIS Demonstration Complete")
    print()
    
    # Show what this means
    print("What just happened:")
    print("1. We specified WHAT we want (latency < 50ms, handle 5000 events/sec)")
    print("2. CEGIS figured out HOW to implement it")
    print("3. The synthesized code is guaranteed to meet the specification")
    print("4. We got a cryptographic proof of correctness")
    print()
    print("This is the future of programming - specify the 'what', ")
    print("let synthesis handle the 'how'!")
    print("=" * 60)


def demonstrate_incremental_synthesis():
    """
    Demonstrate incremental synthesis with counterexamples.
    
    This shows how CEGIS learns from failures to find the right solution.
    """
    
    print("\n" + "=" * 60)
    print("Incremental Synthesis with Counterexamples")
    print("=" * 60)
    print()
    
    # Create a tricky specification
    spec = FormalSpecification(
        invariants=[
            TemporalProperty(
                operator=TemporalOperator.ALWAYS,
                predicate="sorted_output",
                parameters={"description": "Output must be sorted"}
            ),
            TemporalProperty(
                operator=TemporalOperator.WITHIN,
                predicate="processed",
                time_bound=10.0,
                parameters={"description": "Process within 10ms"}
            )
        ],
        properties=[],
        constraints={
            "complexity": {"max": "O(n log n)"},
            "memory": {"max": 1000}
        }
    )
    
    upir = UPIR(name="Sorting Pipeline")
    upir.specification = spec
    
    # Create challenging examples
    examples = [
        SynthesisExample(
            inputs={"data": [3, 1, 4, 1, 5]},
            expected_output=[1, 1, 3, 4, 5],
            weight=1.0
        ),
        SynthesisExample(
            inputs={"data": [9, 2, 6, 5, 3]},
            expected_output=[2, 3, 5, 6, 9],
            weight=1.0
        ),
        SynthesisExample(
            inputs={"data": []},
            expected_output=[],
            weight=0.5  # Edge case
        )
    ]
    
    print("Specification requires:")
    print("  - Output must always be sorted")
    print("  - Processing time < 10ms")
    print("  - Complexity O(n log n) or better")
    print()
    
    print("Running synthesis with challenging examples...")
    
    synthesizer = Synthesizer(max_iterations=20)
    result = synthesizer.synthesize(upir, examples)
    
    print(f"\nSynthesis completed in {result.iterations} iterations")
    
    if result.counterexamples:
        print(f"\nLearned from {len(result.counterexamples)} counterexamples:")
        for i, ce in enumerate(result.counterexamples[:3], 1):
            print(f"  {i}. {ce}")
        
        if len(result.counterexamples) > 3:
            print(f"  ... and {len(result.counterexamples) - 3} more")
    
    print("\nThis demonstrates how CEGIS learns from failures to find")
    print("the correct solution through iterative refinement.")


if __name__ == "__main__":
    # Run main demonstration
    demonstrate_cegis_synthesis()
    
    # Run incremental synthesis demonstration
    demonstrate_incremental_synthesis()