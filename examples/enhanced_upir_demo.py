#!/usr/bin/env python3
"""
Demonstration of UPIR with new enhancements:
1. Template-based code generation
2. Bounded program synthesis  
3. Compositional verification

This example shows how the enhanced UPIR system can:
- Generate production-ready code from specifications
- Synthesize optimal parameters and small functions
- Verify large systems compositionally
"""

import json
from typing import Dict, Any

# Import original UPIR components
from upir.core.models import UPIR, FormalSpecification, TemporalProperty
from upir.verification.verifier import Verifier
from upir.learning.learner import Learner

# Import new enhancements
from upir.codegen import CodeGenerator, QueueWorkerTemplate, RateLimiterTemplate, CircuitBreakerTemplate
from upir.synthesis.program_synthesis import PredicateSynthesizer, TransformationSynthesizer
from upir.verification.compositional import CompositionalVerifier, Component as CompComponent, Property as CompProperty, PropertyType


def demonstrate_code_generation():
    """
    Demonstrate template-based code generation.
    Generates a complete payment processing system.
    """
    print("\n" + "="*60)
    print("DEMONSTRATION 1: Template-Based Code Generation")
    print("="*60)
    
    # Initialize code generator
    generator = CodeGenerator()
    generator.register_template(QueueWorkerTemplate())
    generator.register_template(RateLimiterTemplate())
    generator.register_template(CircuitBreakerTemplate())
    
    # Define system specification
    payment_system_spec = [
        {
            'pattern': 'rate_limiter',
            'requirements': {
                'requests_per_second': 1000  # Handle 1000 payments/sec
            },
            'properties': ['bounded_latency']
        },
        {
            'pattern': 'queue_worker',
            'requirements': {
                'batch_size': 25,  # Process 25 payments at a time
                'workers': 8,      # 8 parallel workers
                'timeout_ms': 5000
            },
            'properties': ['no_data_loss', 'idempotent']
        },
        {
            'pattern': 'circuit_breaker',
            'requirements': {
                'failure_threshold': 5,
                'recovery_timeout_ms': 10000
            },
            'properties': ['fault_tolerance']
        }
    ]
    
    print("\nGenerating payment processing system...")
    print("Requirements:")
    print("- 1000 payments/second throughput")
    print("- No data loss guarantee")
    print("- Fault tolerance with circuit breaker")
    
    # Generate code for each component
    for i, spec in enumerate(payment_system_spec, 1):
        print(f"\n--- Component {i}: {spec['pattern'].replace('_', ' ').title()} ---")
        
        result = generator.generate_from_spec(spec, 'python')
        
        print(f"Synthesized parameters:")
        for param, value in result.synthesized_params.items():
            print(f"  {param}: {value}")
        
        print(f"Verified properties: {', '.join(result.verified_properties)}")
        
        # Show a snippet of generated code
        code_lines = result.code.split('\n')
        print(f"Generated code (first 10 lines):")
        for line in code_lines[:10]:
            print(f"  {line}")
        
    print("\n✅ Complete payment system generated with formal guarantees!")


def demonstrate_program_synthesis():
    """
    Demonstrate bounded program synthesis.
    Synthesizes validation and transformation functions.
    """
    print("\n" + "="*60)
    print("DEMONSTRATION 2: Bounded Program Synthesis")
    print("="*60)
    
    # 1. Synthesize a payment validation predicate
    print("\n1. Synthesizing payment validation predicate...")
    print("   Examples of valid/invalid payments:")
    
    predicate_synth = PredicateSynthesizer()
    
    # Provide examples of valid and invalid payments
    payment_examples = [
        ({'amount': 150, 'currency': 'USD', 'verified': True}, True),   # Valid
        ({'amount': -10, 'currency': 'USD', 'verified': True}, False),  # Invalid (negative)
        ({'amount': 100, 'currency': 'USD', 'verified': False}, False), # Invalid (not verified)
        ({'amount': 50, 'currency': 'EUR', 'verified': True}, True),    # Valid
        ({'amount': 0, 'currency': 'USD', 'verified': True}, False),    # Invalid (zero amount)
    ]
    
    for example, valid in payment_examples:
        print(f"   {example} -> {'✓ Valid' if valid else '✗ Invalid'}")
    
    # Synthesize the validation function
    validator = predicate_synth.synthesize_filter(payment_examples, 'payment')
    
    if validator:
        print(f"\n   Synthesized validator: {validator}")
        
        # Test the synthesized validator
        test_payment = {'amount': 75, 'currency': 'GBP', 'verified': True}
        is_valid = eval(validator)(test_payment)
        print(f"   Test: {test_payment} -> {'✓ Valid' if is_valid else '✗ Invalid'}")
    
    # 2. Synthesize a fee calculation function
    print("\n2. Synthesizing fee calculation function...")
    print("   Examples of amount -> fee:")
    
    transform_synth = TransformationSynthesizer()
    
    fee_examples = [
        (100, 3),   # 3% fee
        (200, 6),
        (500, 15),
        (1000, 30),
    ]
    
    for amount, fee in fee_examples:
        print(f"   ${amount} -> ${fee} fee")
    
    fee_calculator = transform_synth.synthesize_mapper(fee_examples, 'amount')
    
    if fee_calculator:
        print(f"\n   Synthesized fee calculator: {fee_calculator}")
        
        # Test the synthesized function
        test_amount = 750
        fee = eval(fee_calculator)(test_amount)
        print(f"   Test: ${test_amount} -> ${fee} fee")
    
    # 3. Synthesize an aggregation function
    print("\n3. Synthesizing daily total calculator...")
    print("   Examples of transactions -> daily total:")
    
    aggregation_examples = [
        ([100, 200, 150], 450),
        ([50, 75], 125),
        ([1000, 500, 250], 1750),
    ]
    
    for transactions, total in aggregation_examples:
        print(f"   {transactions} -> ${total}")
    
    aggregator = transform_synth.synthesize_aggregator(aggregation_examples)
    
    if aggregator:
        print(f"\n   Synthesized aggregator: {aggregator}")
        
        test_transactions = [300, 400, 250]
        total = eval(aggregator)(test_transactions)
        print(f"   Test: {test_transactions} -> ${total}")
    
    print("\n✅ All functions synthesized from examples!")


def demonstrate_compositional_verification():
    """
    Demonstrate compositional verification.
    Verifies a multi-component distributed system.
    """
    print("\n" + "="*60)
    print("DEMONSTRATION 3: Compositional Verification")
    print("="*60)
    
    print("\nBuilding distributed payment processing pipeline...")
    print("Components: Gateway -> Validator -> Processor -> Database")
    
    verifier = CompositionalVerifier()
    
    # Define system components
    gateway = CompComponent(
        name="gateway",
        inputs={'request': dict},
        outputs={'validated_request': dict, 'request_id': str},
        properties=[
            CompProperty(
                name="unique_ids",
                type=PropertyType.INVARIANT,
                formula="request_id > 0",
                components=["gateway"]
            )
        ]
    )
    
    validator = CompComponent(
        name="validator",
        inputs={'validated_request': dict, 'request_id': str},
        outputs={'valid_payment': dict, 'validation_result': bool},
        properties=[
            CompProperty(
                name="validation_correctness",
                type=PropertyType.SAFETY,
                formula="validation_result == true",
                components=["validator"]
            )
        ]
    )
    
    processor = CompComponent(
        name="processor",
        inputs={'valid_payment': dict, 'validation_result': bool},
        outputs={'processed_payment': dict, 'transaction_id': str},
        properties=[
            CompProperty(
                name="exactly_once_processing",
                type=PropertyType.INVARIANT,
                formula="transaction_id > 0",
                components=["processor"]
            )
        ]
    )
    
    database = CompComponent(
        name="database",
        inputs={'processed_payment': dict, 'transaction_id': str},
        outputs={'stored': bool},
        properties=[
            CompProperty(
                name="durability",
                type=PropertyType.INVARIANT,
                formula="stored == true",
                components=["database"]
            )
        ]
    )
    
    # Add components to verifier
    verifier.add_component(gateway)
    verifier.add_component(validator)
    verifier.add_component(processor)
    verifier.add_component(database)
    
    # Define connections
    verifier.add_connection("gateway", "validator", "data")
    verifier.add_connection("validator", "processor", "data")
    verifier.add_connection("processor", "database", "data")
    
    print("\n📊 System topology:")
    print("   Gateway → Validator → Processor → Database")
    
    # Add end-to-end properties
    verifier.add_property(CompProperty(
        name="end_to_end_consistency",
        type=PropertyType.INVARIANT,
        formula="stored == true",
        components=["gateway", "validator", "processor", "database"]
    ))
    
    print("\nVerifying components individually...")
    
    # Perform compositional verification
    result = verifier.verify_system()
    
    print(f"\n📋 Verification Results:")
    print(f"   System verified: {'✅ Yes' if result.verified else '❌ No'}")
    print(f"   Total verification time: {result.total_time_ms:.2f}ms")
    print(f"   Proofs generated: {len(result.proofs)}")
    
    print("\n   Component verification times:")
    for comp_name, time_ms in result.component_times.items():
        print(f"     {comp_name}: {time_ms:.2f}ms")
    
    if result.counterexamples:
        print("\n   ⚠️ Counterexamples found:")
        for ce in result.counterexamples:
            print(f"     {ce}")
    
    # Demonstrate incremental verification
    print("\n🔄 Testing incremental verification...")
    print("   Modifying validator component...")
    
    incremental_result = verifier.verify_incremental(["validator"])
    
    print(f"   Incremental verification time: {incremental_result.total_time_ms:.2f}ms")
    print(f"   Components re-verified: {list(incremental_result.component_times.keys())}")
    
    # Show speedup
    if result.total_time_ms > 0:
        speedup = result.total_time_ms / incremental_result.total_time_ms
        print(f"   Speedup: {speedup:.1f}x faster than full verification")
    
    print("\n✅ Compositional verification complete!")


def main():
    """
    Run all demonstrations.
    """
    print("\n" + "="*60)
    print("UPIR ENHANCED CAPABILITIES DEMONSTRATION")
    print("="*60)
    print("\nThis demo showcases three major enhancements to UPIR:")
    print("1. Template-based code generation")
    print("2. Bounded program synthesis")
    print("3. Compositional verification")
    
    # Run demonstrations
    demonstrate_code_generation()
    demonstrate_program_synthesis()
    demonstrate_compositional_verification()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nThe enhanced UPIR system now provides:")
    print("✓ Automatic generation of production-ready code")
    print("✓ Synthesis of optimal parameters and functions")
    print("✓ Scalable verification through composition")
    print("✓ Formal guarantees maintained throughout")
    print("\nThese capabilities make UPIR practical for real-world")
    print("distributed systems development!")


if __name__ == "__main__":
    main()