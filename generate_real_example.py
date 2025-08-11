#!/usr/bin/env python3
"""
Actually run UPIR to generate real code.
"""

import sys
import json
from pathlib import Path

# Add UPIR to path
sys.path.insert(0, '/Users/subhadipmitra/Dev/upir')

from upir.codegen.generator import CodeGenerator
from upir.codegen.templates import (
    QueueWorkerTemplate,
    RateLimiterTemplate,
    CircuitBreakerTemplate
)

def main():
    """Generate real code using UPIR."""
    
    print("="*60)
    print("UPIR Code Generation - Real Example")
    print("="*60)
    
    # Initialize generator
    generator = CodeGenerator()
    
    # Register templates
    templates = {
        'queue_worker': QueueWorkerTemplate(),
        'rate_limiter': RateLimiterTemplate(),
        'circuit_breaker': CircuitBreakerTemplate()
    }
    
    for name, template in templates.items():
        generator.register_template(name, template)
    
    # Generate Rate Limiter
    print("\n1. Generating Rate Limiter...")
    rate_limiter_spec = {
        'pattern': 'rate_limiter',
        'requirements': {
            'requests_per_second': 1000,
            'burst_size': 100
        }
    }
    
    try:
        rate_limiter_code = generator.generate(rate_limiter_spec, language='python')
        
        # Save generated code
        output_file = Path('examples/upir_generated_rate_limiter.py')
        output_file.write_text(rate_limiter_code['code'])
        
        print(f"   ✓ Generated: {output_file}")
        print(f"   Parameters: {rate_limiter_code.get('parameters', {})}")
        print(f"   Generation time: {rate_limiter_code.get('generation_time_ms', 'N/A')}ms")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Generate Queue Worker
    print("\n2. Generating Queue Worker...")
    queue_worker_spec = {
        'pattern': 'queue_worker',
        'requirements': {
            'throughput': 5000,
            'batch_size': 25,
            'workers': 100
        }
    }
    
    try:
        queue_worker_code = generator.generate(queue_worker_spec, language='python')
        
        output_file = Path('examples/upir_generated_queue_worker.py')
        output_file.write_text(queue_worker_code['code'])
        
        print(f"   ✓ Generated: {output_file}")
        print(f"   Parameters: {queue_worker_code.get('parameters', {})}")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Generate Circuit Breaker
    print("\n3. Generating Circuit Breaker...")
    circuit_breaker_spec = {
        'pattern': 'circuit_breaker',
        'requirements': {
            'failure_threshold': 5,
            'recovery_timeout': 10
        }
    }
    
    try:
        circuit_breaker_code = generator.generate(circuit_breaker_spec, language='python')
        
        output_file = Path('examples/upir_generated_circuit_breaker.py')
        output_file.write_text(circuit_breaker_code['code'])
        
        print(f"   ✓ Generated: {output_file}")
        print(f"   Parameters: {circuit_breaker_code.get('parameters', {})}")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n" + "="*60)
    print("Generation complete! Check examples/ directory for output.")
    print("="*60)

if __name__ == "__main__":
    main()