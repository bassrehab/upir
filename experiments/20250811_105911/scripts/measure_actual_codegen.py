#!/usr/bin/env python3
"""
Measure actual code generation performance from UPIR templates.
"""

import sys
import time
import json
from pathlib import Path
sys.path.append('/Users/subhadipmitra/Dev/upir')

from upir.codegen.templates import (
    QueueWorkerTemplate,
    RateLimiterTemplate, 
    CircuitBreakerTemplate,
    RetryTemplate,
    CacheTemplate,
    LoadBalancerTemplate
)

def measure_template_generation():
    """Measure actual template generation times."""
    
    templates = {
        'queue_worker': QueueWorkerTemplate(),
        'rate_limiter': RateLimiterTemplate(),
        'circuit_breaker': CircuitBreakerTemplate(),
        'retry': RetryTemplate(),
        'cache': CacheTemplate(),
        'load_balancer': LoadBalancerTemplate()
    }
    
    iterations = 100
    results = {}
    
    print("Measuring actual code generation performance...")
    print("=" * 60)
    
    for name, template in templates.items():
        times = []
        
        # Warm up
        for _ in range(10):
            _ = template.generate('python')
        
        # Measure
        for i in range(iterations):
            start = time.perf_counter()
            code = template.generate('python')
            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        results[name] = {
            'average_ms': avg_time,
            'min_ms': min_time,
            'max_ms': max_time,
            'code_lines': len(code.code.split('\n')) if hasattr(code, 'code') else 0
        }
        
        print(f"{name:20} | Avg: {avg_time:6.2f}ms | Min: {min_time:6.2f}ms | Max: {max_time:6.2f}ms")
    
    # Save results
    output = {
        'experiment': 'actual_code_generation_timing',
        'iterations': iterations,
        'results': results,
        'average_ms': sum(r['average_ms'] for r in results.values()) / len(results)
    }
    
    output_path = Path(__file__).parent.parent / 'data' / 'actual_codegen_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print("=" * 60)
    print(f"Overall average: {output['average_ms']:.2f}ms")
    print(f"Results saved to {output_path}")
    
    return output


if __name__ == "__main__":
    measure_template_generation()