#!/usr/bin/env python3
"""
Code Generation Benchmarks - Exhaustive Testing
Tests all templates with volumetric load (1000+ iterations each)
"""

import json
import time
import random
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add UPIR to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from upir.codegen import (
    CodeGenerator,
    QueueWorkerTemplate,
    RateLimiterTemplate,
    CircuitBreakerTemplate,
    RetryTemplate,
    CacheTemplate,
    LoadBalancerTemplate
)

EXPERIMENT_DIR = Path(__file__).parent.parent
DATA_DIR = EXPERIMENT_DIR / "data"
RESULTS_DIR = EXPERIMENT_DIR / "results"

# Volumetric test configuration
ITERATIONS_PER_TEMPLATE = 1000  # No shortcuts!
PARAMETER_VARIATIONS = 50  # Different parameter combinations
LANGUAGES = ['python', 'go', 'javascript']

def benchmark_template(template_class, template_name: str) -> Dict[str, Any]:
    """Benchmark a single template exhaustively."""
    print(f"\nBenchmarking {template_name}...")
    print(f"  Iterations: {ITERATIONS_PER_TEMPLATE}")
    print(f"  Languages: {LANGUAGES}")
    
    generator = CodeGenerator()
    template = template_class()
    generator.register_template(template)
    
    results = {
        'template': template_name,
        'iterations': ITERATIONS_PER_TEMPLATE,
        'languages': {},
        'parameter_synthesis': {
            'times': [],
            'success_rate': 0,
            'failures': []
        },
        'statistics': {}
    }
    
    # Test each language
    for language in LANGUAGES:
        lang_results = {
            'generation_times': [],
            'code_sizes': [],
            'success_count': 0,
            'failure_count': 0,
            'errors': []
        }
        
        print(f"  Testing {language}...")
        
        for i in range(ITERATIONS_PER_TEMPLATE):
            # Generate random requirements
            requirements = generate_random_requirements(template_name)
            
            try:
                # Time parameter synthesis
                synthesis_start = time.perf_counter()
                params = template.synthesize_parameters(requirements)
                synthesis_time = (time.perf_counter() - synthesis_start) * 1000
                results['parameter_synthesis']['times'].append(synthesis_time)
                
                # Time code generation
                gen_start = time.perf_counter()
                code = template.generate(language, params)
                gen_time = (time.perf_counter() - gen_start) * 1000
                
                lang_results['generation_times'].append(gen_time)
                lang_results['code_sizes'].append(len(code))
                lang_results['success_count'] += 1
                
            except Exception as e:
                lang_results['failure_count'] += 1
                lang_results['errors'].append(str(e))
                results['parameter_synthesis']['failures'].append({
                    'requirements': requirements,
                    'error': str(e)
                })
            
            # Progress indicator every 100 iterations
            if (i + 1) % 100 == 0:
                print(f"    Progress: {i + 1}/{ITERATIONS_PER_TEMPLATE}")
        
        # Calculate statistics
        if lang_results['generation_times']:
            lang_results['statistics'] = {
                'mean_time': sum(lang_results['generation_times']) / len(lang_results['generation_times']),
                'min_time': min(lang_results['generation_times']),
                'max_time': max(lang_results['generation_times']),
                'mean_size': sum(lang_results['code_sizes']) / len(lang_results['code_sizes']),
                'success_rate': lang_results['success_count'] / ITERATIONS_PER_TEMPLATE * 100
            }
        
        results['languages'][language] = lang_results
    
    # Calculate overall statistics
    if results['parameter_synthesis']['times']:
        results['parameter_synthesis']['statistics'] = {
            'mean_time': sum(results['parameter_synthesis']['times']) / len(results['parameter_synthesis']['times']),
            'min_time': min(results['parameter_synthesis']['times']),
            'max_time': max(results['parameter_synthesis']['times']),
            'success_rate': (len(results['parameter_synthesis']['times']) / 
                           (ITERATIONS_PER_TEMPLATE * len(LANGUAGES))) * 100
        }
    
    return results

def generate_random_requirements(template_name: str) -> Dict[str, Any]:
    """Generate random but valid requirements for a template."""
    requirements = {}
    
    if template_name == 'queue_worker':
        requirements['batch_size'] = random.randint(1, 100)
        requirements['workers'] = random.randint(1, 50)
        if random.random() > 0.5:
            requirements['timeout_ms'] = random.randint(100, 10000)
    
    elif template_name == 'rate_limiter':
        requirements['requests_per_second'] = random.randint(10, 10000)
        if random.random() > 0.5:
            requirements['burst_size'] = random.randint(10, 1000)
    
    elif template_name == 'circuit_breaker':
        requirements['failure_threshold'] = random.randint(1, 20)
        if random.random() > 0.5:
            requirements['recovery_timeout_ms'] = random.randint(1000, 30000)
    
    elif template_name == 'retry':
        requirements['max_attempts'] = random.randint(1, 10)
        if random.random() > 0.5:
            requirements['initial_delay_ms'] = random.randint(10, 1000)
    
    elif template_name == 'cache':
        requirements['max_size'] = random.randint(100, 10000)
        if random.random() > 0.5:
            requirements['ttl_seconds'] = random.randint(60, 3600)
    
    elif template_name == 'load_balancer':
        algorithms = ['round_robin', 'least_connections', 'random', 'weighted']
        requirements['algorithm'] = random.choice(algorithms)
        if random.random() > 0.5:
            requirements['health_check_interval_ms'] = random.randint(1000, 10000)
    
    return requirements

def run_volumetric_tests():
    """Run volumetric tests on all templates."""
    templates = [
        (QueueWorkerTemplate, 'queue_worker'),
        (RateLimiterTemplate, 'rate_limiter'),
        (CircuitBreakerTemplate, 'circuit_breaker'),
        (RetryTemplate, 'retry'),
        (CacheTemplate, 'cache'),
        (LoadBalancerTemplate, 'load_balancer')
    ]
    
    all_results = {
        'experiment': 'code_generation_benchmarks',
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'iterations_per_template': ITERATIONS_PER_TEMPLATE,
            'parameter_variations': PARAMETER_VARIATIONS,
            'languages': LANGUAGES
        },
        'templates': {},
        'summary': {}
    }
    
    print(f"""
{'='*60}
CODE GENERATION BENCHMARKS
{'='*60}
Templates: {len(templates)}
Iterations per template: {ITERATIONS_PER_TEMPLATE}
Total iterations: {len(templates) * ITERATIONS_PER_TEMPLATE * len(LANGUAGES)}
Languages: {', '.join(LANGUAGES)}
{'='*60}
""")
    
    total_start = time.time()
    
    for template_class, template_name in templates:
        results = benchmark_template(template_class, template_name)
        all_results['templates'][template_name] = results
    
    total_time = time.time() - total_start
    
    # Calculate summary statistics
    all_results['summary'] = {
        'total_time_seconds': total_time,
        'total_iterations': len(templates) * ITERATIONS_PER_TEMPLATE * len(LANGUAGES),
        'iterations_per_second': (len(templates) * ITERATIONS_PER_TEMPLATE * len(LANGUAGES)) / total_time,
        'template_statistics': {}
    }
    
    for template_name, results in all_results['templates'].items():
        template_stats = {
            'mean_synthesis_time': 0,
            'mean_generation_time': {},
            'overall_success_rate': 0
        }
        
        if 'statistics' in results['parameter_synthesis']:
            template_stats['mean_synthesis_time'] = results['parameter_synthesis']['statistics']['mean_time']
        
        success_counts = []
        for lang, lang_results in results['languages'].items():
            if 'statistics' in lang_results:
                template_stats['mean_generation_time'][lang] = lang_results['statistics']['mean_time']
                success_counts.append(lang_results['statistics']['success_rate'])
        
        if success_counts:
            template_stats['overall_success_rate'] = sum(success_counts) / len(success_counts)
        
        all_results['summary']['template_statistics'][template_name] = template_stats
    
    # Save detailed results
    results_file = DATA_DIR / "code_generation_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save summary
    summary_file = RESULTS_DIR / "code_generation_summary.json"
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_iterations': all_results['summary']['total_iterations'],
        'total_time': total_time,
        'iterations_per_second': all_results['summary']['iterations_per_second'],
        'templates': {}
    }
    
    for template_name, stats in all_results['summary']['template_statistics'].items():
        summary['templates'][template_name] = {
            'mean_synthesis_time_ms': stats['mean_synthesis_time'],
            'mean_generation_time_ms': stats['mean_generation_time'],
            'success_rate': stats['overall_success_rate']
        }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"""
{'='*60}
RESULTS SUMMARY
{'='*60}
Total time: {total_time:.2f} seconds
Total iterations: {all_results['summary']['total_iterations']}
Throughput: {all_results['summary']['iterations_per_second']:.2f} iterations/second

Per-Template Performance:
""")
    
    for template_name, stats in all_results['summary']['template_statistics'].items():
        print(f"\n{template_name}:")
        print(f"  Synthesis time: {stats['mean_synthesis_time']:.2f}ms")
        print(f"  Generation times:")
        for lang, time_ms in stats['mean_generation_time'].items():
            print(f"    {lang}: {time_ms:.2f}ms")
        print(f"  Success rate: {stats['overall_success_rate']:.1f}%")
    
    print(f"""
{'='*60}
Full results saved to: {results_file}
Summary saved to: {summary_file}
{'='*60}
""")

if __name__ == "__main__":
    run_volumetric_tests()