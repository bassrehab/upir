#!/usr/bin/env python3
"""
Program Synthesis Benchmarks - Exhaustive Testing
Tests all synthesis capabilities with 100+ examples per function type
"""

import json
import time
import random
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Add UPIR to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from upir.synthesis.program_synthesis import (
    ProgramSynthesizer,
    PredicateSynthesizer,
    TransformationSynthesizer,
    SynthesisSpec
)

EXPERIMENT_DIR = Path(__file__).parent.parent
DATA_DIR = EXPERIMENT_DIR / "data"
RESULTS_DIR = EXPERIMENT_DIR / "results"

# Exhaustive test configuration
EXAMPLES_PER_TYPE = 100  # 100+ examples per function type
MAX_DEPTHS = [1, 2, 3, 4, 5]  # Test different complexity levels
PREDICATE_TYPES = ['comparison', 'boolean', 'arithmetic', 'mixed']
TRANSFORMATION_TYPES = ['linear', 'quadratic', 'modular', 'conditional']

def generate_predicate_examples(predicate_type: str, count: int) -> List[Tuple[Dict, bool]]:
    """Generate examples for predicate synthesis."""
    examples = []
    
    if predicate_type == 'comparison':
        # Simple comparison: x > threshold
        threshold = random.randint(20, 80)
        for _ in range(count):
            value = random.randint(0, 100)
            examples.append(({'x': value}, value > threshold))
    
    elif predicate_type == 'boolean':
        # Boolean combination: x > 30 and x < 70
        for _ in range(count):
            value = random.randint(0, 100)
            examples.append(({'x': value}, 30 < value < 70))
    
    elif predicate_type == 'arithmetic':
        # Arithmetic predicate: x % 5 == 0
        for _ in range(count):
            value = random.randint(0, 100)
            examples.append(({'x': value}, value % 5 == 0))
    
    elif predicate_type == 'mixed':
        # Complex: (x > 20 and x < 80) or x % 10 == 0
        for _ in range(count):
            value = random.randint(0, 100)
            result = (20 < value < 80) or (value % 10 == 0)
            examples.append(({'x': value}, result))
    
    return examples

def generate_transformation_examples(transform_type: str, count: int) -> List[Tuple[Any, Any]]:
    """Generate examples for transformation synthesis."""
    examples = []
    
    if transform_type == 'linear':
        # Linear: y = ax + b
        a = random.randint(2, 10)
        b = random.randint(-10, 10)
        for _ in range(count):
            x = random.randint(-50, 50)
            examples.append((x, a * x + b))
    
    elif transform_type == 'quadratic':
        # Quadratic: y = x^2 + c
        c = random.randint(-10, 10)
        for _ in range(count):
            x = random.randint(-10, 10)
            examples.append((x, x * x + c))
    
    elif transform_type == 'modular':
        # Modular: y = x % m
        m = random.randint(3, 10)
        for _ in range(count):
            x = random.randint(0, 100)
            examples.append((x, x % m))
    
    elif transform_type == 'conditional':
        # Conditional: y = x * 2 if x > 0 else x - 5
        for _ in range(count):
            x = random.randint(-20, 20)
            y = x * 2 if x > 0 else x - 5
            examples.append((x, y))
    
    return examples

def benchmark_predicate_synthesis() -> Dict[str, Any]:
    """Benchmark predicate synthesis exhaustively."""
    print("\nBenchmarking Predicate Synthesis...")
    print(f"  Types: {PREDICATE_TYPES}")
    print(f"  Examples per type: {EXAMPLES_PER_TYPE}")
    print(f"  Max depths: {MAX_DEPTHS}")
    
    synthesizer = PredicateSynthesizer()
    results = {
        'function_type': 'predicate',
        'types_tested': PREDICATE_TYPES,
        'examples_per_type': EXAMPLES_PER_TYPE,
        'depths_tested': MAX_DEPTHS,
        'results_by_type': {}
    }
    
    for pred_type in PREDICATE_TYPES:
        print(f"\n  Testing {pred_type} predicates...")
        type_results = {
            'type': pred_type,
            'depths': {}
        }
        
        # Generate examples
        examples = generate_predicate_examples(pred_type, EXAMPLES_PER_TYPE)
        
        for depth in MAX_DEPTHS:
            print(f"    Depth {depth}...")
            depth_results = {
                'synthesis_times': [],
                'success_count': 0,
                'failure_count': 0,
                'synthesized_functions': []
            }
            
            # Try synthesis with increasing subsets of examples
            for num_examples in [5, 10, 20, 50, EXAMPLES_PER_TYPE]:
                subset = examples[:num_examples]
                
                # Create synthesis spec
                spec = SynthesisSpec(
                    name=f"{pred_type}_depth{depth}",
                    inputs={'x': int},
                    output_type=bool,
                    examples=subset,
                    max_depth=depth
                )
                
                # Time synthesis
                start_time = time.perf_counter()
                try:
                    result = synthesizer.synthesize(spec)
                    synthesis_time = (time.perf_counter() - start_time) * 1000
                    
                    if result:
                        depth_results['synthesis_times'].append(synthesis_time)
                        depth_results['success_count'] += 1
                        depth_results['synthesized_functions'].append({
                            'code': result.code,
                            'time_ms': synthesis_time,
                            'num_examples': num_examples
                        })
                    else:
                        depth_results['failure_count'] += 1
                except Exception as e:
                    depth_results['failure_count'] += 1
            
            # Calculate statistics
            if depth_results['synthesis_times']:
                depth_results['statistics'] = {
                    'mean_time': sum(depth_results['synthesis_times']) / len(depth_results['synthesis_times']),
                    'min_time': min(depth_results['synthesis_times']),
                    'max_time': max(depth_results['synthesis_times']),
                    'success_rate': depth_results['success_count'] / (depth_results['success_count'] + depth_results['failure_count']) * 100
                }
            
            type_results['depths'][depth] = depth_results
        
        results['results_by_type'][pred_type] = type_results
    
    return results

def benchmark_transformation_synthesis() -> Dict[str, Any]:
    """Benchmark transformation synthesis exhaustively."""
    print("\nBenchmarking Transformation Synthesis...")
    print(f"  Types: {TRANSFORMATION_TYPES}")
    print(f"  Examples per type: {EXAMPLES_PER_TYPE}")
    
    synthesizer = TransformationSynthesizer()
    results = {
        'function_type': 'transformation',
        'types_tested': TRANSFORMATION_TYPES,
        'examples_per_type': EXAMPLES_PER_TYPE,
        'results_by_type': {}
    }
    
    for transform_type in TRANSFORMATION_TYPES:
        print(f"\n  Testing {transform_type} transformations...")
        type_results = {
            'type': transform_type,
            'synthesis_attempts': [],
            'success_count': 0,
            'failure_count': 0
        }
        
        # Generate examples
        examples = generate_transformation_examples(transform_type, EXAMPLES_PER_TYPE)
        
        # Try synthesis with increasing example counts
        for num_examples in [3, 5, 10, 20, 50]:
            subset = examples[:num_examples]
            
            start_time = time.perf_counter()
            try:
                result = synthesizer.synthesize_mapper(subset)
                synthesis_time = (time.perf_counter() - start_time) * 1000
                
                if result:
                    type_results['synthesis_attempts'].append({
                        'num_examples': num_examples,
                        'success': True,
                        'time_ms': synthesis_time,
                        'function': result
                    })
                    type_results['success_count'] += 1
                else:
                    type_results['failure_count'] += 1
            except Exception as e:
                type_results['failure_count'] += 1
        
        # Calculate statistics
        successful_times = [a['time_ms'] for a in type_results['synthesis_attempts'] if a['success']]
        if successful_times:
            type_results['statistics'] = {
                'mean_time': sum(successful_times) / len(successful_times),
                'min_time': min(successful_times),
                'max_time': max(successful_times),
                'success_rate': type_results['success_count'] / (type_results['success_count'] + type_results['failure_count']) * 100
            }
        
        results['results_by_type'][transform_type] = type_results
    
    return results

def benchmark_aggregator_synthesis() -> Dict[str, Any]:
    """Benchmark aggregator synthesis."""
    print("\nBenchmarking Aggregator Synthesis...")
    
    synthesizer = TransformationSynthesizer()
    results = {
        'function_type': 'aggregator',
        'operations_tested': ['sum', 'max', 'min', 'average'],
        'results': []
    }
    
    operations = {
        'sum': lambda lst: sum(lst),
        'max': lambda lst: max(lst),
        'min': lambda lst: min(lst),
        'average': lambda lst: sum(lst) / len(lst) if lst else 0
    }
    
    for op_name, op_func in operations.items():
        print(f"  Testing {op_name} aggregation...")
        
        # Generate examples
        examples = []
        for _ in range(20):
            lst = [random.randint(1, 100) for _ in range(random.randint(3, 10))]
            examples.append((lst, op_func(lst)))
        
        start_time = time.perf_counter()
        try:
            result = synthesizer.synthesize_aggregator(examples, op_name)
            synthesis_time = (time.perf_counter() - start_time) * 1000
            
            results['results'].append({
                'operation': op_name,
                'success': result is not None,
                'time_ms': synthesis_time,
                'function': result if result else None
            })
        except Exception as e:
            results['results'].append({
                'operation': op_name,
                'success': False,
                'error': str(e)
            })
    
    return results

def run_synthesis_benchmarks():
    """Run all synthesis benchmarks."""
    all_results = {
        'experiment': 'synthesis_benchmarks',
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'examples_per_type': EXAMPLES_PER_TYPE,
            'max_depths': MAX_DEPTHS,
            'predicate_types': PREDICATE_TYPES,
            'transformation_types': TRANSFORMATION_TYPES
        },
        'benchmarks': {}
    }
    
    print(f"""
{'='*60}
PROGRAM SYNTHESIS BENCHMARKS
{'='*60}
Examples per type: {EXAMPLES_PER_TYPE}
Max depths tested: {MAX_DEPTHS}
Predicate types: {len(PREDICATE_TYPES)}
Transformation types: {len(TRANSFORMATION_TYPES)}
{'='*60}
""")
    
    total_start = time.time()
    
    # Run benchmarks
    all_results['benchmarks']['predicates'] = benchmark_predicate_synthesis()
    all_results['benchmarks']['transformations'] = benchmark_transformation_synthesis()
    all_results['benchmarks']['aggregators'] = benchmark_aggregator_synthesis()
    
    total_time = time.time() - total_start
    all_results['total_time_seconds'] = total_time
    
    # Save detailed results
    results_file = DATA_DIR / "synthesis_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_time': total_time,
        'function_types': {}
    }
    
    # Summarize predicate results
    pred_stats = []
    for pred_type, type_results in all_results['benchmarks']['predicates']['results_by_type'].items():
        for depth, depth_results in type_results['depths'].items():
            if 'statistics' in depth_results:
                pred_stats.append({
                    'type': pred_type,
                    'depth': depth,
                    'mean_time': depth_results['statistics']['mean_time'],
                    'success_rate': depth_results['statistics']['success_rate']
                })
    
    if pred_stats:
        summary['function_types']['predicates'] = {
            'mean_time_ms': sum(s['mean_time'] for s in pred_stats) / len(pred_stats),
            'mean_success_rate': sum(s['success_rate'] for s in pred_stats) / len(pred_stats),
            'configurations_tested': len(pred_stats)
        }
    
    # Summarize transformation results
    trans_stats = []
    for trans_type, type_results in all_results['benchmarks']['transformations']['results_by_type'].items():
        if 'statistics' in type_results:
            trans_stats.append(type_results['statistics'])
    
    if trans_stats:
        summary['function_types']['transformations'] = {
            'mean_time_ms': sum(s['mean_time'] for s in trans_stats) / len(trans_stats),
            'mean_success_rate': sum(s['success_rate'] for s in trans_stats) / len(trans_stats),
            'types_tested': len(trans_stats)
        }
    
    # Save summary
    summary_file = RESULTS_DIR / "synthesis_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"""
{'='*60}
RESULTS SUMMARY
{'='*60}
Total time: {total_time:.2f} seconds

Synthesis Performance:
""")
    
    for func_type, stats in summary['function_types'].items():
        print(f"\n{func_type}:")
        print(f"  Mean time: {stats.get('mean_time_ms', 0):.2f}ms")
        print(f"  Success rate: {stats.get('mean_success_rate', 0):.1f}%")
    
    print(f"""
{'='*60}
Full results saved to: {results_file}
Summary saved to: {summary_file}
{'='*60}
""")

if __name__ == "__main__":
    run_synthesis_benchmarks()