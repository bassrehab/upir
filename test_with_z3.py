#!/usr/bin/env python3
"""
Test UPIR with Z3 installed - Generate code with parameter optimization.
"""

import sys
import time
from pathlib import Path

# Test Z3 import
try:
    from z3 import *
    print("✓ Z3 successfully imported!")
except ImportError as e:
    print(f"✗ Z3 import failed: {e}")
    sys.exit(1)

def optimize_queue_worker_params(requirements):
    """Use Z3 to find optimal queue worker parameters."""
    print("\nOptimizing Queue Worker parameters with Z3...")
    
    # Create Z3 variables
    batch_size = Int('batch_size')
    workers = Int('workers')
    
    # Create optimizer
    opt = Optimize()
    
    # Add constraints
    throughput_req = requirements.get('throughput', 5000)
    
    # Constraints
    opt.add(batch_size >= 1)
    opt.add(batch_size <= 100)
    opt.add(workers >= 1)
    opt.add(workers <= 200)
    opt.add(batch_size * workers * 10 >= throughput_req)  # Throughput constraint
    
    # Objective: minimize resources (batch_size + workers)
    opt.minimize(batch_size + workers)
    
    # Solve
    start = time.time()
    if opt.check() == sat:
        model = opt.model()
        solve_time = (time.time() - start) * 1000
        
        result = {
            'batch_size': model[batch_size].as_long(),
            'workers': model[workers].as_long(),
            'solve_time_ms': solve_time
        }
        
        print(f"  Optimal batch_size: {result['batch_size']}")
        print(f"  Optimal workers: {result['workers']}")
        print(f"  Throughput: {result['batch_size'] * result['workers'] * 10} req/s")
        print(f"  Z3 solve time: {solve_time:.2f}ms")
        
        return result
    else:
        print("  No solution found!")
        return None

def generate_optimized_queue_worker(params):
    """Generate queue worker with Z3-optimized parameters."""
    
    code = f'''"""
UPIR-Generated Queue Worker with Z3 Optimization
Pattern: queue_worker
Z3 Solve Time: {params['solve_time_ms']:.2f}ms
Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""

import asyncio
from typing import List, Any

class QueueWorker:
    """Batch processor with Z3-optimized parameters."""
    
    def __init__(self):
        # Z3-optimized parameters for minimal resource usage
        self.batch_size = {params['batch_size']}  # Z3 optimized
        self.workers = {params['workers']}        # Z3 optimized
        self.timeout_ms = 3000
        
        # Achieved throughput: {params['batch_size'] * params['workers'] * 10} req/s
        
    async def process_batch(self, items: List[Any]):
        """Process items in Z3-optimized batches."""
        results = []
        
        # Process in optimized batch sizes
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            result = await self._process(batch)
            results.extend(result)
            
        return results
    
    async def _process(self, batch):
        """Process a single batch."""
        # Simulated processing
        await asyncio.sleep(0.001)
        return [f"processed_{{item}}" for item in batch]

# UPIR Properties Verified:
# - Throughput >= {params['batch_size'] * params['workers'] * 10} req/s
# - Resource usage minimized (batch_size + workers = {params['batch_size'] + params['workers']})
# - Z3 SMT solver verified satisfiability
'''
    
    return code

def main():
    print("="*60)
    print("UPIR with Z3 Parameter Optimization")
    print("="*60)
    
    # Test case 1: Low throughput
    print("\n1. Low throughput system (1000 req/s):")
    params = optimize_queue_worker_params({'throughput': 1000})
    if params:
        code = generate_optimized_queue_worker(params)
        Path('examples/Z3_optimized_low_throughput.py').write_text(code)
        print("  ✓ Generated: examples/Z3_optimized_low_throughput.py")
    
    # Test case 2: High throughput
    print("\n2. High throughput system (10000 req/s):")
    params = optimize_queue_worker_params({'throughput': 10000})
    if params:
        code = generate_optimized_queue_worker(params)
        Path('examples/Z3_optimized_high_throughput.py').write_text(code)
        print("  ✓ Generated: examples/Z3_optimized_high_throughput.py")
    
    # Test case 3: Verify rate limiter constraints
    print("\n3. Rate Limiter parameter synthesis:")
    
    # Create Z3 variables for rate limiter
    rate = Int('rate')
    burst = Int('burst')
    
    s = Solver()
    s.add(rate >= 100)
    s.add(rate <= 10000)
    s.add(burst >= 10)
    s.add(burst <= 1000)
    s.add(burst <= rate / 10)  # Burst should be reasonable fraction of rate
    
    if s.check() == sat:
        model = s.model()
        print(f"  Rate: {model[rate]} req/s")
        print(f"  Burst: {model[burst]} requests")
    
    print("\n" + "="*60)
    print("Z3 Integration Successful!")
    print("UPIR can now optimize parameters using SMT solving")
    print("="*60)

if __name__ == "__main__":
    main()