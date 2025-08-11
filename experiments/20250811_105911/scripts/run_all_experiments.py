#!/usr/bin/env python3
"""
Comprehensive UPIR Experiments Runner
No shortcuts - exhaustive testing on real GCP infrastructure
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path

# Add UPIR to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# GCP Configuration
GCP_PROJECT = "subhadipmitra-pso-team-369906"
GCP_REGION = "us-central1"
EXPERIMENT_DIR = Path(__file__).parent.parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
DATA_DIR = EXPERIMENT_DIR / "data"
LOGS_DIR = EXPERIMENT_DIR / "logs"

print(f"""
{'='*60}
UPIR COMPREHENSIVE EXPERIMENTS
{'='*60}
Date: {datetime.now().isoformat()}
GCP Project: {GCP_PROJECT}
Region: {GCP_REGION}
Experiment Directory: {EXPERIMENT_DIR}
{'='*60}
""")

def setup_gcp():
    """Configure GCP project and authentication."""
    print("\n[1/9] Setting up GCP Configuration...")
    
    # Set the project
    cmd = f"gcloud config set project {GCP_PROJECT}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Warning: Could not set project. Make sure gcloud is authenticated.")
        print(f"Run: gcloud auth login")
        print(f"Then: gcloud config set project {GCP_PROJECT}")
    else:
        print(f"✓ GCP Project set to: {GCP_PROJECT}")
    
    # Check authentication
    cmd = "gcloud auth list --filter=status:ACTIVE --format='value(account)'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout.strip():
        print(f"✓ Authenticated as: {result.stdout.strip()}")
    else:
        print("⚠ No active authentication. Please run: gcloud auth login")
        return False
    
    return True

def run_code_generation_benchmarks():
    """Run exhaustive code generation benchmarks."""
    print("\n[2/9] Running Code Generation Benchmarks...")
    print("Volumetric test: 1000 iterations per template")
    
    script_path = EXPERIMENT_DIR / "scripts" / "benchmark_code_generation.py"
    log_path = LOGS_DIR / "code_generation.log"
    
    with open(log_path, 'w') as log:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True
        )
    
    if result.returncode == 0:
        print("✓ Code generation benchmarks complete")
        return True
    else:
        print(f"✗ Code generation benchmarks failed. Check {log_path}")
        return False

def run_synthesis_benchmarks():
    """Run exhaustive synthesis benchmarks."""
    print("\n[3/9] Running Program Synthesis Benchmarks...")
    print("Testing 100+ examples per function type")
    
    script_path = EXPERIMENT_DIR / "scripts" / "benchmark_synthesis.py"
    log_path = LOGS_DIR / "synthesis.log"
    
    with open(log_path, 'w') as log:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True
        )
    
    if result.returncode == 0:
        print("✓ Synthesis benchmarks complete")
        return True
    else:
        print(f"✗ Synthesis benchmarks failed. Check {log_path}")
        return False

def run_verification_benchmarks():
    """Run compositional verification benchmarks."""
    print("\n[4/9] Running Compositional Verification Benchmarks...")
    print("Scaling test: 2 to 100+ components")
    
    script_path = EXPERIMENT_DIR / "scripts" / "benchmark_verification.py"
    log_path = LOGS_DIR / "verification.log"
    
    with open(log_path, 'w') as log:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True
        )
    
    if result.returncode == 0:
        print("✓ Verification benchmarks complete")
        return True
    else:
        print(f"✗ Verification benchmarks failed. Check {log_path}")
        return False

def deploy_to_cloud_run():
    """Deploy UPIR to Cloud Run and collect metrics."""
    print("\n[5/9] Deploying to Cloud Run...")
    print(f"Project: {GCP_PROJECT}")
    
    script_path = EXPERIMENT_DIR / "scripts" / "deploy_cloud_run.py"
    log_path = LOGS_DIR / "deployment.log"
    
    with open(log_path, 'w') as log:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True
        )
    
    if result.returncode == 0:
        print("✓ Cloud Run deployment complete")
        return True
    else:
        print(f"✗ Cloud Run deployment failed. Check {log_path}")
        return False

def run_load_tests():
    """Run load tests against deployed service."""
    print("\n[6/9] Running Load Tests...")
    print("Target: 10,000+ requests")
    
    script_path = EXPERIMENT_DIR / "scripts" / "load_test.py"
    log_path = LOGS_DIR / "load_test.log"
    
    with open(log_path, 'w') as log:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True
        )
    
    if result.returncode == 0:
        print("✓ Load tests complete")
        return True
    else:
        print(f"✗ Load tests failed. Check {log_path}")
        return False

def run_learning_experiments():
    """Run learning system experiments."""
    print("\n[7/9] Running Learning System Experiments...")
    print("Training for 100+ episodes")
    
    script_path = EXPERIMENT_DIR / "scripts" / "learning_experiments.py"
    log_path = LOGS_DIR / "learning.log"
    
    with open(log_path, 'w') as log:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True
        )
    
    if result.returncode == 0:
        print("✓ Learning experiments complete")
        return True
    else:
        print(f"✗ Learning experiments failed. Check {log_path}")
        return False

def collect_cloud_metrics():
    """Collect metrics from Cloud Monitoring."""
    print("\n[8/9] Collecting Cloud Monitoring Metrics...")
    
    script_path = EXPERIMENT_DIR / "scripts" / "collect_metrics.py"
    log_path = LOGS_DIR / "metrics.log"
    
    with open(log_path, 'w') as log:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True
        )
    
    if result.returncode == 0:
        print("✓ Metrics collection complete")
        return True
    else:
        print(f"✗ Metrics collection failed. Check {log_path}")
        return False

def generate_visualizations():
    """Generate all visualizations from experimental data."""
    print("\n[9/9] Generating Visualizations...")
    
    script_path = EXPERIMENT_DIR / "scripts" / "generate_visualizations.py"
    log_path = LOGS_DIR / "visualizations.log"
    
    with open(log_path, 'w') as log:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True
        )
    
    if result.returncode == 0:
        print("✓ Visualizations generated")
        return True
    else:
        print(f"✗ Visualization generation failed. Check {log_path}")
        return False

def main():
    """Run all experiments."""
    start_time = time.time()
    
    # Create directories
    RESULTS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    
    # Track results
    results = {
        'start_time': datetime.now().isoformat(),
        'gcp_project': GCP_PROJECT,
        'region': GCP_REGION,
        'experiments': {}
    }
    
    # Run experiments
    steps = [
        ('gcp_setup', setup_gcp),
        ('code_generation', run_code_generation_benchmarks),
        ('synthesis', run_synthesis_benchmarks),
        ('verification', run_verification_benchmarks),
        ('cloud_deployment', deploy_to_cloud_run),
        ('load_testing', run_load_tests),
        ('learning', run_learning_experiments),
        ('metrics_collection', collect_cloud_metrics),
        ('visualization', generate_visualizations)
    ]
    
    for name, func in steps:
        try:
            success = func()
            results['experiments'][name] = {
                'status': 'success' if success else 'failed',
                'timestamp': datetime.now().isoformat()
            }
            if not success and name == 'gcp_setup':
                print("\n⚠ GCP setup required. Stopping experiments.")
                break
        except Exception as e:
            print(f"✗ Error in {name}: {str(e)}")
            results['experiments'][name] = {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    # Save results summary
    results['end_time'] = datetime.now().isoformat()
    results['total_time_seconds'] = time.time() - start_time
    
    summary_path = RESULTS_DIR / "experiment_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"""
{'='*60}
EXPERIMENTS COMPLETE
{'='*60}
Total Time: {results['total_time_seconds']:.2f} seconds
Results: {summary_path}
{'='*60}
""")

if __name__ == "__main__":
    main()