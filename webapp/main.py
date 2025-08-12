"""
UPIR Web Application - App Engine
Universal Plan Intermediate Representation Demo
"""

from flask import Flask, render_template, request, jsonify, session
import json
import time
import hashlib
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'upir-demo-secret-key-change-in-production')

# Sample UPIR templates for demonstration
TEMPLATES = {
    'queue_worker': {
        'name': 'Queue Worker',
        'description': 'High-throughput batch processing with Z3-optimized parameters',
        'generation_time': 2.83,
        'parameters': {'batch_size': 94, 'workers': 14}
    },
    'rate_limiter': {
        'name': 'Rate Limiter',
        'description': 'Token bucket rate limiting with burst support',
        'generation_time': 1.60,
        'parameters': {'rate': 1000, 'burst_size': 100}
    },
    'circuit_breaker': {
        'name': 'Circuit Breaker',
        'description': 'Fault tolerance with automatic recovery',
        'generation_time': 1.51,
        'parameters': {'failure_threshold': 5, 'recovery_timeout': 10}
    },
    'cache': {
        'name': 'Cache',
        'description': 'LRU cache with configurable TTL',
        'generation_time': 1.47,
        'parameters': {'size': 10000, 'ttl_seconds': 3600}
    },
    'load_balancer': {
        'name': 'Load Balancer',
        'description': 'Round-robin load distribution with health checks',
        'generation_time': 1.37,
        'parameters': {'backends': 5, 'health_check_interval': 30}
    }
}

@app.route('/')
def index():
    """Main landing page."""
    return render_template('index.html')

@app.route('/demo')
def demo():
    """Interactive demo page."""
    return render_template('demo.html', templates=TEMPLATES)

@app.route('/api/generate', methods=['POST'])
def generate_code():
    """API endpoint for code generation."""
    data = request.json
    template_type = data.get('template', 'queue_worker')
    language = data.get('language', 'python')
    
    if template_type not in TEMPLATES:
        return jsonify({'error': 'Invalid template type'}), 400
    
    template = TEMPLATES[template_type]
    
    # Simulate code generation
    start_time = time.time()
    
    # Generate sample code based on template
    if language == 'python':
        code = generate_python_code(template_type, template)
    elif language == 'go':
        code = generate_go_code(template_type, template)
    else:
        code = generate_javascript_code(template_type, template)
    
    generation_time = (time.time() - start_time) * 1000  # Convert to ms
    
    return jsonify({
        'success': True,
        'code': code,
        'language': language,
        'template': template['name'],
        'generation_time': round(generation_time, 2),
        'parameters': template['parameters']
    })

@app.route('/api/verify', methods=['POST'])
def verify_specification():
    """API endpoint for specification verification."""
    data = request.json
    spec = data.get('specification', '')
    
    # Simulate verification
    time.sleep(0.1)  # Simulate processing
    
    # Basic validation
    errors = []
    warnings = []
    
    if 'system' not in spec:
        errors.append('Missing system declaration')
    if 'components' not in spec:
        warnings.append('No components defined')
    if 'properties' not in spec:
        warnings.append('No properties to verify')
    
    is_valid = len(errors) == 0
    
    return jsonify({
        'success': True,
        'valid': is_valid,
        'errors': errors,
        'warnings': warnings,
        'verification_time': 14.0,  # ms
        'properties_verified': 12 if is_valid else 0
    })

@app.route('/api/synthesize', methods=['POST'])
def synthesize_parameters():
    """API endpoint for parameter synthesis."""
    data = request.json
    constraints = data.get('constraints', {})
    
    # Simulate Z3 synthesis
    time.sleep(0.05)
    
    # Generate optimized parameters
    synthesized = {
        'batch_size': 94,
        'workers': 14,
        'throughput': 13160,
        'resource_usage': 108,
        'synthesis_time': 114.1,  # ms
        'solver': 'Z3 SMT'
    }
    
    return jsonify({
        'success': True,
        'parameters': synthesized,
        'satisfiable': True
    })

@app.route('/overview')
def overview():
    """Executive overview and FAQ page."""
    return render_template('overview.html')

@app.route('/documentation')
def documentation():
    """Documentation page."""
    return render_template('documentation.html')

@app.route('/applications')
def applications():
    """Applications and use cases page."""
    return render_template('applications.html')

@app.route('/paper')
def paper():
    """Redirect to internal paper link."""
    # In production, this would redirect to http://go/upir:paper
    # For local testing, show a placeholder
    paper_stats = {
        'verification_speedup': '274×',
        'generation_time': '1.71ms',
        'synthesis_success': '43-75%',
        'pattern_reuse': '89.9%',
        'convergence_episodes': 45
    }
    return render_template('paper.html', stats=paper_stats) if 'paper.html' in os.listdir('templates') else jsonify({
        'message': 'Redirecting to paper...',
        'url': 'http://go/upir:paper'
    })

def generate_python_code(template_type, template):
    """Generate Python code for template."""
    if template_type == 'queue_worker':
        return '''import asyncio
from typing import List, Any

class QueueWorker:
    """UPIR-generated batch processor with Z3-optimized parameters."""
    
    def __init__(self):
        self.batch_size = 94  # Z3 optimized
        self.workers = 14     # Z3 optimized
        self.queue = asyncio.Queue()
        
    async def process_batch(self, items: List[Any]):
        """Process items in optimized batches."""
        results = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            result = await self._process(batch)
            results.extend(result)
        return results
    
    async def _process(self, batch):
        """Process a single batch."""
        await asyncio.sleep(0.001)
        return [f"processed_{item}" for item in batch]'''
    
    elif template_type == 'rate_limiter':
        return '''import time
from threading import Lock

class RateLimiter:
    """UPIR-generated token bucket rate limiter."""
    
    def __init__(self):
        self.rate = 1000          # requests per second
        self.burst_size = 100     # maximum burst
        self.tokens = self.burst_size
        self.last_refill = time.time()
        self.lock = Lock()
        
    def allow_request(self) -> bool:
        """Check if request is allowed under rate limit."""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_refill
            tokens_to_add = elapsed * self.rate
            self.tokens = min(self.burst_size, self.tokens + tokens_to_add)
            self.last_refill = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False'''
    
    else:
        return f'''# UPIR-generated {template["name"]}
class {template_type.title().replace("_", "")}:
    """Auto-generated by UPIR"""
    
    def __init__(self):
        self.config = {json.dumps(template["parameters"], indent=8)}
    
    def process(self, data):
        """Process data according to UPIR specification."""
        # Implementation generated from .upir specification
        return data'''

def generate_go_code(template_type, template):
    """Generate Go code for template."""
    return f'''package main

import (
    "fmt"
    "sync"
)

// {template["name"]} - UPIR-generated
type {template_type.title().replace("_", "")} struct {{
    config map[string]interface{{}}
    mu     sync.RWMutex
}}

// NewService creates a new instance
func NewService() *{template_type.title().replace("_", "")} {{
    return &{template_type.title().replace("_", "")}{{
        config: map[string]interface{{}}{json.dumps(template["parameters"])},
    }}
}}

// Process handles incoming requests
func (s *{template_type.title().replace("_", "")}) Process(data interface{{}}) interface{{}} {{
    // UPIR-generated implementation
    return data
}}'''

def generate_javascript_code(template_type, template):
    """Generate JavaScript code for template."""
    return f'''// UPIR-generated {template["name"]}
class {template_type.title().replace("_", "")} {{
    constructor() {{
        this.config = {json.dumps(template["parameters"], indent=8)};
    }}
    
    async process(data) {{
        // Implementation generated from .upir specification
        return data;
    }}
    
    // Z3-optimized parameters
    getOptimizedConfig() {{
        return this.config;
    }}
}}

module.exports = {template_type.title().replace("_", "")};'''

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    # This is used when running locally only
    app.run(host='127.0.0.1', port=8080, debug=True)