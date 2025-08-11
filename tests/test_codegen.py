"""
Tests for template-based code generation.
"""

import pytest
from upir.codegen import (
    CodeGenerator, 
    QueueWorkerTemplate,
    RateLimiterTemplate,
    CircuitBreakerTemplate,
    RetryTemplate,
    CacheTemplate,
    LoadBalancerTemplate
)


class TestCodeGenerator:
    """Test the main code generator."""
    
    def setup_method(self):
        self.generator = CodeGenerator()
        # Register templates
        self.generator.register_template(QueueWorkerTemplate())
        self.generator.register_template(RateLimiterTemplate())
        self.generator.register_template(CircuitBreakerTemplate())
        
    def test_generate_queue_worker(self):
        """Test generating a queue worker."""
        spec = {
            'pattern': 'queue_worker',
            'requirements': {
                'batch_size': 50,
                'workers': 8
            },
            'properties': ['no_data_loss', 'bounded_latency']
        }
        
        result = self.generator.generate_from_spec(spec, 'python')
        
        assert result.language == 'python'
        assert 'class QueueWorker' in result.code
        assert 'batch_size = 50' in result.code
        assert 'workers = 8' in result.code
        assert 'no_data_loss' in result.verified_properties
        
    def test_generate_rate_limiter(self):
        """Test generating a rate limiter."""
        spec = {
            'pattern': 'rate_limiter',
            'requirements': {
                'requests_per_second': 1000
            }
        }
        
        result = self.generator.generate_from_spec(spec, 'python')
        
        assert 'class RateLimiter' in result.code
        assert 'rate = 1000' in result.code
        assert result.synthesized_params['requests_per_second'] == 1000
        
    def test_generate_multiple_languages(self):
        """Test generating code in different languages."""
        spec = {
            'pattern': 'queue_worker',
            'requirements': {}
        }
        
        # Python
        py_result = self.generator.generate_from_spec(spec, 'python')
        assert 'class QueueWorker' in py_result.code
        assert 'def process_batch' in py_result.code
        
        # Go
        go_result = self.generator.generate_from_spec(spec, 'go')
        assert 'type QueueWorker struct' in go_result.code
        assert 'func NewQueueWorker' in go_result.code
        
        # JavaScript
        js_result = self.generator.generate_from_spec(spec, 'javascript')
        assert 'class QueueWorker' in js_result.code
        assert 'async processBatch' in js_result.code
        
    def test_system_generation(self):
        """Test generating an entire system."""
        components = [
            {'pattern': 'rate_limiter', 'requirements': {'requests_per_second': 100}},
            {'pattern': 'queue_worker', 'requirements': {'batch_size': 10}},
            {'pattern': 'circuit_breaker', 'requirements': {'failure_threshold': 5}}
        ]
        
        results = self.generator.generate_system(components, 'python')
        
        assert len(results) == 3
        assert 'RateLimiter' in results[0].code
        assert 'QueueWorker' in results[1].code
        assert 'CircuitBreaker' in results[2].code


class TestTemplates:
    """Test individual templates."""
    
    def test_queue_worker_synthesis(self):
        """Test parameter synthesis for queue worker."""
        template = QueueWorkerTemplate()
        
        # Synthesize with constraints
        params = template.synthesize_parameters({
            'batch_size': 25,
            'workers': 4
        })
        
        assert params['batch_size'] == 25
        assert params['workers'] == 4
        assert params['timeout_ms'] >= params['batch_size'] * 10
        
    def test_rate_limiter_synthesis(self):
        """Test parameter synthesis for rate limiter."""
        template = RateLimiterTemplate()
        
        params = template.synthesize_parameters({
            'requests_per_second': 500
        })
        
        assert params['requests_per_second'] == 500
        assert params['burst_size'] >= 50  # At least 1/10 of rate
        assert params['window_ms'] >= 2  # At least 1000/500
        
    def test_circuit_breaker_synthesis(self):
        """Test parameter synthesis for circuit breaker."""
        template = CircuitBreakerTemplate()
        
        params = template.synthesize_parameters({
            'failure_threshold': 10
        })
        
        assert params['failure_threshold'] == 10
        assert params['recovery_timeout_ms'] >= 1000  # At least 10 * 100
        
    def test_retry_template(self):
        """Test retry template generation."""
        template = RetryTemplate()
        
        params = template.synthesize_parameters({
            'max_attempts': 5
        })
        
        python_code = template.generate('python', params)
        assert 'def retry_with_backoff' in python_code
        assert 'max_attempts = 5' in python_code
        assert 'exponential backoff' in python_code.lower() or 'backoff' in python_code.lower()
        
    def test_cache_template(self):
        """Test cache template generation."""
        template = CacheTemplate()
        
        params = template.synthesize_parameters({
            'max_size': 1000,
            'ttl_seconds': 60
        })
        
        python_code = template.generate('python', params)
        assert 'Cache' in python_code
        assert 'max_size = 1000' in python_code
        assert 'ttl_seconds = 60' in python_code
        
    def test_load_balancer_template(self):
        """Test load balancer template generation."""
        template = LoadBalancerTemplate()
        
        params = template.synthesize_parameters({
            'algorithm': 'round_robin'
        })
        
        python_code = template.generate('python', params)
        assert 'class LoadBalancer' in python_code
        assert 'select_backend' in python_code
        assert 'round robin' in python_code.lower() or 'current_index' in python_code


class TestConstraintSatisfaction:
    """Test that synthesized parameters satisfy constraints."""
    
    def test_queue_worker_constraints(self):
        """Test queue worker constraints are satisfied."""
        template = QueueWorkerTemplate()
        
        # Try various requirements
        test_cases = [
            {'batch_size': 100, 'workers': 10},
            {'batch_size': 1, 'workers': 100},
            {'batch_size': 50, 'workers': 20}
        ]
        
        for requirements in test_cases:
            params = template.synthesize_parameters(requirements)
            
            # Check constraints
            assert params['batch_size'] * params['workers'] <= 1000
            assert params['timeout_ms'] >= params['batch_size'] * 10
            
    def test_rate_limiter_constraints(self):
        """Test rate limiter constraints are satisfied."""
        template = RateLimiterTemplate()
        
        test_cases = [
            {'requests_per_second': 10},
            {'requests_per_second': 1000},
            {'requests_per_second': 50000}
        ]
        
        for requirements in test_cases:
            params = template.synthesize_parameters(requirements)
            
            # Check constraints
            assert params['burst_size'] >= params['requests_per_second'] / 10
            assert params['window_ms'] >= 1000 / params['requests_per_second']
            
    def test_unsatisfiable_constraints(self):
        """Test handling of unsatisfiable constraints."""
        template = QueueWorkerTemplate()
        
        # These constraints can't be satisfied
        # batch_size * workers must be <= 1000, but we're requiring both to be high
        with pytest.raises(ValueError, match="Cannot synthesize"):
            template.synthesize_parameters({
                'batch_size': 500,
                'workers': 50  # 500 * 50 = 25000 > 1000
            })