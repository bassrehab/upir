"""
Code generation module for UPIR.

Provides template-based code generation for common distributed system patterns.
"""

from .generator import CodeGenerator, Template
from .templates import (
    QueueWorkerTemplate,
    RateLimiterTemplate,
    CircuitBreakerTemplate,
    RetryTemplate,
    CacheTemplate,
    LoadBalancerTemplate
)

__all__ = [
    'CodeGenerator',
    'Template',
    'QueueWorkerTemplate',
    'RateLimiterTemplate', 
    'CircuitBreakerTemplate',
    'RetryTemplate',
    'CacheTemplate',
    'LoadBalancerTemplate'
]