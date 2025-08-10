"""
Integration module for UPIR end-to-end workflows.

This module provides complete system integration for the UPIR lifecycle:
specification → verification → synthesis → deployment → learning → optimization

Author: subhadipmitra@google.com
"""

from .orchestrator import UPIROrchestrator, WorkflowConfig, DeploymentResult
from .pipeline import UPIRPipeline, PipelineStage, PipelineResult
from .monitoring import MetricsCollector, PerformanceMonitor
from .deployer import Deployer, DeploymentStrategy

__all__ = [
    'UPIROrchestrator',
    'WorkflowConfig', 
    'DeploymentResult',
    'UPIRPipeline',
    'PipelineStage',
    'PipelineResult',
    'MetricsCollector',
    'PerformanceMonitor',
    'Deployer',
    'DeploymentStrategy'
]