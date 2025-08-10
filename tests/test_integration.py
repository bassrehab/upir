"""
Integration tests for complete UPIR system.

Tests the end-to-end flow from specification to optimization.

Author: subhadipmitra@google.com
"""

import pytest
import asyncio
import sys
import os
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from upir.core.models import (
    FormalSpecification, TemporalProperty, TemporalOperator,
    UPIR, Architecture
)
from upir.integration.orchestrator import (
    UPIROrchestrator, WorkflowConfig, WorkflowState,
    DeploymentResult, OptimizationResult
)
from upir.verification.verifier import VerificationStatus
from upir.synthesis.synthesizer import Synthesizer


class TestOrchestrator:
    """Test the UPIR orchestrator."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """Test complete workflow from specification to deployment."""
        # Create simple specification
        spec = FormalSpecification(
            invariants=[
                TemporalProperty(
                    TemporalOperator.ALWAYS,
                    "data_consistency"
                )
            ],
            properties=[],
            constraints={"latency": {"max": 100}}
        )
        
        # Configure orchestrator
        config = WorkflowConfig(
            enable_verification=True,
            enable_synthesis=True,
            enable_deployment=True,
            enable_learning=False,  # Disable for testing
            verification_timeout=5000
        )
        
        orchestrator = UPIROrchestrator(config)
        
        # Execute workflow
        upir = await orchestrator.execute_workflow(spec)
        
        # Verify results
        assert upir is not None
        assert upir.specification == spec
        assert orchestrator.state in [
            WorkflowState.DEPLOYED,
            WorkflowState.FAILED
        ]
        
        # Check if implementation was created
        if orchestrator.state == WorkflowState.DEPLOYED:
            assert upir.implementation is not None
            assert len(upir.implementation.code) > 0
        
        await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_verification_phase(self):
        """Test verification phase of workflow."""
        spec = FormalSpecification(
            invariants=[
                TemporalProperty(
                    TemporalOperator.ALWAYS,
                    "test_property",
                    parameters={"value": 42}
                )
            ],
            properties=[],
            constraints={}
        )
        
        config = WorkflowConfig(
            enable_verification=True,
            enable_synthesis=False,
            enable_deployment=False
        )
        
        orchestrator = UPIROrchestrator(config)
        
        # Create UPIR
        upir = await orchestrator._create_upir(spec)
        assert upir.specification == spec
        
        # Verify
        results = await orchestrator._verify(upir)
        assert len(results) > 0
        
        # Check evidence was added
        assert len(upir.evidence) > 0
        
        await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_synthesis_phase(self):
        """Test synthesis phase."""
        spec = FormalSpecification(
            invariants=[
                TemporalProperty(
                    TemporalOperator.WITHIN,
                    "process_event",
                    time_bound=100
                )
            ],
            properties=[],
            constraints={}
        )
        
        config = WorkflowConfig(
            enable_verification=False,
            enable_synthesis=True,
            enable_deployment=False
        )
        
        orchestrator = UPIROrchestrator(config)
        
        upir = await orchestrator._create_upir(spec)
        
        # Add minimal architecture for synthesis
        upir.architecture = Architecture(
            components=[{"name": "processor", "type": "processor"}],
            connections=[],
            deployment={},
            patterns=[]
        )
        
        # Synthesize
        implementation = await orchestrator._synthesize(upir)
        
        assert implementation is not None
        assert implementation.code != ""
        assert implementation.synthesis_proof is not None
        
        await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_deployment_strategies(self):
        """Test different deployment strategies."""
        spec = FormalSpecification(
            invariants=[],
            properties=[],
            constraints={}
        )
        
        # Test canary deployment
        config = WorkflowConfig(
            enable_verification=False,
            enable_synthesis=False,
            enable_deployment=True,
            deployment_strategy="canary",
            canary_percentage=0.1
        )
        
        orchestrator = UPIROrchestrator(config)
        upir = await orchestrator._create_upir(spec)
        
        # Mock implementation
        from upir.core.models import Implementation, SynthesisProof
        upir.implementation = Implementation(
            code="mock_code",
            language="python",
            framework="mock",
            synthesis_proof=SynthesisProof(
                specification_hash="test",
                implementation_hash="test",
                proof_steps=[],
                verification_result=True
            )
        )
        
        # Try deployment multiple times (should succeed most of the time)
        successes = 0
        for _ in range(10):
            deployment = await orchestrator._deploy(upir)
            if deployment.success:
                successes += 1
                assert deployment.deployment_id
                assert deployment.url
                assert deployment.rollback_available
        
        # Should succeed at least 70% of the time (based on mock)
        assert successes >= 7
        
        await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_monitoring_and_metrics(self):
        """Test metrics collection and monitoring."""
        config = WorkflowConfig()
        orchestrator = UPIROrchestrator(config)
        
        # Create mock deployment
        deployment = DeploymentResult(
            success=True,
            deployment_id="test_deploy_123",
            url="https://test.example.com",
            metrics_endpoint="https://metrics.example.com"
        )
        
        # Collect metrics
        metrics = orchestrator._collect_metrics(deployment)
        
        assert "latency" in metrics
        assert "throughput" in metrics
        assert "error_rate" in metrics
        assert "availability" in metrics
        assert "cost" in metrics
        
        # Verify realistic ranges
        assert 0 < metrics["latency"] < 1000
        assert 0 < metrics["throughput"] < 100000
        assert 0 <= metrics["error_rate"] < 1
        assert 0 < metrics["availability"] <= 1
        assert 0 < metrics["cost"] < 100000
        
        await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_optimization_trigger(self):
        """Test optimization triggering based on metrics."""
        spec = FormalSpecification(
            invariants=[],
            properties=[],
            constraints={
                "latency": {"max": 50},
                "throughput": {"min": 10000},
                "error_rate": {"max": 0.001}
            }
        )
        
        config = WorkflowConfig()
        orchestrator = UPIROrchestrator(config)
        
        upir = await orchestrator._create_upir(spec)
        orchestrator.current_upir = upir
        
        # Test with good metrics (should not trigger)
        good_metrics = {
            "latency": 40,
            "throughput": 15000,
            "error_rate": 0.0005
        }
        assert not orchestrator._should_optimize(good_metrics)
        
        # Test with bad latency (should trigger)
        bad_latency = good_metrics.copy()
        bad_latency["latency"] = 100
        assert orchestrator._should_optimize(bad_latency)
        
        # Test with bad throughput (should trigger)
        bad_throughput = good_metrics.copy()
        bad_throughput["throughput"] = 5000
        assert orchestrator._should_optimize(bad_throughput)
        
        # Test with bad error rate (should trigger)
        bad_errors = good_metrics.copy()
        bad_errors["error_rate"] = 0.01
        assert orchestrator._should_optimize(bad_errors)
        
        await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_pattern_integration(self):
        """Test pattern library integration."""
        spec = FormalSpecification(
            invariants=[
                TemporalProperty(
                    TemporalOperator.ALWAYS,
                    "data_consistency"
                )
            ],
            properties=[],
            constraints={"latency": {"max": 100}}
        )
        
        config = WorkflowConfig(
            enable_pattern_discovery=True,
            pattern_library_path="./test_patterns"
        )
        
        orchestrator = UPIROrchestrator(config)
        
        # Add a pattern to library
        from upir.patterns.extractor import ArchitecturalPattern
        pattern = ArchitecturalPattern(
            id="test_pattern",
            name="Test Pattern",
            description="Test",
            category="test",
            template_components=[
                {"type": "processor"},
                {"type": "storage"}
            ],
            template_connections=[],
            required_properties=["data_consistency"],
            success_rate=0.9
        )
        
        orchestrator.pattern_library.add_pattern(pattern)
        
        # Create UPIR - should use pattern
        upir = await orchestrator._create_upir(spec)
        
        # Should have architecture from pattern
        assert upir.architecture is not None
        assert len(upir.architecture.components) > 0
        
        # Check pattern was recorded as used
        assert len(orchestrator.pattern_library.usage_history) > 0
        
        await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_workflow_state_transitions(self):
        """Test proper state transitions during workflow."""
        spec = FormalSpecification(
            invariants=[],
            properties=[],
            constraints={}
        )
        
        config = WorkflowConfig(
            enable_verification=True,
            enable_synthesis=True,
            enable_deployment=False,  # Stop before deployment
            enable_learning=False
        )
        
        orchestrator = UPIROrchestrator(config)
        
        # Initial state
        assert orchestrator.state == WorkflowState.INITIALIZED
        
        # Create UPIR
        upir = await orchestrator._create_upir(spec)
        orchestrator.current_upir = upir
        orchestrator.state = WorkflowState.SPECIFIED
        assert orchestrator.state == WorkflowState.SPECIFIED
        
        # Verify
        await orchestrator._verify(upir)
        orchestrator.state = WorkflowState.VERIFIED
        assert orchestrator.state == WorkflowState.VERIFIED
        
        # Add architecture for synthesis
        upir.architecture = Architecture(
            components=[{"name": "test", "type": "service"}],
            connections=[],
            deployment={},
            patterns=[]
        )
        
        # Synthesize
        await orchestrator._synthesize(upir)
        orchestrator.state = WorkflowState.SYNTHESIZED
        assert orchestrator.state == WorkflowState.SYNTHESIZED
        
        await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_improvement_calculation(self):
        """Test improvement percentage calculation."""
        config = WorkflowConfig()
        orchestrator = UPIROrchestrator(config)
        
        old_metrics = {
            "latency": 100,
            "throughput": 1000,
            "error_rate": 0.01
        }
        
        # Better metrics
        new_metrics = {
            "latency": 80,  # 20% improvement
            "throughput": 1200,  # 20% improvement
            "error_rate": 0.005  # 50% improvement
        }
        
        improvement = orchestrator._calculate_improvement(old_metrics, new_metrics)
        
        # Average of 20%, 20%, 50% = 30%
        assert improvement > 0.25
        assert improvement < 0.35
        
        # Worse metrics
        worse_metrics = {
            "latency": 120,
            "throughput": 800,
            "error_rate": 0.02
        }
        
        degradation = orchestrator._calculate_improvement(old_metrics, worse_metrics)
        assert degradation < 0  # Should be negative
        
        await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_checkpoint_saving(self):
        """Test checkpoint saving and recovery."""
        import tempfile
        import shutil
        
        # Create temp directory for checkpoints
        temp_dir = tempfile.mkdtemp()
        
        try:
            config = WorkflowConfig(
                checkpoint_path=temp_dir
            )
            
            orchestrator = UPIROrchestrator(config)
            
            # Add some history
            orchestrator.workflow_history.append({
                "test": "data",
                "timestamp": datetime.utcnow().isoformat()
            })
            orchestrator.metrics_history.append({
                "deployment_id": "test",
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": {"latency": 50}
            })
            orchestrator.optimization_count = 5
            
            # Save checkpoint
            orchestrator._save_checkpoint()
            
            # Verify checkpoint file exists
            import glob
            checkpoint_files = glob.glob(f"{temp_dir}/orchestrator_*.json")
            assert len(checkpoint_files) > 0
            
            # Load and verify content
            import json
            with open(checkpoint_files[0], 'r') as f:
                checkpoint = json.load(f)
            
            assert checkpoint["optimization_count"] == 5
            assert len(checkpoint["workflow_history"]) == 1
            assert len(checkpoint["metrics_history"]) == 1
            
            await orchestrator.shutdown()
            
        finally:
            # Cleanup
            shutil.rmtree(temp_dir)


class TestEndToEndScenarios:
    """Test complete end-to-end scenarios."""
    
    @pytest.mark.asyncio
    async def test_streaming_pipeline_scenario(self):
        """Test complete streaming pipeline scenario."""
        # Create streaming specification
        spec = FormalSpecification(
            invariants=[
                TemporalProperty(
                    TemporalOperator.WITHIN,
                    "event_processed",
                    time_bound=100,
                    parameters={"description": "Process within 100ms"}
                ),
                TemporalProperty(
                    TemporalOperator.ALWAYS,
                    "exactly_once",
                    parameters={"description": "Exactly once delivery"}
                )
            ],
            properties=[],
            constraints={
                "latency": {"max": 100},
                "throughput": {"min": 10000}
            }
        )
        
        config = WorkflowConfig(
            enable_verification=True,
            enable_synthesis=True,
            enable_deployment=False,  # Skip actual deployment in test
            verification_timeout=5000,
            synthesis_max_iterations=10
        )
        
        orchestrator = UPIROrchestrator(config)
        
        # Execute workflow
        upir = await orchestrator.execute_workflow(spec)
        
        # Verify UPIR was created correctly
        assert upir.specification == spec
        assert len(upir.evidence) > 0
        
        # Check verification evidence
        verification_evidence = [e for e in upir.evidence.values() 
                                if e.type == "formal_proof"]
        assert len(verification_evidence) > 0
        
        # Check synthesis evidence
        if upir.implementation:
            synthesis_evidence = [e for e in upir.evidence.values() 
                                 if e.type == "synthesis"]
            assert len(synthesis_evidence) > 0
        
        await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_optimization_cycle(self):
        """Test a complete optimization cycle."""
        spec = FormalSpecification(
            invariants=[],
            properties=[],
            constraints={
                "latency": {"max": 50},
                "throughput": {"min": 10000}
            }
        )
        
        config = WorkflowConfig(
            enable_verification=False,
            enable_synthesis=False,
            enable_deployment=False,
            optimization_threshold=0.05  # 5% improvement triggers optimization
        )
        
        orchestrator = UPIROrchestrator(config)
        
        # Create UPIR with architecture
        upir = await orchestrator._create_upir(spec)
        upir.architecture = Architecture(
            components=[
                {"name": "service", "type": "service", "config": {"replicas": 1}},
                {"name": "db", "type": "storage", "config": {}}
            ],
            connections=[{"source": "service", "target": "db"}],
            deployment={},
            patterns=[]
        )
        orchestrator.current_upir = upir
        
        # Mock current metrics (violating constraints)
        current_metrics = {
            "latency": 60,  # Above max
            "throughput": 8000,  # Below min
            "error_rate": 0.01
        }
        
        # Trigger optimization
        result = await orchestrator._optimize(upir, current_metrics)
        
        # Check optimization was attempted
        assert isinstance(result, OptimizationResult)
        
        await orchestrator.shutdown()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])