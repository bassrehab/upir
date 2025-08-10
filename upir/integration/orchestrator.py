"""
UPIR Orchestrator - Complete End-to-End System Integration

This orchestrator manages the entire UPIR lifecycle from specification to
continuous optimization. It coordinates all components and ensures smooth
transitions between phases.

The orchestrator implements a closed-loop system:
1. Accept formal specifications
2. Verify properties mathematically
3. Synthesize implementations
4. Deploy to production
5. Learn from metrics
6. Optimize architecture
7. Loop back to verification

Author: subhadipmitra@google.com
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import hashlib
import numpy as np

from ..core.models import (
    UPIR, FormalSpecification, TemporalProperty, 
    Evidence, ReasoningNode, Implementation
)
from ..verification.verifier import Verifier, VerificationResult
from ..synthesis.synthesizer import Synthesizer
from ..learning.learner import ArchitectureLearner
from ..patterns.library import PatternLibrary
from ..patterns.extractor import FeatureExtractor, PatternClusterer

logger = logging.getLogger(__name__)


class WorkflowState(Enum):
    """States in the UPIR workflow."""
    INITIALIZED = "initialized"
    SPECIFIED = "specified"
    VERIFIED = "verified"
    SYNTHESIZED = "synthesized"
    DEPLOYED = "deployed"
    MONITORING = "monitoring"
    LEARNING = "learning"
    OPTIMIZING = "optimizing"
    FAILED = "failed"


@dataclass
class WorkflowConfig:
    """Configuration for UPIR workflow."""
    enable_verification: bool = True
    enable_synthesis: bool = True
    enable_deployment: bool = True
    enable_learning: bool = True
    enable_pattern_discovery: bool = True
    
    verification_timeout: int = 30000  # ms
    synthesis_max_iterations: int = 100
    learning_interval: int = 300  # seconds
    optimization_threshold: float = 0.1  # Trigger optimization if improvement > 10%
    
    deployment_strategy: str = "canary"  # "canary", "blue_green", "rolling"
    canary_percentage: float = 0.1
    
    monitoring_metrics: List[str] = field(default_factory=lambda: [
        "latency", "throughput", "error_rate", "cost", "availability"
    ])
    
    pattern_library_path: str = "./pattern_library"
    checkpoint_path: str = "./checkpoints"


@dataclass
class DeploymentResult:
    """Result of deployment operation."""
    success: bool
    deployment_id: str
    url: Optional[str] = None
    metrics_endpoint: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    rollback_available: bool = False


@dataclass
class OptimizationResult:
    """Result of optimization cycle."""
    improved: bool
    old_metrics: Dict[str, float]
    new_metrics: Dict[str, float]
    changes_applied: List[str]
    improvement_percentage: float


class UPIROrchestrator:
    """
    Master orchestrator for the complete UPIR system.
    
    This is the conductor of our symphony - coordinating all components
    to deliver a seamless experience from specification to optimization.
    """
    
    def __init__(self, config: WorkflowConfig = None):
        """Initialize orchestrator with configuration."""
        self.config = config or WorkflowConfig()
        self.state = WorkflowState.INITIALIZED
        
        # Core components
        self.verifier = Verifier(timeout=self.config.verification_timeout)
        self.synthesizer = Synthesizer()
        self.learner = ArchitectureLearner()
        self.pattern_library = PatternLibrary(self.config.pattern_library_path)
        
        # Workflow tracking
        self.current_upir: Optional[UPIR] = None
        self.workflow_history: List[Dict[str, Any]] = []
        self.active_deployments: Dict[str, DeploymentResult] = {}
        self.metrics_history: List[Dict[str, Any]] = []
        
        # Optimization tracking
        self.optimization_count = 0
        self.last_optimization = None
        
        logger.info("UPIR Orchestrator initialized")
    
    async def execute_workflow(self, specification: FormalSpecification) -> UPIR:
        """
        Execute complete UPIR workflow from specification to optimization.
        
        This is the main entry point for the end-to-end flow.
        """
        logger.info("Starting UPIR workflow execution")
        start_time = time.time()
        
        try:
            # Step 1: Create UPIR from specification
            upir = await self._create_upir(specification)
            self.current_upir = upir
            self.state = WorkflowState.SPECIFIED
            
            # Step 2: Verify formal properties
            if self.config.enable_verification:
                verification_results = await self._verify(upir)
                if not all(r.verified for r in verification_results):
                    logger.warning("Some properties could not be verified")
                    # Deciding whether to continue despite verification failures
                    # In production, this would be configurable
                self.state = WorkflowState.VERIFIED
            
            # Step 3: Synthesize implementation
            if self.config.enable_synthesis:
                implementation = await self._synthesize(upir)
                upir.implementation = implementation
                self.state = WorkflowState.SYNTHESIZED
            
            # Step 4: Deploy to production
            if self.config.enable_deployment:
                deployment = await self._deploy(upir)
                if deployment.success:
                    self.active_deployments[deployment.deployment_id] = deployment
                    self.state = WorkflowState.DEPLOYED
                    
                    # Start monitoring
                    asyncio.create_task(self._monitor_deployment(
                        upir, deployment
                    ))
                else:
                    logger.error(f"Deployment failed: {deployment.errors}")
                    self.state = WorkflowState.FAILED
                    return upir
            
            # Step 5: Start learning loop
            if self.config.enable_learning:
                asyncio.create_task(self._learning_loop(upir))
                self.state = WorkflowState.LEARNING
            
            # Step 6: Pattern discovery
            if self.config.enable_pattern_discovery:
                await self._discover_patterns(upir)
            
            # Record workflow completion
            self.workflow_history.append({
                "upir_id": upir.id,
                "specification_hash": upir.specification.hash() if upir.specification else None,
                "start_time": start_time,
                "end_time": time.time(),
                "state": self.state.value,
                "success": True
            })
            
            logger.info(f"Workflow completed in {time.time() - start_time:.2f} seconds")
            return upir
            
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            self.state = WorkflowState.FAILED
            self.workflow_history.append({
                "error": str(e),
                "state": self.state.value,
                "success": False
            })
            raise
    
    async def _create_upir(self, specification: FormalSpecification) -> UPIR:
        """Create UPIR instance from specification."""
        logger.info("Creating UPIR from specification")
        
        upir = UPIR(
            name=f"System_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            description="Automatically generated from formal specification"
        )
        upir.specification = specification
        
        # Check pattern library for matching patterns
        recommendations = self.pattern_library.recommend(upir, top_k=3)
        
        if recommendations:
            logger.info(f"Found {len(recommendations)} matching patterns")
            
            # Use best matching pattern as starting point
            best_pattern, score = recommendations[0]
            logger.info(f"Using pattern {best_pattern.name} (score: {score:.2f})")
            
            # Instantiate pattern as initial architecture
            upir.architecture = best_pattern.instantiate({
                "instance_id": upir.id[:8]
            })
            
            # Record pattern usage
            self.pattern_library.record_usage(
                pattern_id=best_pattern.id,
                upir_id=upir.id,
                success=True,  # Will be updated based on actual performance
                performance={},
                feedback="Pattern selected for new system"
            )
        else:
            logger.info("No matching patterns found, will synthesize from scratch")
        
        return upir
    
    async def _verify(self, upir: UPIR) -> List[VerificationResult]:
        """Verify formal properties of UPIR."""
        logger.info("Verifying formal properties")
        
        results = self.verifier.verify_specification(upir)
        
        # Add verification evidence
        verified_props = [r.property.predicate for r in results if r.verified]
        failed_props = [r.property.predicate for r in results if not r.verified]
        
        evidence = Evidence(
            source="formal_verification",
            type="formal_proof",
            data={
                "verified": verified_props,
                "failed": failed_props,
                "timestamp": datetime.utcnow().isoformat()
            },
            confidence=1.0 if not failed_props else 0.5
        )
        upir.add_evidence(evidence)
        
        return results
    
    async def _synthesize(self, upir: UPIR) -> Implementation:
        """Synthesize implementation from UPIR."""
        logger.info("Synthesizing implementation")
        
        # If we already have an architecture from patterns, synthesize from it
        if upir.architecture:
            # Create sketch from existing architecture
            sketch = self.synthesizer.create_sketch_from_architecture(upir.architecture)
        else:
            # Create sketch from specification
            sketch = self.synthesizer.create_sketch(upir.specification)
        
        # Run CEGIS synthesis
        implementation = self.synthesizer.synthesize(
            specification=upir.specification,
            sketch=sketch,
            max_iterations=self.config.synthesis_max_iterations
        )
        
        if implementation:
            logger.info("Synthesis successful")
            
            # Add synthesis evidence
            evidence = Evidence(
                source="synthesis",
                type="synthesis",
                data={
                    "method": "CEGIS",
                    "iterations": len(implementation.synthesis_proof.proof_steps),
                    "verified": implementation.synthesis_proof.verification_result
                },
                confidence=0.9
            )
            upir.add_evidence(evidence)
        else:
            logger.error("Synthesis failed")
            raise RuntimeError("Could not synthesize implementation")
        
        return implementation
    
    async def _deploy(self, upir: UPIR) -> DeploymentResult:
        """Deploy UPIR implementation to production."""
        logger.info(f"Deploying using {self.config.deployment_strategy} strategy")
        
        # In a real system, this would interact with cloud APIs
        # Here we simulate deployment
        
        deployment_id = hashlib.md5(
            f"{upir.id}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:12]
        
        # Simulate deployment based on strategy
        if self.config.deployment_strategy == "canary":
            # Deploy to small percentage of traffic
            logger.info(f"Deploying canary to {self.config.canary_percentage*100}% of traffic")
            
            # Simulate deployment success (90% success rate)
            success = np.random.random() > 0.1
            
            if success:
                return DeploymentResult(
                    success=True,
                    deployment_id=deployment_id,
                    url=f"https://upir-{deployment_id}.upir-dev.app",
                    metrics_endpoint=f"https://metrics.upir-dev.app/{deployment_id}",
                    rollback_available=True
                )
        
        elif self.config.deployment_strategy == "blue_green":
            # Deploy to parallel environment
            logger.info("Deploying to green environment")
            success = np.random.random() > 0.05  # 95% success rate
            
            if success:
                return DeploymentResult(
                    success=True,
                    deployment_id=deployment_id,
                    url=f"https://green-{deployment_id}.upir-dev.app",
                    metrics_endpoint=f"https://metrics.upir-dev.app/{deployment_id}",
                    rollback_available=True
                )
        
        # Default or failed deployment
        return DeploymentResult(
            success=False,
            deployment_id=deployment_id,
            errors=["Deployment simulation failed"]
        )
    
    async def _monitor_deployment(self, upir: UPIR, deployment: DeploymentResult):
        """Monitor deployed system and collect metrics."""
        logger.info(f"Starting monitoring for deployment {deployment.deployment_id}")
        
        while deployment.deployment_id in self.active_deployments:
            # Simulate collecting metrics
            metrics = self._collect_metrics(deployment)
            
            # Store metrics
            self.metrics_history.append({
                "deployment_id": deployment.deployment_id,
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": metrics
            })
            
            # Add metrics as evidence
            evidence = Evidence(
                source=f"production_{deployment.deployment_id}",
                type="production",
                data=metrics,
                confidence=0.95
            )
            upir.add_evidence(evidence)
            
            # Check if optimization is needed
            if self._should_optimize(metrics):
                logger.info("Optimization triggered based on metrics")
                asyncio.create_task(self._optimize(upir, metrics))
            
            # Wait before next collection
            await asyncio.sleep(30)  # Collect every 30 seconds
    
    def _collect_metrics(self, deployment: DeploymentResult) -> Dict[str, float]:
        """Collect metrics from deployed system."""
        # In real system, this would query monitoring APIs
        # Here we simulate realistic metrics with some variation
        
        base_metrics = {
            "latency": 50 + np.random.normal(0, 10),  # 50ms ± 10ms
            "throughput": 10000 + np.random.normal(0, 1000),  # 10k ± 1k events/sec
            "error_rate": max(0, 0.001 + np.random.normal(0, 0.0005)),  # 0.1% ± 0.05%
            "availability": min(1.0, 0.999 + np.random.normal(0, 0.0005)),  # 99.9% ± 0.05%
            "cost": 4000 + np.random.normal(0, 200)  # $4000 ± $200
        }
        
        # Add some realistic patterns
        hour_of_day = datetime.utcnow().hour
        
        # Higher load during business hours
        if 9 <= hour_of_day <= 17:
            base_metrics["throughput"] *= 1.5
            base_metrics["latency"] *= 1.2
        
        # Occasional spikes
        if np.random.random() < 0.05:  # 5% chance of spike
            base_metrics["latency"] *= 2
            base_metrics["error_rate"] *= 3
        
        return base_metrics
    
    async def _learning_loop(self, upir: UPIR):
        """Continuous learning loop for architecture improvement."""
        logger.info("Starting learning loop")
        
        while self.state in [WorkflowState.DEPLOYED, WorkflowState.LEARNING, WorkflowState.OPTIMIZING]:
            # Wait for learning interval
            await asyncio.sleep(self.config.learning_interval)
            
            # Collect recent metrics
            recent_metrics = self._get_recent_metrics()
            
            if recent_metrics:
                # Learn from metrics
                self.learner.learn_from_metrics(upir, recent_metrics)
                
                # Get optimization suggestions
                suggestions = self.learner.suggest_optimization(upir)
                
                if suggestions:
                    logger.info(f"Learner suggested {len(suggestions)} optimizations")
                    
                    # Evaluate suggestions
                    for suggestion in suggestions[:3]:  # Consider top 3
                        expected_improvement = suggestion.expected_improvement
                        
                        if expected_improvement > self.config.optimization_threshold:
                            logger.info(f"Applying optimization: {suggestion.description}")
                            
                            # Apply optimization
                            await self._apply_optimization(upir, suggestion)
                            break
    
    def _should_optimize(self, metrics: Dict[str, float]) -> bool:
        """Determine if optimization is needed based on metrics."""
        # Check if we've optimized recently
        if self.last_optimization:
            time_since_last = datetime.utcnow() - self.last_optimization
            if time_since_last < timedelta(minutes=30):
                return False  # Don't optimize too frequently
        
        # Check if metrics violate constraints
        if self.current_upir and self.current_upir.specification:
            constraints = self.current_upir.specification.constraints
            
            # Check latency
            if "latency" in constraints and "latency" in metrics:
                if metrics["latency"] > constraints["latency"].get("max", float('inf')):
                    logger.warning(f"Latency {metrics['latency']} exceeds max {constraints['latency']['max']}")
                    return True
            
            # Check throughput
            if "throughput" in constraints and "throughput" in metrics:
                if metrics["throughput"] < constraints["throughput"].get("min", 0):
                    logger.warning(f"Throughput {metrics['throughput']} below min {constraints['throughput']['min']}")
                    return True
            
            # Check error rate
            if "error_rate" in constraints and "error_rate" in metrics:
                if metrics["error_rate"] > constraints["error_rate"].get("max", 1.0):
                    logger.warning(f"Error rate {metrics['error_rate']} exceeds max {constraints['error_rate']['max']}")
                    return True
        
        return False
    
    async def _optimize(self, upir: UPIR, current_metrics: Dict[str, float]) -> OptimizationResult:
        """Optimize architecture based on current metrics."""
        logger.info("Starting optimization cycle")
        self.state = WorkflowState.OPTIMIZING
        
        # Get optimization suggestions from learner
        suggestions = self.learner.suggest_optimization(upir)
        
        if not suggestions:
            logger.info("No optimizations suggested")
            return OptimizationResult(
                improved=False,
                old_metrics=current_metrics,
                new_metrics=current_metrics,
                changes_applied=[],
                improvement_percentage=0.0
            )
        
        # Apply best suggestion
        best_suggestion = suggestions[0]
        logger.info(f"Applying optimization: {best_suggestion.description}")
        
        # Create new version with optimization
        optimized_upir = await self._apply_optimization(upir, best_suggestion)
        
        # Verify optimized version
        if self.config.enable_verification:
            verification_results = await self._verify(optimized_upir)
            if not all(r.verified for r in verification_results):
                logger.warning("Optimized version fails verification, rolling back")
                return OptimizationResult(
                    improved=False,
                    old_metrics=current_metrics,
                    new_metrics=current_metrics,
                    changes_applied=[],
                    improvement_percentage=0.0
                )
        
        # Deploy optimized version (canary)
        if self.config.enable_deployment:
            deployment = await self._deploy(optimized_upir)
            
            if deployment.success:
                # Monitor new deployment for comparison
                await asyncio.sleep(60)  # Wait for metrics
                
                new_metrics = self._collect_metrics(deployment)
                
                # Calculate improvement
                improvement = self._calculate_improvement(current_metrics, new_metrics)
                
                if improvement > 0:
                    logger.info(f"Optimization successful: {improvement:.1%} improvement")
                    
                    # Promote canary to full deployment
                    self.current_upir = optimized_upir
                    self.optimization_count += 1
                    self.last_optimization = datetime.utcnow()
                    
                    return OptimizationResult(
                        improved=True,
                        old_metrics=current_metrics,
                        new_metrics=new_metrics,
                        changes_applied=[best_suggestion.description],
                        improvement_percentage=improvement
                    )
                else:
                    logger.info("Optimization did not improve metrics, rolling back")
                    # Rollback deployment
                    await self._rollback_deployment(deployment)
        
        return OptimizationResult(
            improved=False,
            old_metrics=current_metrics,
            new_metrics=current_metrics,
            changes_applied=[],
            improvement_percentage=0.0
        )
    
    async def _apply_optimization(self, upir: UPIR, suggestion: Any) -> UPIR:
        """Apply optimization suggestion to create new UPIR version."""
        # Clone current UPIR
        import copy
        optimized = copy.deepcopy(upir)
        optimized.id = f"{upir.id}_opt_{self.optimization_count}"
        
        # Apply changes based on suggestion type
        if hasattr(suggestion, 'action_type'):
            if suggestion.action_type == "scale":
                # Scale components
                for comp in optimized.architecture.components:
                    if "config" in comp and "replicas" in comp["config"]:
                        comp["config"]["replicas"] = int(comp["config"]["replicas"] * 1.5)
            
            elif suggestion.action_type == "add_cache":
                # Add caching layer
                cache_comp = {
                    "name": "optimization_cache",
                    "type": "cache",
                    "config": {"size_mb": 1024, "ttl": 300}
                }
                optimized.architecture.components.append(cache_comp)
            
            elif suggestion.action_type == "optimize_query":
                # Add query optimization hints
                for comp in optimized.architecture.components:
                    if comp.get("type") == "storage":
                        if "config" not in comp:
                            comp["config"] = {}
                        comp["config"]["query_cache"] = True
                        comp["config"]["indexes"] = ["optimized"]
        
        # Re-synthesize if needed
        if self.config.enable_synthesis:
            optimized.implementation = await self._synthesize(optimized)
        
        return optimized
    
    def _calculate_improvement(self, old_metrics: Dict[str, float], 
                              new_metrics: Dict[str, float]) -> float:
        """Calculate overall improvement percentage."""
        improvements = []
        
        # Latency improvement (lower is better)
        if "latency" in old_metrics and "latency" in new_metrics:
            latency_imp = (old_metrics["latency"] - new_metrics["latency"]) / old_metrics["latency"]
            improvements.append(latency_imp)
        
        # Throughput improvement (higher is better)
        if "throughput" in old_metrics and "throughput" in new_metrics:
            throughput_imp = (new_metrics["throughput"] - old_metrics["throughput"]) / old_metrics["throughput"]
            improvements.append(throughput_imp)
        
        # Error rate improvement (lower is better)
        if "error_rate" in old_metrics and "error_rate" in new_metrics:
            if old_metrics["error_rate"] > 0:
                error_imp = (old_metrics["error_rate"] - new_metrics["error_rate"]) / old_metrics["error_rate"]
                improvements.append(error_imp)
        
        return np.mean(improvements) if improvements else 0.0
    
    async def _rollback_deployment(self, deployment: DeploymentResult):
        """Rollback a deployment."""
        logger.info(f"Rolling back deployment {deployment.deployment_id}")
        
        if deployment.rollback_available:
            # Remove from active deployments
            if deployment.deployment_id in self.active_deployments:
                del self.active_deployments[deployment.deployment_id]
            
            logger.info("Rollback completed")
        else:
            logger.warning("Rollback not available for this deployment")
    
    def _get_recent_metrics(self, window_minutes: int = 30) -> Dict[str, float]:
        """Get average of recent metrics."""
        if not self.metrics_history:
            return {}
        
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
        recent = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m["timestamp"]) > cutoff
        ]
        
        if not recent:
            return {}
        
        # Average all metrics
        aggregated = {}
        for entry in recent:
            for key, value in entry["metrics"].items():
                if key not in aggregated:
                    aggregated[key] = []
                aggregated[key].append(value)
        
        return {k: np.mean(v) for k, v in aggregated.items()}
    
    async def _discover_patterns(self, upir: UPIR):
        """Discover and register new patterns."""
        logger.info("Discovering patterns from successful deployments")
        
        # Collect successful UPIRs
        successful_upirs = [upir]  # Start with current
        
        # In a real system, we'd query a database of successful deployments
        # Here we just work with the current one
        
        if len(successful_upirs) >= 3:  # Need minimum for clustering
            discovered = self.pattern_library.discover_patterns(
                successful_upirs,
                min_cluster_size=2
            )
            
            if discovered:
                logger.info(f"Discovered {len(discovered)} new patterns")
                
                for pattern in discovered:
                    logger.info(f"  - {pattern.name}: {pattern.description}")
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status."""
        return {
            "state": self.state.value,
            "current_upir": self.current_upir.id if self.current_upir else None,
            "active_deployments": len(self.active_deployments),
            "optimization_count": self.optimization_count,
            "metrics": self._get_recent_metrics(window_minutes=5),
            "pattern_library_size": len(self.pattern_library.patterns)
        }
    
    async def shutdown(self):
        """Gracefully shutdown orchestrator."""
        logger.info("Shutting down orchestrator")
        
        # Stop monitoring
        self.state = WorkflowState.FAILED  # Stops loops
        
        # Save state
        if self.config.checkpoint_path:
            self._save_checkpoint()
        
        # Save pattern library
        self.pattern_library.save()
        
        logger.info("Orchestrator shutdown complete")
    
    def _save_checkpoint(self):
        """Save orchestrator state to checkpoint."""
        checkpoint = {
            "state": self.state.value,
            "workflow_history": self.workflow_history,
            "metrics_history": self.metrics_history[-1000:],  # Keep last 1000
            "optimization_count": self.optimization_count,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        checkpoint_file = f"{self.config.checkpoint_path}/orchestrator_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        
        import os
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        logger.info(f"Checkpoint saved to {checkpoint_file}")