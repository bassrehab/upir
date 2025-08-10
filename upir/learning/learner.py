"""
Reinforcement Learning Module for Architecture Optimization

This module connects PPO with UPIR to enable learning from production
metrics while maintaining formal invariants. The system learns to optimize
architectures based on real-world performance data.

The learning process:
1. Encode current architecture as state
2. PPO suggests optimization actions
3. Verify actions preserve invariants
4. Apply safe actions and observe metrics
5. Learn from rewards to improve future decisions

Author: subhadipmitra@google.com
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
import json

from ..core.models import UPIR, Architecture, FormalSpecification, TemporalProperty
from ..verification.verifier import Verifier, VerificationStatus
from .ppo import PPOLearner, InvariantPreservingPPO, NetworkConfig, PPOConfig

logger = logging.getLogger(__name__)


@dataclass
class ProductionMetrics:
    """Metrics collected from production system."""
    timestamp: datetime
    latency_p50: float  # Median latency in ms
    latency_p99: float  # 99th percentile latency
    throughput: float  # Events/requests per second
    error_rate: float  # Percentage of errors
    cpu_usage: float  # CPU utilization (0-1)
    memory_usage: float  # Memory utilization (0-1)
    cost: float  # Hourly cost in dollars
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "latency_p50": self.latency_p50,
            "latency_p99": self.latency_p99,
            "throughput": self.throughput,
            "error_rate": self.error_rate,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "cost": self.cost,
            "custom_metrics": self.custom_metrics
        }


@dataclass
class OptimizationAction:
    """An action that modifies the architecture."""
    action_type: str  # "scale", "tune", "restructure", etc.
    component: str  # Which component to modify
    parameter: str  # Which parameter to change
    old_value: Any  # Previous value
    new_value: Any  # New value
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def apply(self, architecture: Architecture) -> Architecture:
        """Apply action to architecture."""
        # This is simplified - real implementation would properly modify architecture
        modified_arch = Architecture(
            components=architecture.components.copy(),
            connections=architecture.connections.copy(),
            deployment=architecture.deployment.copy(),
            patterns=architecture.patterns.copy()
        )
        
        # Find and modify component
        for comp in modified_arch.components:
            if comp.get("name") == self.component:
                if "config" in comp:
                    comp["config"][self.parameter] = self.new_value
                break
        
        return modified_arch
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "action_type": self.action_type,
            "component": self.component,
            "parameter": self.parameter,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "timestamp": self.timestamp.isoformat()
        }


class StateEncoder:
    """
    Encodes UPIR architectures into state vectors for RL.
    
    This is crucial - we need to represent complex architectures
    as fixed-size vectors that capture the essential information
    for optimization decisions.
    """
    
    def __init__(self, state_dim: int = 128):
        self.state_dim = state_dim
        self.feature_extractors = {
            "components": self._extract_component_features,
            "connections": self._extract_connection_features,
            "metrics": self._extract_metric_features,
            "patterns": self._extract_pattern_features
        }
    
    def encode(self, upir: UPIR, metrics: Optional[ProductionMetrics] = None) -> np.ndarray:
        """
        Encode UPIR and metrics into state vector.
        
        We extract features from different aspects of the architecture
        and combine them into a fixed-size representation.
        """
        features = []
        
        if upir.architecture:
            # Extract architectural features
            features.extend(self._extract_component_features(upir.architecture))
            features.extend(self._extract_connection_features(upir.architecture))
            features.extend(self._extract_pattern_features(upir.architecture))
        
        if metrics:
            # Extract metric features
            features.extend(self._extract_metric_features(metrics))
        
        if upir.specification:
            # Extract specification features
            features.extend(self._extract_spec_features(upir.specification))
        
        # Pad or truncate to state_dim
        if len(features) < self.state_dim:
            features.extend([0.0] * (self.state_dim - len(features)))
        else:
            features = features[:self.state_dim]
        
        return np.array(features, dtype=np.float32)
    
    def _extract_component_features(self, architecture: Architecture) -> List[float]:
        """Extract features from components."""
        features = []
        
        # Number of components
        features.append(float(len(architecture.components)))
        
        # Component type distribution
        type_counts = {}
        for comp in architecture.components:
            comp_type = comp.get("type", "unknown")
            type_counts[comp_type] = type_counts.get(comp_type, 0) + 1
        
        # Normalize counts
        total = max(1, len(architecture.components))
        for comp_type in ["service", "storage", "queue", "cache", "processor"]:
            features.append(type_counts.get(comp_type, 0) / total)
        
        # Average component complexity (simplified)
        complexities = []
        for comp in architecture.components:
            config = comp.get("config", {})
            complexity = len(config)  # Simple proxy
            complexities.append(complexity)
        
        if complexities:
            features.append(np.mean(complexities))
            features.append(np.std(complexities))
        else:
            features.extend([0.0, 0.0])
        
        return features
    
    def _extract_connection_features(self, architecture: Architecture) -> List[float]:
        """Extract features from connections."""
        features = []
        
        # Number of connections
        features.append(float(len(architecture.connections)))
        
        # Connection density (connections per component)
        if architecture.components:
            density = len(architecture.connections) / len(architecture.components)
            features.append(density)
        else:
            features.append(0.0)
        
        # Fan-out distribution
        fan_outs = {}
        for conn in architecture.connections:
            source = conn.get("source")
            if source:
                fan_outs[source] = fan_outs.get(source, 0) + 1
        
        if fan_outs:
            features.append(np.mean(list(fan_outs.values())))
            features.append(np.max(list(fan_outs.values())))
        else:
            features.extend([0.0, 0.0])
        
        return features
    
    def _extract_metric_features(self, metrics: ProductionMetrics) -> List[float]:
        """Extract features from production metrics."""
        features = []
        
        # Normalize metrics to [0, 1] range
        features.append(min(1.0, metrics.latency_p50 / 1000.0))  # Normalize to seconds
        features.append(min(1.0, metrics.latency_p99 / 1000.0))
        features.append(min(1.0, metrics.throughput / 10000.0))  # Normalize to 10k
        features.append(metrics.error_rate)  # Already in [0, 1]
        features.append(metrics.cpu_usage)
        features.append(metrics.memory_usage)
        features.append(min(1.0, metrics.cost / 100.0))  # Normalize to $100/hour
        
        return features
    
    def _extract_pattern_features(self, architecture: Architecture) -> List[float]:
        """Extract features from architectural patterns."""
        features = []
        
        # One-hot encoding of common patterns
        common_patterns = [
            "microservices", "monolithic", "serverless",
            "event_sourcing", "cqrs", "layered",
            "streaming", "batch", "lambda"
        ]
        
        for pattern in common_patterns:
            features.append(1.0 if pattern in architecture.patterns else 0.0)
        
        return features
    
    def _extract_spec_features(self, spec: FormalSpecification) -> List[float]:
        """Extract features from formal specification."""
        features = []
        
        # Number of invariants and properties
        features.append(float(len(spec.invariants)))
        features.append(float(len(spec.properties)))
        
        # Constraint features
        if "latency" in spec.constraints:
            max_latency = spec.constraints["latency"].get("max", 1000)
            features.append(max_latency / 1000.0)  # Normalize
        else:
            features.append(1.0)
        
        if "throughput" in spec.constraints:
            min_throughput = spec.constraints["throughput"].get("min", 0)
            features.append(min_throughput / 10000.0)  # Normalize
        else:
            features.append(0.0)
        
        if "cost" in spec.constraints:
            max_cost = spec.constraints["cost"].get("max", 10000)
            features.append(max_cost / 10000.0)  # Normalize
        else:
            features.append(1.0)
        
        return features


class RewardComputer:
    """
    Computes rewards from production metrics.
    
    The reward function is critical - it determines what the agent
    optimizes for. We balance performance, cost, and reliability.
    """
    
    def __init__(self, spec: Optional[FormalSpecification] = None):
        self.spec = spec
        self.baseline_metrics = None
        
        # Reward weights (can be tuned)
        self.weights = {
            "latency": 0.3,
            "throughput": 0.2,
            "error_rate": 0.2,
            "cost": 0.2,
            "resource_usage": 0.1
        }
    
    def set_baseline(self, metrics: ProductionMetrics) -> None:
        """Set baseline metrics for relative reward computation."""
        self.baseline_metrics = metrics
    
    def compute_reward(self, metrics: ProductionMetrics, 
                      invariant_violated: bool = False) -> float:
        """
        Compute reward from metrics.
        
        We use a combination of absolute and relative rewards to
        encourage improvement while maintaining good performance.
        """
        if invariant_violated:
            # Heavy penalty for violating invariants
            return -100.0
        
        reward = 0.0
        
        # Latency reward (lower is better)
        latency_score = self._compute_latency_reward(metrics)
        reward += self.weights["latency"] * latency_score
        
        # Throughput reward (higher is better)
        throughput_score = self._compute_throughput_reward(metrics)
        reward += self.weights["throughput"] * throughput_score
        
        # Error rate reward (lower is better)
        error_score = self._compute_error_reward(metrics)
        reward += self.weights["error_rate"] * error_score
        
        # Cost reward (lower is better)
        cost_score = self._compute_cost_reward(metrics)
        reward += self.weights["cost"] * cost_score
        
        # Resource usage reward (efficient usage is better)
        resource_score = self._compute_resource_reward(metrics)
        reward += self.weights["resource_usage"] * resource_score
        
        # Bonus for meeting all constraints
        if self._meets_all_constraints(metrics):
            reward += 10.0
        
        return reward
    
    def _compute_latency_reward(self, metrics: ProductionMetrics) -> float:
        """Compute reward for latency."""
        # Target latency from spec or default
        if self.spec and "latency" in self.spec.constraints:
            target = self.spec.constraints["latency"].get("max", 100)
        else:
            target = 100  # Default 100ms
        
        # Exponential decay reward
        if metrics.latency_p99 <= target:
            return 10.0  # Full reward for meeting target
        else:
            # Penalty that increases with distance from target
            excess_ratio = (metrics.latency_p99 - target) / target
            return max(-10.0, 10.0 * np.exp(-excess_ratio))
    
    def _compute_throughput_reward(self, metrics: ProductionMetrics) -> float:
        """Compute reward for throughput."""
        if self.spec and "throughput" in self.spec.constraints:
            target = self.spec.constraints["throughput"].get("min", 1000)
        else:
            target = 1000
        
        if metrics.throughput >= target:
            # Bonus for exceeding target
            excess_ratio = (metrics.throughput - target) / target
            return min(10.0, 10.0 * (1 + 0.1 * excess_ratio))
        else:
            # Penalty for not meeting target
            deficit_ratio = (target - metrics.throughput) / target
            return max(-10.0, -10.0 * deficit_ratio)
    
    def _compute_error_reward(self, metrics: ProductionMetrics) -> float:
        """Compute reward for error rate."""
        # Target near-zero errors
        if metrics.error_rate <= 0.001:  # 0.1% error rate
            return 10.0
        elif metrics.error_rate <= 0.01:  # 1% error rate
            return 5.0
        elif metrics.error_rate <= 0.05:  # 5% error rate
            return 0.0
        else:
            # Heavy penalty for high error rates
            return -10.0 * metrics.error_rate
    
    def _compute_cost_reward(self, metrics: ProductionMetrics) -> float:
        """Compute reward for cost efficiency."""
        if self.spec and "cost" in self.spec.constraints:
            target = self.spec.constraints["cost"].get("max", 1000)
        else:
            target = 1000  # $1000/hour default
        
        if metrics.cost <= target:
            # Reward for staying under budget
            savings_ratio = (target - metrics.cost) / target
            return 10.0 * (1 + savings_ratio)
        else:
            # Penalty for exceeding budget
            excess_ratio = (metrics.cost - target) / target
            return max(-10.0, -10.0 * excess_ratio)
    
    def _compute_resource_reward(self, metrics: ProductionMetrics) -> float:
        """Compute reward for resource efficiency."""
        # Optimal resource usage is around 70-80%
        cpu_score = 10.0 if 0.6 <= metrics.cpu_usage <= 0.8 else \
                   5.0 if 0.5 <= metrics.cpu_usage <= 0.9 else \
                   0.0 if 0.3 <= metrics.cpu_usage <= 0.95 else -5.0
        
        memory_score = 10.0 if 0.6 <= metrics.memory_usage <= 0.8 else \
                      5.0 if 0.5 <= metrics.memory_usage <= 0.9 else \
                      0.0 if 0.3 <= metrics.memory_usage <= 0.95 else -5.0
        
        return (cpu_score + memory_score) / 2
    
    def _meets_all_constraints(self, metrics: ProductionMetrics) -> bool:
        """Check if metrics meet all specification constraints."""
        if not self.spec:
            return True
        
        if "latency" in self.spec.constraints:
            if metrics.latency_p99 > self.spec.constraints["latency"].get("max", float('inf')):
                return False
        
        if "throughput" in self.spec.constraints:
            if metrics.throughput < self.spec.constraints["throughput"].get("min", 0):
                return False
        
        if "error_rate" in self.spec.constraints:
            if metrics.error_rate > self.spec.constraints["error_rate"].get("max", 1.0):
                return False
        
        if "cost" in self.spec.constraints:
            if metrics.cost > self.spec.constraints["cost"].get("max", float('inf')):
                return False
        
        return True


class ArchitectureLearner:
    """
    Main learning system that optimizes architectures using PPO.
    
    This brings everything together - encoding states, computing rewards,
    selecting actions, and learning from experience while maintaining invariants.
    """
    
    def __init__(self, upir: UPIR, use_invariant_preservation: bool = True):
        self.upir = upir
        self.verifier = Verifier()
        
        # Initialize components
        self.state_encoder = StateEncoder(state_dim=128)
        self.reward_computer = RewardComputer(upir.specification)
        
        # Configure PPO
        network_config = NetworkConfig(
            state_dim=128,
            action_dim=20,  # Different optimization actions
            hidden_sizes=[256, 256],
            learning_rate=3e-4
        )
        
        ppo_config = PPOConfig(
            clip_epsilon=0.2,
            gamma=0.99,
            lambda_gae=0.95,
            invariant_penalty=100.0
        )
        
        # Initialize PPO learner
        if use_invariant_preservation:
            self.ppo = InvariantPreservingPPO(
                network_config, 
                ppo_config,
                invariant_checker=self._check_invariants
            )
        else:
            self.ppo = PPOLearner(network_config, ppo_config)
        
        # Action space definition
        self.action_space = self._define_action_space()
        
        # Learning history
        self.optimization_history = []
        self.metric_history = []
        self.current_state = None
        self.current_action = None
    
    def _define_action_space(self) -> List[OptimizationAction]:
        """
        Define possible optimization actions.
        
        These are the levers the agent can pull to optimize the architecture.
        """
        actions = []
        
        # Scaling actions
        for scale_factor in [0.5, 0.8, 1.2, 2.0]:
            actions.append(OptimizationAction(
                action_type="scale",
                component="workers",
                parameter="count",
                old_value=None,
                new_value=scale_factor
            ))
        
        # Tuning actions
        for window_size in [10, 30, 60, 300]:
            actions.append(OptimizationAction(
                action_type="tune",
                component="processor",
                parameter="window_size",
                old_value=None,
                new_value=window_size
            ))
        
        # Buffer size adjustments
        for buffer_size in [100, 500, 1000, 5000]:
            actions.append(OptimizationAction(
                action_type="tune",
                component="buffer",
                parameter="size",
                old_value=None,
                new_value=buffer_size
            ))
        
        # Caching strategies
        for cache_ttl in [60, 300, 600, 3600]:
            actions.append(OptimizationAction(
                action_type="tune",
                component="cache",
                parameter="ttl",
                old_value=None,
                new_value=cache_ttl
            ))
        
        return actions
    
    def observe_metrics(self, metrics: ProductionMetrics) -> None:
        """
        Process new production metrics and learn.
        
        This is called periodically with fresh metrics from the production system.
        """
        # Encode current state
        state = self.state_encoder.encode(self.upir, metrics)
        
        # If we have a previous state and action, learn from the transition
        if self.current_state is not None and self.current_action is not None:
            # Compute reward
            reward = self.reward_computer.compute_reward(metrics)
            
            # Check if invariants are violated
            invariant_violated = not self._check_invariants(state)
            
            # Store experience
            self.ppo.step(
                self.current_state,
                self.current_action,
                reward,
                state,
                done=False,
                invariant_violated=invariant_violated
            )
        
        # Store current state for next iteration
        self.current_state = state
        
        # Record metrics
        self.metric_history.append(metrics)
    
    def suggest_optimization(self) -> Optional[OptimizationAction]:
        """
        Suggest an optimization action based on current state.
        
        This is where the learned policy is applied to make decisions.
        """
        if self.current_state is None:
            return None
        
        # Get action from policy
        action_idx = self.ppo.act(self.current_state, deterministic=False)
        self.current_action = action_idx
        
        # Map to actual optimization action
        if 0 <= action_idx < len(self.action_space):
            action = self.action_space[action_idx]
            
            # Record in history
            self.optimization_history.append(action)
            
            return action
        
        return None
    
    def apply_optimization(self, action: OptimizationAction) -> bool:
        """
        Apply optimization action to architecture.
        
        Returns True if action was successfully applied and invariants preserved.
        """
        # Apply action to get new architecture
        new_architecture = action.apply(self.upir.architecture)
        
        # Verify invariants are preserved
        temp_upir = UPIR(
            specification=self.upir.specification,
            architecture=new_architecture
        )
        
        if self._verify_architecture(temp_upir):
            # Update architecture
            self.upir.architecture = new_architecture
            self.upir.updated_at = datetime.utcnow()
            
            logger.info(f"Applied optimization: {action.action_type} on {action.component}")
            return True
        else:
            logger.warning(f"Optimization violates invariants: {action.action_type}")
            return False
    
    def _check_invariants(self, state: np.ndarray) -> bool:
        """
        Check if state preserves all invariants.
        
        This is critical for safety - we never want to violate formal properties.
        """
        # For now, simplified check
        # Real implementation would decode state and verify against specification
        return True  # Placeholder
    
    def _verify_architecture(self, upir: UPIR) -> bool:
        """Verify architecture against specification."""
        if not upir.specification or not upir.architecture:
            return True
        
        results = self.verifier.verify_specification(upir)
        
        # Check all invariants are satisfied
        for result in results:
            if result.property in upir.specification.invariants:
                if not result.verified:
                    return False
        
        return True
    
    def train(self) -> Dict[str, float]:
        """
        Train the PPO agent on collected experiences.
        
        This should be called periodically to improve the policy.
        """
        return self.ppo.train()
    
    def save(self, filepath: str) -> None:
        """Save learner state."""
        self.ppo.save_checkpoint(filepath)
    
    def load(self, filepath: str) -> None:
        """Load learner state."""
        self.ppo.load_checkpoint(filepath)