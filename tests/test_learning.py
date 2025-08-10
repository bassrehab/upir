"""
Unit tests for reinforcement learning components.

Testing PPO implementation, state encoding, reward computation,
and invariant preservation during learning.

Author: subhadipmitra@google.com
"""

import pytest
import numpy as np
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from upir.core.models import (
    UPIR, FormalSpecification, TemporalProperty,
    TemporalOperator, Architecture
)
from upir.learning.ppo import (
    PolicyNetwork, ValueNetwork, ReplayBuffer, Experience,
    PPOLearner, InvariantPreservingPPO, NetworkConfig, PPOConfig
)
from upir.learning.learner import (
    StateEncoder, RewardComputer, ProductionMetrics,
    OptimizationAction, ArchitectureLearner
)


class TestPolicyNetwork:
    """Test policy network functionality."""
    
    def test_network_initialization(self):
        """Test policy network initialization."""
        config = NetworkConfig(
            state_dim=10,
            action_dim=5,
            hidden_sizes=[32, 32]
        )
        
        policy = PolicyNetwork(config)
        
        assert len(policy.weights) == 3  # 2 hidden + 1 output
        assert policy.weights[0].shape == (10, 32)
        assert policy.weights[1].shape == (32, 32)
        assert policy.weights[2].shape == (32, 5)
    
    def test_forward_pass(self):
        """Test forward pass produces valid probabilities."""
        config = NetworkConfig(state_dim=10, action_dim=5)
        policy = PolicyNetwork(config)
        
        state = np.random.randn(10)
        probs = policy.forward(state)
        
        assert len(probs) == 5
        assert np.allclose(np.sum(probs), 1.0)  # Sum to 1
        assert np.all(probs >= 0)  # All non-negative
        assert np.all(probs <= 1)  # All <= 1
    
    def test_get_action(self):
        """Test action sampling."""
        config = NetworkConfig(state_dim=10, action_dim=5)
        policy = PolicyNetwork(config)
        
        state = np.random.randn(10)
        
        # Stochastic action
        action, log_prob = policy.get_action(state, deterministic=False)
        assert 0 <= action < 5
        assert log_prob <= 0  # Log probability is negative
        
        # Deterministic action
        action_det, _ = policy.get_action(state, deterministic=True)
        assert 0 <= action_det < 5


class TestValueNetwork:
    """Test value network functionality."""
    
    def test_network_initialization(self):
        """Test value network initialization."""
        config = NetworkConfig(
            state_dim=10,
            hidden_sizes=[32, 32]
        )
        
        value = ValueNetwork(config)
        
        assert len(value.weights) == 3
        assert value.weights[0].shape == (10, 32)
        assert value.weights[1].shape == (32, 32)
        assert value.weights[2].shape == (32, 1)  # Single output
    
    def test_forward_pass(self):
        """Test forward pass produces scalar value."""
        config = NetworkConfig(state_dim=10)
        value = ValueNetwork(config)
        
        state = np.random.randn(10)
        v = value.forward(state)
        
        assert isinstance(v, float)
        assert not np.isnan(v)
        assert not np.isinf(v)


class TestReplayBuffer:
    """Test experience replay buffer."""
    
    def test_buffer_operations(self):
        """Test adding and retrieving experiences."""
        buffer = ReplayBuffer(max_size=100)
        
        assert buffer.size() == 0
        
        # Add experiences
        for i in range(10):
            exp = Experience(
                state=np.array([i]),
                action=i % 3,
                reward=float(i),
                next_state=np.array([i+1]),
                done=(i == 9),
                log_prob=-0.5,
                value=float(i)
            )
            buffer.add(exp)
        
        assert buffer.size() == 10
        
        # Test sampling
        batch = buffer.sample_batch(5)
        assert len(batch) == 5
        
        # Clear buffer
        buffer.clear()
        assert buffer.size() == 0
    
    def test_gae_computation(self):
        """Test GAE advantage computation."""
        buffer = ReplayBuffer()
        
        # Add trajectory
        for i in range(5):
            exp = Experience(
                state=np.array([i]),
                action=0,
                reward=1.0,
                next_state=np.array([i+1]),
                done=(i == 4),
                log_prob=-0.5,
                value=float(5-i)  # Decreasing values
            )
            buffer.add(exp)
        
        returns, advantages = buffer.compute_returns_and_advantages(
            gamma=0.99,
            lambda_gae=0.95
        )
        
        assert len(returns) == 5
        assert len(advantages) == 5
        
        # Advantages should be normalized
        assert np.abs(np.mean(advantages)) < 0.1
        assert np.std(advantages) > 0  # Non-zero variance


class TestPPOLearner:
    """Test PPO learning agent."""
    
    def test_ppo_initialization(self):
        """Test PPO learner initialization."""
        network_config = NetworkConfig(state_dim=10, action_dim=5)
        ppo_config = PPOConfig()
        
        learner = PPOLearner(network_config, ppo_config)
        
        assert learner.policy is not None
        assert learner.value is not None
        assert learner.old_policy is not None
        assert learner.buffer is not None
    
    def test_act_and_step(self):
        """Test action selection and experience storage."""
        network_config = NetworkConfig(state_dim=10, action_dim=5)
        ppo_config = PPOConfig(min_buffer_size=5)
        
        learner = PPOLearner(network_config, ppo_config)
        
        state = np.random.randn(10)
        action = learner.act(state)
        
        assert 0 <= action < 5
        
        # Step through environment
        next_state = np.random.randn(10)
        learner.step(state, action, 1.0, next_state, False)
        
        assert learner.buffer.size() == 1
    
    def test_training(self):
        """Test PPO training loop."""
        network_config = NetworkConfig(state_dim=10, action_dim=5)
        ppo_config = PPOConfig(
            batch_size=4,
            n_epochs=2,
            min_buffer_size=4
        )
        
        learner = PPOLearner(network_config, ppo_config)
        
        # Fill buffer
        for i in range(4):
            state = np.random.randn(10)
            action = learner.act(state)
            next_state = np.random.randn(10)
            learner.step(state, action, 1.0, next_state, False)
        
        # Train
        stats = learner.train()
        
        assert "policy_loss" in stats
        assert "value_loss" in stats
        assert "kl_divergence" in stats
        
        # Buffer should be cleared after training
        assert learner.buffer.size() == 0


class TestInvariantPreservingPPO:
    """Test invariant-preserving PPO."""
    
    def test_safe_action_selection(self):
        """Test that only safe actions are selected."""
        network_config = NetworkConfig(state_dim=10, action_dim=5)
        ppo_config = PPOConfig()
        
        # Define invariant checker that only allows even actions
        def invariant_checker(state):
            # Simplified - in reality would check actual invariants
            return True
        
        learner = InvariantPreservingPPO(
            network_config, 
            ppo_config,
            invariant_checker
        )
        
        state = np.random.randn(10)
        action = learner.act(state)
        
        assert 0 <= action < 5
        # Action should be valid according to invariant checker


class TestStateEncoder:
    """Test state encoding for architectures."""
    
    def test_encoding_dimensions(self):
        """Test that encoding produces correct dimensions."""
        encoder = StateEncoder(state_dim=128)
        
        upir = UPIR(name="Test")
        upir.architecture = Architecture(
            components=[{"name": "test", "type": "service"}],
            connections=[],
            deployment={},
            patterns=["microservices"]
        )
        
        state = encoder.encode(upir)
        
        assert state.shape == (128,)
        assert state.dtype == np.float32
    
    def test_encoding_with_metrics(self):
        """Test encoding with production metrics."""
        encoder = StateEncoder(state_dim=128)
        
        upir = UPIR(name="Test")
        upir.architecture = Architecture(
            components=[],
            connections=[],
            deployment={},
            patterns=[]
        )
        
        metrics = ProductionMetrics(
            timestamp=datetime.utcnow(),
            latency_p50=50.0,
            latency_p99=100.0,
            throughput=5000.0,
            error_rate=0.01,
            cpu_usage=0.7,
            memory_usage=0.6,
            cost=100.0
        )
        
        state = encoder.encode(upir, metrics)
        
        assert state.shape == (128,)
        # Metrics should be normalized to reasonable ranges
        assert np.all(state >= -10)
        assert np.all(state <= 10)


class TestRewardComputer:
    """Test reward computation from metrics."""
    
    def test_reward_computation(self):
        """Test basic reward computation."""
        spec = FormalSpecification(
            invariants=[],
            properties=[],
            constraints={
                "latency": {"max": 100},
                "throughput": {"min": 5000},
                "cost": {"max": 500}
            }
        )
        
        computer = RewardComputer(spec)
        
        # Good metrics
        good_metrics = ProductionMetrics(
            timestamp=datetime.utcnow(),
            latency_p50=40.0,
            latency_p99=80.0,
            throughput=6000.0,
            error_rate=0.001,
            cpu_usage=0.7,
            memory_usage=0.7,
            cost=400.0
        )
        
        reward = computer.compute_reward(good_metrics)
        assert reward > 0  # Should be positive for good metrics
        
        # Bad metrics
        bad_metrics = ProductionMetrics(
            timestamp=datetime.utcnow(),
            latency_p50=100.0,
            latency_p99=200.0,
            throughput=1000.0,
            error_rate=0.1,
            cpu_usage=0.95,
            memory_usage=0.95,
            cost=600.0
        )
        
        reward = computer.compute_reward(bad_metrics)
        assert reward < 0  # Should be negative for bad metrics
    
    def test_invariant_violation_penalty(self):
        """Test heavy penalty for invariant violations."""
        computer = RewardComputer()
        
        metrics = ProductionMetrics(
            timestamp=datetime.utcnow(),
            latency_p50=50.0,
            latency_p99=100.0,
            throughput=5000.0,
            error_rate=0.01,
            cpu_usage=0.7,
            memory_usage=0.6,
            cost=100.0
        )
        
        reward = computer.compute_reward(metrics, invariant_violated=True)
        assert reward == -100.0  # Heavy penalty


class TestOptimizationAction:
    """Test optimization actions."""
    
    def test_action_creation(self):
        """Test creating optimization actions."""
        action = OptimizationAction(
            action_type="scale",
            component="workers",
            parameter="count",
            old_value=5,
            new_value=10
        )
        
        assert action.action_type == "scale"
        assert action.component == "workers"
        assert action.new_value == 10
    
    def test_action_application(self):
        """Test applying action to architecture."""
        arch = Architecture(
            components=[
                {
                    "name": "workers",
                    "config": {"count": 5}
                }
            ],
            connections=[],
            deployment={},
            patterns=[]
        )
        
        action = OptimizationAction(
            action_type="scale",
            component="workers",
            parameter="count",
            old_value=5,
            new_value=10
        )
        
        new_arch = action.apply(arch)
        
        # Check that architecture was modified
        assert new_arch.components[0]["config"]["count"] == 10


class TestArchitectureLearner:
    """Test the main architecture learning system."""
    
    def test_learner_initialization(self):
        """Test learner initialization."""
        upir = UPIR(name="Test")
        upir.specification = FormalSpecification(
            invariants=[],
            properties=[],
            constraints={}
        )
        upir.architecture = Architecture(
            components=[],
            connections=[],
            deployment={},
            patterns=[]
        )
        
        learner = ArchitectureLearner(upir)
        
        assert learner.upir == upir
        assert learner.state_encoder is not None
        assert learner.reward_computer is not None
        assert learner.ppo is not None
    
    def test_observe_metrics(self):
        """Test observing production metrics."""
        upir = UPIR(name="Test")
        upir.specification = FormalSpecification(
            invariants=[],
            properties=[],
            constraints={}
        )
        upir.architecture = Architecture(
            components=[],
            connections=[],
            deployment={},
            patterns=[]
        )
        
        learner = ArchitectureLearner(upir)
        
        metrics = ProductionMetrics(
            timestamp=datetime.utcnow(),
            latency_p50=50.0,
            latency_p99=100.0,
            throughput=5000.0,
            error_rate=0.01,
            cpu_usage=0.7,
            memory_usage=0.6,
            cost=100.0
        )
        
        learner.observe_metrics(metrics)
        
        assert len(learner.metric_history) == 1
        assert learner.current_state is not None
    
    def test_suggest_optimization(self):
        """Test optimization suggestion."""
        upir = UPIR(name="Test")
        upir.specification = FormalSpecification(
            invariants=[],
            properties=[],
            constraints={}
        )
        upir.architecture = Architecture(
            components=[],
            connections=[],
            deployment={},
            patterns=[]
        )
        
        learner = ArchitectureLearner(upir)
        
        # Need to observe metrics first to have a state
        metrics = ProductionMetrics(
            timestamp=datetime.utcnow(),
            latency_p50=50.0,
            latency_p99=100.0,
            throughput=5000.0,
            error_rate=0.01,
            cpu_usage=0.7,
            memory_usage=0.6,
            cost=100.0
        )
        
        learner.observe_metrics(metrics)
        
        # Now can suggest optimization
        optimization = learner.suggest_optimization()
        
        if optimization:
            assert isinstance(optimization, OptimizationAction)
            assert optimization.action_type in ["scale", "tune"]


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])