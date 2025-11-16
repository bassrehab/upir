"""
Unit tests for ArchitectureLearner.

Tests verify:
- Experience dataclass
- ArchitectureLearner initialization
- State encoding from UPIR architectures
- Action decoding to architectural modifications
- Reward computation from metrics
- Full learning loop integration

Author: Subhadip Mitra
License: Apache 2.0
"""

import numpy as np

from upir.core.architecture import Architecture
from upir.core.specification import FormalSpecification
from upir.core.temporal import TemporalOperator, TemporalProperty
from upir.core.upir import UPIR
from upir.learning.learner import ArchitectureLearner, Experience
from upir.learning.ppo import PPOConfig


class TestExperience:
    """Tests for Experience dataclass."""

    def test_create_experience(self):
        """Test creating experience."""
        exp = Experience(
            state=np.array([0.1, 0.2]),
            action=5,
            reward=0.5,
            log_prob=-1.2,
            value=0.3,
            done=False
        )
        assert exp.action == 5
        assert exp.reward == 0.5
        assert exp.done is False


class TestArchitectureLearnerInit:
    """Tests for ArchitectureLearner initialization."""

    def test_create_default(self):
        """Test creating learner with defaults."""
        learner = ArchitectureLearner()
        assert learner.state_dim == 64
        assert learner.action_dim == 40
        assert learner.ppo is not None

    def test_create_custom(self):
        """Test creating learner with custom parameters."""
        config = PPOConfig(learning_rate=1e-3)
        learner = ArchitectureLearner(
            state_dim=32,
            action_dim=20,
            config=config,
            buffer_size=500
        )
        assert learner.state_dim == 32
        assert learner.action_dim == 20
        assert learner.experience_buffer.maxlen == 500

    def test_feature_stats_initialized(self):
        """Test that feature normalization stats are initialized."""
        learner = ArchitectureLearner()
        assert "num_components_max" in learner.feature_stats
        assert learner.feature_stats["num_components_max"] > 0

    def test_str_repr(self):
        """Test string representations."""
        learner = ArchitectureLearner()
        str_rep = str(learner)
        assert "ArchitectureLearner" in str_rep

        repr_rep = repr(learner)
        assert "ArchitectureLearner" in repr_rep


class TestEncodeState:
    """Tests for state encoding."""

    def test_encode_empty_architecture(self):
        """Test encoding UPIR with no architecture."""
        learner = ArchitectureLearner()
        upir = UPIR(id="test", name="Test", description="Test")

        state = learner.encode_state(upir)

        # Should return zero vector
        assert state.shape == (64,)
        assert np.all(state == 0)

    def test_encode_simple_architecture(self):
        """Test encoding simple architecture."""
        learner = ArchitectureLearner()

        # Create simple architecture with dict components
        comp1 = {
            "id": "c1",
            "name": "Component1",
            "type": "processor",
            "latency_ms": 100,
            "throughput_qps": 1000,
            "parallelism": 10
        }
        comp2 = {
            "id": "c2",
            "name": "Component2",
            "type": "processor",
            "latency_ms": 50,
            "throughput_qps": 2000,
            "parallelism": 5
        }
        arch = Architecture(
            components=[comp1, comp2],
            connections=[]
        )
        upir = UPIR(id="test", name="Test", description="Test", architecture=arch)

        state = learner.encode_state(upir)

        # Check shape and range
        assert state.shape == (64,)
        assert np.all(state >= 0)
        assert np.all(state <= 1)

        # Should have non-zero features for components
        assert np.any(state > 0)

    def test_encode_with_connections(self):
        """Test encoding architecture with connections."""
        learner = ArchitectureLearner()

        comp1 = {"id": "c1", "name": "C1", "type": "source"}
        comp2 = {"id": "c2", "name": "C2", "type": "sink"}
        conn = {"from": "c1", "to": "c2"}

        arch = Architecture(components=[comp1, comp2], connections=[conn])
        upir = UPIR(id="test", name="Test", description="Test", architecture=arch)

        state = learner.encode_state(upir)

        assert state.shape == (64,)
        assert np.any(state > 0)  # Should have features for connections

    def test_encode_normalization(self):
        """Test that features are normalized to [0, 1]."""
        learner = ArchitectureLearner()

        # Create architecture with large values
        comp = {"id": "c1", "name": "C1", "type": "processor", **{"latency_ms": 50000, "throughput_qps": 500000}}
        arch = Architecture(components=[comp], connections=[])
        upir = UPIR(id="test", name="Test", description="Test", architecture=arch)

        state = learner.encode_state(upir)

        # All values should still be in [0, 1] due to clipping
        assert np.all(state >= 0)
        assert np.all(state <= 1)

    def test_encode_deterministic(self):
        """Test that encoding is deterministic."""
        learner = ArchitectureLearner()

        comp = {"id": "c1", "name": "C1", "type": "processor"}
        arch = Architecture(components=[comp], connections=[])
        upir = UPIR(id="test", name="Test", description="Test", architecture=arch)

        state1 = learner.encode_state(upir)
        state2 = learner.encode_state(upir)

        assert np.allclose(state1, state2)


class TestDecodeAction:
    """Tests for action decoding."""

    def test_decode_increase_parallelism(self):
        """Test action to increase parallelism."""
        learner = ArchitectureLearner()

        comp = {"id": "c1", "name": "C1", "type": "processor", **{"parallelism": 10}}
        arch = Architecture(components=[comp], connections=[])
        upir = UPIR(id="test", name="Test", description="Test", architecture=arch)

        # Action 0: increase parallelism of first component
        modified = learner.decode_action(0, upir)

        # Should increase parallelism
        assert modified.architecture.components[0]["parallelism"] == 11

        # Original should be unchanged
        assert upir.architecture.components[0]["parallelism"] == 10

    def test_decode_decrease_parallelism(self):
        """Test action to decrease parallelism."""
        learner = ArchitectureLearner()

        comp = {"id": "c1", "name": "C1", "type": "processor", **{"parallelism": 10}}
        arch = Architecture(components=[comp], connections=[])
        upir = UPIR(id="test", name="Test", description="Test", architecture=arch)

        # Action 10: decrease parallelism of first component
        modified = learner.decode_action(10, upir)

        assert modified.architecture.components[0]["parallelism"] == 9

    def test_decode_change_type(self):
        """Test action to change component type."""
        learner = ArchitectureLearner()

        comp = {"id": "c1", "name": "C1", "type": "streaming_processor"}
        arch = Architecture(components=[comp], connections=[])
        upir = UPIR(id="test", name="Test", description="Test", architecture=arch)

        # Action 20: change component type
        modified = learner.decode_action(20, upir)

        # Should change from streaming to batch
        assert modified.architecture.components[0]["type"] == "batch_processor"

    def test_decode_modify_connection(self):
        """Test action to modify connection."""
        learner = ArchitectureLearner()

        comp1 = {"id": "c1", "name": "C1", "type": "source"}
        comp2 = {"id": "c2", "name": "C2", "type": "sink"}
        conn = {"from": "c1", "to": "c2", **{"batched": False}}
        arch = Architecture(components=[comp1, comp2], connections=[conn])
        upir = UPIR(id="test", name="Test", description="Test", architecture=arch)

        # Action 30: modify connection
        modified = learner.decode_action(30, upir)

        # Should toggle batched property
        assert modified.architecture.connections[0]["batched"] is True

    def test_decode_empty_architecture(self):
        """Test decoding with no architecture."""
        learner = ArchitectureLearner()
        upir = UPIR(id="test", name="Test", description="Test")

        # Should not crash
        modified = learner.decode_action(0, upir)
        assert modified.architecture is None

    def test_decode_parallelism_bounds(self):
        """Test that parallelism stays within bounds."""
        learner = ArchitectureLearner()

        # Test maximum bound
        comp = {"id": "c1", "name": "C1", "type": "processor", "parallelism": 100}  # Already at max
        arch = Architecture(components=[comp], connections=[])
        upir = UPIR(id="test", name="Test", description="Test", architecture=arch)

        modified = learner.decode_action(0, upir)  # Try to increase
        assert modified.architecture.components[0]["parallelism"] == 100

        # Test minimum bound
        comp2 = {"id": "c1", "name": "C1", "type": "processor", "parallelism": 1}  # At minimum
        arch2 = Architecture(components=[comp2], connections=[])
        upir2 = UPIR(id="test", name="Test", description="Test", architecture=arch2)
        modified = learner.decode_action(10, upir2)  # Try to decrease
        assert modified.architecture.components[0]["parallelism"] == 1


class TestComputeReward:
    """Tests for reward computation."""

    def test_compute_reward_no_constraints(self):
        """Test reward with no constraints."""
        learner = ArchitectureLearner()
        metrics = {"latency_p99": 100, "throughput_qps": 1000}
        spec = FormalSpecification()

        reward = learner.compute_reward(metrics, spec)

        # Should get base reward
        assert isinstance(reward, (float, np.floating))
        assert -1.0 <= reward <= 1.0

    def test_compute_reward_met_latency_constraint(self):
        """Test reward when latency constraint is met."""
        learner = ArchitectureLearner()

        spec = FormalSpecification(
            properties=[
                TemporalProperty(
                    operator=TemporalOperator.WITHIN,
                    predicate="respond",
                    time_bound=200  # 200ms requirement
                )
            ]
        )
        metrics = {"latency_p99": 100}  # Meets requirement

        reward = learner.compute_reward(metrics, spec)

        # Should get bonus for meeting constraint (but clipped to 1.0)
        assert reward == 1.0  # Clipped max

    def test_compute_reward_violated_latency_constraint(self):
        """Test reward when latency constraint is violated."""
        learner = ArchitectureLearner()

        spec = FormalSpecification(
            properties=[
                TemporalProperty(
                    operator=TemporalOperator.WITHIN,
                    predicate="respond",
                    time_bound=50  # 50ms requirement
                )
            ]
        )
        metrics = {"latency_p99": 100}  # Violates requirement

        reward = learner.compute_reward(metrics, spec)

        # Should get penalty for violation
        assert reward < 1.0  # Base - penalty

    def test_compute_reward_improvement(self):
        """Test reward for performance improvement."""
        learner = ArchitectureLearner()

        spec = FormalSpecification()
        prev_metrics = {"latency_p99": 200, "throughput_qps": 500}
        curr_metrics = {"latency_p99": 100, "throughput_qps": 1000}

        reward = learner.compute_reward(curr_metrics, spec, prev_metrics)

        # Should get reward for improvements (clipped to 1.0)
        assert reward == 1.0  # Latency reduced 50%, throughput increased 100%

    def test_compute_reward_degradation(self):
        """Test penalty for performance degradation."""
        learner = ArchitectureLearner()

        spec = FormalSpecification()
        prev_metrics = {"latency_p99": 100, "throughput_qps": 1000}
        curr_metrics = {"latency_p99": 200, "throughput_qps": 500}

        reward = learner.compute_reward(curr_metrics, spec, prev_metrics)

        # Should get penalty for degradation
        assert reward < 1.0

    def test_compute_reward_high_error_rate(self):
        """Test penalty for high error rate."""
        learner = ArchitectureLearner()

        spec = FormalSpecification()
        metrics = {"latency_p99": 100, "error_rate": 0.1}  # 10% errors

        reward = learner.compute_reward(metrics, spec)

        # Should get heavy penalty
        assert reward < 1.0

    def test_compute_reward_clipped(self):
        """Test that reward is clipped to [-1, 1]."""
        learner = ArchitectureLearner()

        spec = FormalSpecification(
            properties=[
                TemporalProperty(
                    operator=TemporalOperator.WITHIN,
                    predicate="respond",
                    time_bound=50
                )
                for _ in range(10)  # Many constraints
            ]
        )
        metrics = {"latency_p99": 1000}  # Violates all

        reward = learner.compute_reward(metrics, spec)

        # Should be clipped
        assert -1.0 <= reward <= 1.0


class TestLearnFromMetrics:
    """Tests for learning loop."""

    def test_learn_from_metrics_basic(self):
        """Test basic learning from metrics."""
        learner = ArchitectureLearner()

        comp = {"id": "c1", "name": "C1", "type": "processor", **{"parallelism": 10}}
        arch = Architecture(components=[comp], connections=[])
        upir = UPIR(id="test", name="Test", description="Test", architecture=arch)

        metrics = {"latency_p99": 100, "throughput_qps": 1000}

        optimized = learner.learn_from_metrics(upir, metrics)

        # Should return modified UPIR
        assert optimized is not None
        assert optimized.architecture is not None

        # Should have stored experience
        assert len(learner.experience_buffer) > 0

    def test_learn_from_metrics_preserves_original(self):
        """Test that learning doesn't modify original UPIR."""
        learner = ArchitectureLearner()

        comp = {"id": "c1", "name": "C1", "type": "processor", **{"parallelism": 10}}
        arch = Architecture(components=[comp], connections=[])
        upir = UPIR(id="test", name="Test", description="Test", architecture=arch)

        original_parallelism = upir.architecture.components[0]["parallelism"]
        metrics = {"latency_p99": 100}

        optimized = learner.learn_from_metrics(upir, metrics)

        # Original should be unchanged
        assert upir.architecture.components[0]["parallelism"] == original_parallelism

    def test_learn_from_metrics_triggers_update(self):
        """Test that policy update is triggered when buffer is full."""
        config = PPOConfig(batch_size=4)  # Small batch for testing
        learner = ArchitectureLearner(config=config)

        comp = {"id": "c1", "name": "C1", "type": "processor", **{"parallelism": 10}}
        arch = Architecture(components=[comp], connections=[])
        upir = UPIR(id="test", name="Test", description="Test", architecture=arch)

        metrics = {"latency_p99": 100}

        # Run multiple learning steps
        for _ in range(5):
            learner.learn_from_metrics(upir, metrics)

        # Buffer should be cleared after update
        # (triggered when buffer size >= batch_size)
        assert len(learner.experience_buffer) < config.batch_size

    def test_learn_multiple_steps(self):
        """Test learning over multiple steps."""
        learner = ArchitectureLearner()

        comp = {"id": "c1", "name": "C1", "type": "processor", **{"parallelism": 10}}
        arch = Architecture(components=[comp], connections=[])
        upir = UPIR(id="test", name="Test", description="Test", architecture=arch)

        # Simulate multiple steps
        prev_metrics = None
        for step in range(10):
            metrics = {
                "latency_p99": 100 - step,  # Improving
                "throughput_qps": 1000 + step * 100
            }

            optimized = learner.learn_from_metrics(upir, metrics, prev_metrics)
            upir = optimized  # Use optimized for next step
            prev_metrics = metrics

        # Should have experiences
        assert len(learner.experience_buffer) > 0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_metrics(self):
        """Test with empty metrics dict."""
        learner = ArchitectureLearner()
        upir = UPIR(id="test", name="Test", description="Test")

        metrics = {}
        spec = FormalSpecification()

        reward = learner.compute_reward(metrics, spec)
        assert -1.0 <= reward <= 1.0

    def test_many_components(self):
        """Test encoding with many components."""
        learner = ArchitectureLearner()

        # Create 20 components
        components = [
            {"id": f"c{i}", "name": f"C{i}", "type": "processor"}
            for i in range(20)
        ]
        arch = Architecture(components=components, connections=[])
        upir = UPIR(id="test", name="Test", description="Test", architecture=arch)

        state = learner.encode_state(upir)

        # Should handle gracefully
        assert state.shape == (64,)
        assert not np.any(np.isnan(state))

    def test_action_wrapping(self):
        """Test that actions wrap around correctly."""
        learner = ArchitectureLearner()

        # Single component
        comp = {"id": "c1", "name": "C1", "type": "processor"}
        arch = Architecture(components=[comp], connections=[])
        upir = UPIR(id="test", name="Test", description="Test", architecture=arch)

        # Action > num_components should wrap
        modified = learner.decode_action(5, upir)  # Action 5 % 1 = 0
        assert modified is not None

    def test_zero_previous_metrics(self):
        """Test reward computation with zero previous metrics."""
        learner = ArchitectureLearner()

        spec = FormalSpecification()
        prev_metrics = {"latency_p99": 0.0, "throughput_qps": 0.0}
        curr_metrics = {"latency_p99": 100, "throughput_qps": 1000}

        # Should handle division by zero gracefully
        reward = learner.compute_reward(curr_metrics, spec, prev_metrics)
        assert not np.isnan(reward)
        assert -1.0 <= reward <= 1.0
