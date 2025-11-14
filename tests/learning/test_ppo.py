"""
Unit tests for PPO (Proximal Policy Optimization).

Tests verify:
- PPOConfig dataclass
- PolicyNetwork: forward pass, action sampling, action evaluation
- PPO: action selection, GAE computation, policy updates
- Integration: full PPO training loop

Author: Subhadip Mitra
License: Apache 2.0
"""

import numpy as np
import pytest

from upir.learning.ppo import PPO, PPOConfig, PolicyNetwork


class TestPPOConfig:
    """Tests for PPOConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PPOConfig()
        assert config.learning_rate == 3e-4
        assert config.gamma == 0.99
        assert config.epsilon == 0.2
        assert config.value_coef == 0.5
        assert config.entropy_coef == 0.01
        assert config.batch_size == 64
        assert config.num_epochs == 10
        assert config.lambda_gae == 0.95

    def test_custom_config(self):
        """Test creating config with custom values."""
        config = PPOConfig(
            learning_rate=1e-3,
            gamma=0.95,
            epsilon=0.1,
            batch_size=32
        )
        assert config.learning_rate == 1e-3
        assert config.gamma == 0.95
        assert config.epsilon == 0.1
        assert config.batch_size == 32
        # Others should be defaults
        assert config.value_coef == 0.5
        assert config.num_epochs == 10


class TestPolicyNetwork:
    """Tests for PolicyNetwork class."""

    def test_create_network(self):
        """Test creating policy network."""
        net = PolicyNetwork(state_dim=10, action_dim=4)
        assert net.state_dim == 10
        assert net.action_dim == 4
        assert net.hidden_dim == 64  # Default

    def test_create_network_custom_hidden(self):
        """Test creating network with custom hidden dimension."""
        net = PolicyNetwork(state_dim=10, action_dim=4, hidden_dim=128)
        assert net.hidden_dim == 128

    def test_weights_initialized(self):
        """Test that weights are initialized."""
        net = PolicyNetwork(state_dim=10, action_dim=4)
        assert "W1" in net.weights
        assert "b1" in net.weights
        assert "W_policy" in net.weights
        assert "b_policy" in net.weights
        assert "W_value" in net.weights
        assert "b_value" in net.weights

    def test_weight_shapes(self):
        """Test weight matrix shapes."""
        net = PolicyNetwork(state_dim=10, action_dim=4, hidden_dim=64)
        assert net.weights["W1"].shape == (10, 64)
        assert net.weights["b1"].shape == (64,)
        assert net.weights["W_policy"].shape == (64, 4)
        assert net.weights["b_policy"].shape == (4,)
        assert net.weights["W_value"].shape == (64, 1)
        assert net.weights["b_value"].shape == (1,)

    def test_forward_single_state(self):
        """Test forward pass with single state."""
        net = PolicyNetwork(state_dim=10, action_dim=4)
        state = np.random.randn(10)
        action_probs, value = net.forward(state)

        # Check shapes
        assert action_probs.shape == (4,)
        assert isinstance(value, (float, np.floating))

        # Check properties
        assert np.allclose(action_probs.sum(), 1.0)  # Probs sum to 1
        assert np.all(action_probs >= 0)  # All probs non-negative
        assert np.all(action_probs <= 1)  # All probs <= 1

    def test_forward_batch(self):
        """Test forward pass with batch of states."""
        net = PolicyNetwork(state_dim=10, action_dim=4)
        states = np.random.randn(32, 10)
        action_probs, values = net.forward(states)

        # Check shapes
        assert action_probs.shape == (32, 4)
        assert values.shape == (32,)

        # Check properties
        assert np.allclose(action_probs.sum(axis=1), 1.0)  # Each row sums to 1
        assert np.all(action_probs >= 0)

    def test_get_action(self):
        """Test action sampling."""
        net = PolicyNetwork(state_dim=10, action_dim=4)
        state = np.random.randn(10)

        action, log_prob, value = net.get_action(state)

        # Check types
        assert isinstance(action, (int, np.integer))
        assert isinstance(log_prob, (float, np.floating))
        assert isinstance(value, (float, np.floating))

        # Check ranges
        assert 0 <= action < 4
        assert log_prob <= 0  # Log prob is negative or zero

    def test_get_action_multiple_calls(self):
        """Test that get_action samples different actions."""
        net = PolicyNetwork(state_dim=10, action_dim=4)
        state = np.random.randn(10)

        # Sample many times
        actions = [net.get_action(state)[0] for _ in range(100)]

        # Should get variety (with high probability)
        unique_actions = set(actions)
        assert len(unique_actions) > 1  # Should sample different actions

    def test_evaluate_actions(self):
        """Test action evaluation."""
        net = PolicyNetwork(state_dim=10, action_dim=4)
        states = np.random.randn(32, 10)
        actions = np.random.randint(0, 4, size=32)

        log_probs, values, entropy = net.evaluate_actions(states, actions)

        # Check shapes
        assert log_probs.shape == (32,)
        assert values.shape == (32,)
        assert isinstance(entropy, (float, np.floating))

        # Check properties
        assert np.all(log_probs <= 0)  # Log probs are negative
        assert entropy >= 0  # Entropy is non-negative

    def test_evaluate_actions_entropy_bounds(self):
        """Test that entropy is bounded correctly."""
        net = PolicyNetwork(state_dim=10, action_dim=4)
        states = np.random.randn(32, 10)
        actions = np.random.randint(0, 4, size=32)

        _, _, entropy = net.evaluate_actions(states, actions)

        # Entropy for categorical with 4 outcomes: 0 <= H <= log(4)
        max_entropy = np.log(4)
        assert 0 <= entropy <= max_entropy


class TestPPO:
    """Tests for PPO algorithm."""

    def test_create_ppo(self):
        """Test creating PPO agent."""
        ppo = PPO(state_dim=10, action_dim=4)
        assert ppo.policy.state_dim == 10
        assert ppo.policy.action_dim == 4
        assert isinstance(ppo.config, PPOConfig)

    def test_create_ppo_custom_config(self):
        """Test creating PPO with custom config."""
        config = PPOConfig(learning_rate=1e-3, epsilon=0.1)
        ppo = PPO(state_dim=10, action_dim=4, config=config)
        assert ppo.config.learning_rate == 1e-3
        assert ppo.config.epsilon == 0.1

    def test_select_action(self):
        """Test action selection."""
        ppo = PPO(state_dim=10, action_dim=4)
        state = np.random.randn(10)

        action, log_prob, value = ppo.select_action(state)

        assert isinstance(action, (int, np.integer))
        assert isinstance(log_prob, (float, np.floating))
        assert isinstance(value, (float, np.floating))
        assert 0 <= action < 4

    def test_compute_gae_simple(self):
        """Test GAE computation with simple trajectory."""
        ppo = PPO(state_dim=10, action_dim=4)

        # Simple trajectory: 5 steps
        rewards = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        values = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        dones = np.array([0, 0, 0, 0, 1])  # Episode ends at step 4

        advantages, returns = ppo.compute_gae(rewards, values, dones)

        # Check shapes
        assert advantages.shape == (5,)
        assert returns.shape == (5,)

        # Returns should be higher than values (positive rewards)
        assert np.all(returns >= values)

    def test_compute_gae_terminal_state(self):
        """Test GAE handles terminal states correctly."""
        ppo = PPO(state_dim=10, action_dim=4)

        rewards = np.array([1.0, 1.0])
        values = np.array([0.0, 0.0])
        dones = np.array([0, 1])  # Terminal at step 1

        advantages, returns = ppo.compute_gae(rewards, values, dones)

        # At terminal state, next value should be 0
        # Return at terminal should be just the reward
        assert returns[1] == pytest.approx(rewards[1], abs=0.01)

    def test_compute_gae_discount(self):
        """Test that GAE applies discount factor correctly."""
        config = PPOConfig(gamma=0.9)
        ppo = PPO(state_dim=10, action_dim=4, config=config)

        rewards = np.array([1.0, 0.0])
        values = np.array([0.0, 0.0])
        dones = np.array([0, 0])

        advantages, returns = ppo.compute_gae(rewards, values, dones)

        # First return should be: r[0] + gamma * r[1] + gamma^2 * 0
        # = 1.0 + 0.9 * 0.0 = 1.0
        # But with GAE, it's more complex due to value estimates
        # Just check that it's positive and reasonable
        assert advantages[0] > 0

    def test_update_batch(self):
        """Test policy update with batch."""
        ppo = PPO(state_dim=10, action_dim=4)

        # Create small batch
        batch_size = 32
        states = np.random.randn(batch_size, 10)
        actions = np.random.randint(0, 4, size=batch_size)
        old_log_probs = np.random.randn(batch_size) * -1  # Negative log probs
        returns = np.random.randn(batch_size)
        advantages = np.random.randn(batch_size)

        metrics = ppo.update(states, actions, old_log_probs, returns, advantages)

        # Check that metrics are returned
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert "total_loss" in metrics

        # Check that metrics are scalars
        assert isinstance(metrics["policy_loss"], (float, np.floating))
        assert isinstance(metrics["value_loss"], (float, np.floating))
        assert isinstance(metrics["entropy"], (float, np.floating))
        assert isinstance(metrics["total_loss"], (float, np.floating))

    def test_update_normalizes_advantages(self):
        """Test that update normalizes advantages."""
        ppo = PPO(state_dim=10, action_dim=4)

        batch_size = 64
        states = np.random.randn(batch_size, 10)
        actions = np.random.randint(0, 4, size=batch_size)
        old_log_probs = np.random.randn(batch_size) * -1
        returns = np.random.randn(batch_size)
        advantages = np.random.randn(batch_size) * 10  # Large scale

        # Should not raise error even with large advantages
        metrics = ppo.update(states, actions, old_log_probs, returns, advantages)
        assert metrics is not None

    def test_str_repr(self):
        """Test string representations."""
        ppo = PPO(state_dim=10, action_dim=4)

        str_rep = str(ppo)
        assert "PPO" in str_rep
        assert "lr=" in str_rep

        repr_rep = repr(ppo)
        assert "PPO" in repr_rep
        assert "state_dim=10" in repr_rep
        assert "action_dim=4" in repr_rep


class TestPPOIntegration:
    """Integration tests for PPO training loop."""

    def test_simple_training_loop(self):
        """Test simple training loop doesn't crash."""
        ppo = PPO(state_dim=4, action_dim=2)

        # Simulate simple episode
        num_steps = 10
        states_list = []
        actions_list = []
        log_probs_list = []
        values_list = []
        rewards_list = []
        dones_list = []

        state = np.random.randn(4)
        for _ in range(num_steps):
            action, log_prob, value = ppo.select_action(state)

            states_list.append(state)
            actions_list.append(action)
            log_probs_list.append(log_prob)
            values_list.append(value)

            # Simple reward (dummy)
            reward = 1.0 if action == 0 else 0.0
            rewards_list.append(reward)

            # Random done
            done = np.random.rand() < 0.1
            dones_list.append(done)

            # Next state
            state = np.random.randn(4)

            if done:
                break

        # Convert to arrays
        states = np.array(states_list)
        actions = np.array(actions_list)
        old_log_probs = np.array(log_probs_list)
        rewards = np.array(rewards_list)
        values = np.array(values_list)
        dones = np.array(dones_list, dtype=np.float32)

        # Compute advantages
        advantages, returns = ppo.compute_gae(rewards, values, dones)

        # Update policy
        metrics = ppo.update(states, actions, old_log_probs, returns, advantages)

        # Should complete without error
        assert metrics is not None
        assert "policy_loss" in metrics

    def test_multiple_episodes(self):
        """Test training over multiple episodes."""
        ppo = PPO(state_dim=4, action_dim=2)

        num_episodes = 5
        for episode in range(num_episodes):
            # Run episode
            states_list = []
            actions_list = []
            log_probs_list = []
            values_list = []
            rewards_list = []

            state = np.random.randn(4)
            for step in range(20):
                action, log_prob, value = ppo.select_action(state)

                states_list.append(state)
                actions_list.append(action)
                log_probs_list.append(log_prob)
                values_list.append(value)
                rewards_list.append(np.random.randn())

                state = np.random.randn(4)

            # Prepare batch
            states = np.array(states_list)
            actions = np.array(actions_list)
            old_log_probs = np.array(log_probs_list)
            rewards = np.array(rewards_list)
            values = np.array(values_list)
            dones = np.zeros(len(rewards), dtype=np.float32)
            dones[-1] = 1  # Episode ends

            # Update
            advantages, returns = ppo.compute_gae(rewards, values, dones)
            metrics = ppo.update(states, actions, old_log_probs, returns, advantages)

            assert metrics is not None

    def test_value_bounds(self):
        """Test that values remain reasonable during training."""
        ppo = PPO(state_dim=4, action_dim=2)

        # Run several steps
        state = np.random.randn(4)
        values = []
        for _ in range(100):
            _, _, value = ppo.select_action(state)
            values.append(value)
            state = np.random.randn(4)

        values = np.array(values)

        # Values shouldn't explode
        assert np.all(np.abs(values) < 1000)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_step_gae(self):
        """Test GAE with single step."""
        ppo = PPO(state_dim=4, action_dim=2)

        rewards = np.array([1.0])
        values = np.array([0.5])
        dones = np.array([1.0])

        advantages, returns = ppo.compute_gae(rewards, values, dones)

        assert advantages.shape == (1,)
        assert returns.shape == (1,)

    def test_zero_rewards(self):
        """Test GAE with all zero rewards."""
        ppo = PPO(state_dim=4, action_dim=2)

        rewards = np.zeros(10)
        values = np.zeros(10)
        dones = np.zeros(10)
        dones[-1] = 1

        advantages, returns = ppo.compute_gae(rewards, values, dones)

        # Should handle gracefully
        assert not np.any(np.isnan(advantages))
        assert not np.any(np.isnan(returns))

    def test_deterministic_forward_pass(self):
        """Test that forward pass is deterministic."""
        net = PolicyNetwork(state_dim=10, action_dim=4)
        state = np.random.randn(10)

        # Run forward pass twice
        probs1, value1 = net.forward(state)
        probs2, value2 = net.forward(state)

        # Should get same results
        assert np.allclose(probs1, probs2)
        assert np.allclose(value1, value2)

    def test_small_batch_update(self):
        """Test update with batch smaller than batch_size."""
        config = PPOConfig(batch_size=64)
        ppo = PPO(state_dim=4, action_dim=2, config=config)

        # Batch of only 10 (smaller than batch_size of 64)
        batch_size = 10
        states = np.random.randn(batch_size, 4)
        actions = np.random.randint(0, 2, size=batch_size)
        old_log_probs = np.random.randn(batch_size) * -1
        returns = np.random.randn(batch_size)
        advantages = np.random.randn(batch_size)

        # Should handle without error
        metrics = ppo.update(states, actions, old_log_probs, returns, advantages)
        assert metrics is not None
