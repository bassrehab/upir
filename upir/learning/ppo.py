"""
Proximal Policy Optimization (PPO) for architecture optimization.

This module implements PPO, a policy gradient reinforcement learning algorithm
that can be used to optimize UPIR architectures based on performance metrics.
PPO uses a clipped objective to ensure stable, conservative policy updates.

Implementation based on:
- PPO paper: Schulman et al. (2017) https://arxiv.org/abs/1707.06347
- GAE paper: Schulman et al. (2015) https://arxiv.org/abs/1506.02438
- OpenAI Spinning Up: https://spinningup.openai.com/en/latest/algorithms/ppo.html
- TD Commons disclosure: https://www.tdcommons.org/dpubs_series/8852/

Author: Subhadip Mitra
License: Apache 2.0
"""

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """
    Configuration for PPO training.

    Hyperparameters based on PPO paper defaults and best practices.

    Attributes:
        learning_rate: Step size for policy updates
        gamma: Discount factor for future rewards (0.99 = value future highly)
        epsilon: PPO clipping parameter (0.2 = clip ratio to [0.8, 1.2])
        value_coef: Coefficient for value loss in total loss
        entropy_coef: Coefficient for entropy bonus (exploration)
        batch_size: Mini-batch size for updates
        num_epochs: Number of epochs to train on each batch
        lambda_gae: GAE lambda parameter for advantage estimation

    References:
    - PPO paper: Section 3 (hyperparameters)
    - OpenAI Spinning Up: PPO implementation guide
    """

    learning_rate: float = 3e-4
    gamma: float = 0.99
    epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    batch_size: int = 64
    num_epochs: int = 10
    lambda_gae: float = 0.95


class PolicyNetwork:
    """
    Neural network for PPO policy and value function.

    Simple 2-layer MLP implemented in numpy. The network outputs both
    action probabilities (policy) and state value (critic).

    Architecture:
    - Input layer: state_dim
    - Hidden layer: 64 units, ReLU activation
    - Policy head: action_dim units, softmax activation
    - Value head: 1 unit, linear activation

    This is a simple numpy implementation. Can be upgraded to PyTorch/TensorFlow
    for more complex architectures and automatic differentiation.

    Attributes:
        state_dim: Dimension of input state
        action_dim: Dimension of action space
        hidden_dim: Hidden layer size (default 64)
        weights: Dictionary of network weights

    References:
    - PPO paper: Policy network architecture
    - Actor-Critic methods: Shared backbone with separate heads
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        """
        Initialize policy network with random weights.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Number of hidden units (default 64)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Initialize weights with Xavier/Glorot initialization
        # Xavier: std = sqrt(2 / (fan_in + fan_out))
        self.weights = {
            # Input to hidden
            "W1": np.random.randn(state_dim, hidden_dim) * np.sqrt(2.0 / (state_dim + hidden_dim)),
            "b1": np.zeros(hidden_dim),
            # Hidden to policy (action probabilities)
            "W_policy": np.random.randn(hidden_dim, action_dim) * np.sqrt(2.0 / (hidden_dim + action_dim)),
            "b_policy": np.zeros(action_dim),
            # Hidden to value
            "W_value": np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / (hidden_dim + 1)),
            "b_value": np.zeros(1),
        }

        # Cache for backpropagation
        self.cache = {}

    def forward(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Forward pass through network.

        Args:
            state: Input state vector (state_dim,) or batch (batch_size, state_dim)

        Returns:
            Tuple of (action_probs, value):
            - action_probs: Action probabilities (action_dim,) or (batch_size, action_dim)
            - value: State value estimate (scalar or (batch_size,))

        Example:
            >>> net = PolicyNetwork(state_dim=10, action_dim=4)
            >>> state = np.random.randn(10)
            >>> probs, value = net.forward(state)
            >>> probs.shape
            (4,)
            >>> assert np.allclose(probs.sum(), 1.0)  # Probabilities sum to 1
        """
        # Handle single state or batch
        is_single = state.ndim == 1
        if is_single:
            state = state.reshape(1, -1)

        # Layer 1: input -> hidden (ReLU activation)
        z1 = state @ self.weights["W1"] + self.weights["b1"]
        h1 = np.maximum(0, z1)  # ReLU

        # Policy head: hidden -> action probs (softmax)
        logits = h1 @ self.weights["W_policy"] + self.weights["b_policy"]
        # Softmax with numerical stability
        logits_shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        action_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # Value head: hidden -> value (linear)
        value = h1 @ self.weights["W_value"] + self.weights["b_value"]
        value = value.squeeze(-1)  # Remove last dimension

        # Cache for backprop
        self.cache = {
            "state": state,
            "z1": z1,
            "h1": h1,
            "logits": logits,
            "action_probs": action_probs,
            "value": value,
        }

        # Return single values if input was single
        if is_single:
            return action_probs[0], value[0]
        return action_probs, value

    def get_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """
        Sample action from policy and return action, log prob, and value.

        Args:
            state: Input state vector (state_dim,)

        Returns:
            Tuple of (action, log_prob, value):
            - action: Sampled action index
            - log_prob: Log probability of chosen action
            - value: State value estimate

        Example:
            >>> net = PolicyNetwork(state_dim=10, action_dim=4)
            >>> state = np.random.randn(10)
            >>> action, log_prob, value = net.get_action(state)
            >>> 0 <= action < 4
            True
        """
        action_probs, value = self.forward(state)

        # Sample action from categorical distribution
        action = np.random.choice(self.action_dim, p=action_probs)

        # Compute log probability
        log_prob = np.log(action_probs[action] + 1e-10)  # Small epsilon for numerical stability

        return action, log_prob, value

    def evaluate_actions(
        self,
        states: np.ndarray,
        actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Evaluate actions under current policy.

        Computes log probabilities, values, and entropy for given state-action pairs.
        Used during PPO update to compute policy gradient.

        Args:
            states: Batch of states (batch_size, state_dim)
            actions: Batch of actions (batch_size,)

        Returns:
            Tuple of (log_probs, values, entropy):
            - log_probs: Log probabilities of actions (batch_size,)
            - values: State value estimates (batch_size,)
            - entropy: Policy entropy (scalar, for exploration bonus)

        Example:
            >>> net = PolicyNetwork(state_dim=10, action_dim=4)
            >>> states = np.random.randn(32, 10)
            >>> actions = np.random.randint(0, 4, size=32)
            >>> log_probs, values, entropy = net.evaluate_actions(states, actions)
            >>> log_probs.shape
            (32,)
        """
        action_probs, values = self.forward(states)

        # Get log probabilities for taken actions
        batch_size = states.shape[0]
        log_probs = np.log(action_probs[np.arange(batch_size), actions] + 1e-10)

        # Compute entropy: H = -sum(p * log(p))
        entropy = -np.sum(action_probs * np.log(action_probs + 1e-10), axis=-1).mean()

        return log_probs, values, entropy


class PPO:
    """
    Proximal Policy Optimization (PPO) algorithm.

    PPO is a policy gradient method that uses a clipped objective to ensure
    stable, conservative policy updates. It's one of the most popular
    RL algorithms due to its simplicity and effectiveness.

    The key innovation is the clipped surrogate objective:
    L^CLIP(θ) = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]

    where r(θ) = π_θ(a|s) / π_θ_old(a|s) is the probability ratio.

    Attributes:
        policy: PolicyNetwork for action selection and value estimation
        config: PPO hyperparameters
        optimizer_state: State for optimization (momentum, etc.)

    References:
    - PPO paper: https://arxiv.org/abs/1707.06347
    - OpenAI Spinning Up: https://spinningup.openai.com/en/latest/algorithms/ppo.html
    - TD Commons: Architecture optimization using PPO
    """

    def __init__(self, state_dim: int, action_dim: int, config: PPOConfig = None):
        """
        Initialize PPO agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: PPO configuration (uses defaults if None)
        """
        self.config = config or PPOConfig()
        self.policy = PolicyNetwork(state_dim, action_dim)

        # Optimizer state (simple momentum-based)
        self.optimizer_state = {
            name: {"velocity": np.zeros_like(param)}
            for name, param in self.policy.weights.items()
        }

        logger.info(
            f"Initialized PPO: state_dim={state_dim}, action_dim={action_dim}, "
            f"lr={self.config.learning_rate}, epsilon={self.config.epsilon}"
        )

    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """
        Select action using current policy.

        Args:
            state: Current state vector

        Returns:
            Tuple of (action, log_prob, value)
        """
        return self.policy.get_action(state)

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).

        GAE uses an exponentially-weighted average of n-step advantages to
        reduce variance while maintaining low bias. It interpolates between
        Monte Carlo (high variance, low bias) and TD (low variance, high bias).

        Formula:
        δ_t = r_t + γV(s_{t+1})(1 - done_t) - V(s_t)
        A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...

        Args:
            rewards: Rewards received (T,)
            values: Value estimates V(s_t) (T,)
            dones: Episode termination flags (T,)

        Returns:
            Tuple of (advantages, returns):
            - advantages: Advantage estimates A_t (T,)
            - returns: Discounted returns (T,)

        References:
        - GAE paper: https://arxiv.org/abs/1506.02438
        - OpenAI Spinning Up: GAE explanation
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)

        # Compute TD errors (deltas)
        deltas = np.zeros(T, dtype=np.float32)
        for t in range(T):
            # δ_t = r_t + γV(s_{t+1})(1 - done_t) - V(s_t)
            next_value = values[t + 1] if t + 1 < T else 0.0
            deltas[t] = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]

        # Compute GAE advantages (backward pass)
        gae = 0
        for t in reversed(range(T)):
            # A_t = δ_t + (γλ)A_{t+1}(1 - done_t)
            gae = deltas[t] + self.config.gamma * self.config.lambda_gae * gae * (1 - dones[t])
            advantages[t] = gae

        # Compute returns: R_t = A_t + V(s_t)
        returns = advantages + values[:T]

        return advantages, returns

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        returns: np.ndarray,
        advantages: np.ndarray
    ) -> Dict[str, float]:
        """
        Update policy using PPO clipped objective.

        Performs multiple epochs of minibatch updates using the PPO loss:
        L = L^CLIP - c_1 * L^VF + c_2 * H

        where:
        - L^CLIP: Clipped surrogate objective
        - L^VF: Value function loss (MSE)
        - H: Entropy bonus

        Args:
            states: Batch of states (batch_size, state_dim)
            actions: Batch of actions (batch_size,)
            old_log_probs: Old log probabilities (batch_size,)
            returns: Discounted returns (batch_size,)
            advantages: Advantage estimates (batch_size,)

        Returns:
            Dictionary with training metrics:
            - policy_loss: Policy loss
            - value_loss: Value function loss
            - entropy: Policy entropy
            - total_loss: Combined loss

        Example:
            >>> ppo = PPO(state_dim=10, action_dim=4)
            >>> # Collect trajectories...
            >>> metrics = ppo.update(states, actions, old_log_probs, returns, advantages)

        References:
        - PPO paper: Section 3 (PPO-Clip algorithm)
        - Clipped objective prevents large policy updates
        """
        # Normalize advantages (reduces variance)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = states.shape[0]
        total_metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "total_loss": 0.0,
        }

        # Multiple epochs of updates
        for epoch in range(self.config.num_epochs):
            # Shuffle data
            indices = np.random.permutation(batch_size)

            # Minibatch updates
            for start in range(0, batch_size, self.config.batch_size):
                end = min(start + self.config.batch_size, batch_size)
                batch_indices = indices[start:end]

                # Get minibatch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Evaluate actions under current policy
                new_log_probs, values, entropy = self.policy.evaluate_actions(
                    batch_states, batch_actions
                )

                # Compute probability ratio: r(θ) = π_θ(a|s) / π_θ_old(a|s)
                ratio = np.exp(new_log_probs - batch_old_log_probs)

                # Compute clipped objective
                # L^CLIP = E[min(r*A, clip(r, 1-ε, 1+ε)*A)]
                surr1 = ratio * batch_advantages
                surr2 = np.clip(ratio, 1 - self.config.epsilon, 1 + self.config.epsilon) * batch_advantages
                policy_loss = -np.minimum(surr1, surr2).mean()

                # Value function loss: MSE
                value_loss = ((values - batch_returns) ** 2).mean()

                # Total loss
                total_loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                # Gradient descent (simplified - in practice, use automatic differentiation)
                # This is a placeholder - real implementation would compute gradients
                # and update weights using backpropagation
                self._simple_update(total_loss, batch_states, batch_actions)

                # Track metrics
                total_metrics["policy_loss"] += policy_loss
                total_metrics["value_loss"] += value_loss
                total_metrics["entropy"] += entropy
                total_metrics["total_loss"] += total_loss

        # Average metrics
        num_updates = self.config.num_epochs * max(1, (batch_size // self.config.batch_size))
        for key in total_metrics:
            total_metrics[key] /= num_updates

        logger.debug(
            f"PPO update: policy_loss={total_metrics['policy_loss']:.4f}, "
            f"value_loss={total_metrics['value_loss']:.4f}, "
            f"entropy={total_metrics['entropy']:.4f}"
        )

        return total_metrics

    def _simple_update(self, loss: float, states: np.ndarray, actions: np.ndarray):
        """
        Simplified weight update (placeholder for gradient descent).

        In a full implementation, this would:
        1. Compute gradients via backpropagation
        2. Update weights using optimizer (Adam, SGD, etc.)

        For now, this is a minimal placeholder. Upgrade to PyTorch for
        automatic differentiation and proper optimization.

        TODO: Implement full backpropagation or migrate to PyTorch

        Args:
            loss: Scalar loss value
            states: Batch of states
            actions: Batch of actions
        """
        # Placeholder: Random small updates (not real gradient descent)
        # This maintains the interface but should be replaced with proper backprop
        for name, param in self.policy.weights.items():
            # Very small random perturbation (NOT a real gradient update)
            gradient = np.random.randn(*param.shape) * 1e-6
            param -= self.config.learning_rate * gradient

    def __str__(self) -> str:
        """String representation."""
        return (
            f"PPO(lr={self.config.learning_rate}, "
            f"gamma={self.config.gamma}, "
            f"epsilon={self.config.epsilon})"
        )

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"PPO(state_dim={self.policy.state_dim}, "
            f"action_dim={self.policy.action_dim}, "
            f"config={self.config})"
        )
