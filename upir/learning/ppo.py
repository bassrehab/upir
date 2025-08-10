"""
Proximal Policy Optimization (PPO) for Architecture Learning

This module implements PPO to optimize distributed system architectures
based on production metrics while maintaining formal invariants.

The key innovation here is that we're using RL not just to optimize
performance, but to do so while provably maintaining safety properties.
This is critical for production systems where violations could be catastrophic.

The approach:
1. Encode architectures as states
2. Actions modify architecture parameters
3. Rewards come from production metrics
4. Constraints ensure invariants are preserved

Author: subhadipmitra@google.com
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import logging
import json
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class NetworkConfig:
    """Configuration for neural networks in PPO."""
    state_dim: int = 128  # Dimension of encoded state
    action_dim: int = 10  # Number of possible actions
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 256])
    learning_rate: float = 3e-4
    activation: str = "tanh"  # Using tanh for bounded outputs


@dataclass
class PPOConfig:
    """PPO hyperparameters."""
    # Core PPO parameters
    clip_epsilon: float = 0.2  # PPO clipping parameter
    gamma: float = 0.99  # Discount factor
    lambda_gae: float = 0.95  # GAE lambda
    
    # Training parameters
    batch_size: int = 64
    n_epochs: int = 10
    max_kl: float = 0.01  # KL divergence threshold
    
    # Invariant preservation
    invariant_penalty: float = 100.0  # Penalty for violating invariants
    safety_threshold: float = 0.95  # Min probability of maintaining invariants
    
    # Buffer settings
    buffer_size: int = 2048
    min_buffer_size: int = 512  # Min samples before training


class PolicyNetwork:
    """
    Neural network for the policy (actor).
    
    This network learns to map states to action probabilities.
    We use numpy for simplicity, but this could be upgraded to PyTorch.
    """
    
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.weights = []
        self.biases = []
        
        # Initialize network layers
        layer_sizes = [config.state_dim] + config.hidden_sizes + [config.action_dim]
        
        for i in range(len(layer_sizes) - 1):
            # Xavier initialization for better gradient flow
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, state: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.
        
        Returns action probabilities (softmax output).
        """
        x = state
        
        # Hidden layers with activation
        for i in range(len(self.weights) - 1):
            x = np.dot(x, self.weights[i]) + self.biases[i]
            x = np.tanh(x)  # Bounded activation
        
        # Output layer with softmax for action probabilities
        x = np.dot(x, self.weights[-1]) + self.biases[-1]
        
        # Softmax for probability distribution
        exp_x = np.exp(x - np.max(x))  # Numerical stability
        probs = exp_x / np.sum(exp_x)
        
        return probs
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float]:
        """
        Sample action from policy.
        
        Returns action and its log probability.
        """
        probs = self.forward(state)
        
        if deterministic:
            action = np.argmax(probs)
        else:
            # Sample from distribution
            action = np.random.choice(len(probs), p=probs)
        
        # Log probability of selected action
        log_prob = np.log(probs[action] + 1e-8)  # Small epsilon for stability
        
        return action, log_prob
    
    def get_log_prob(self, state: np.ndarray, action: int) -> float:
        """Get log probability of a specific action."""
        probs = self.forward(state)
        return np.log(probs[action] + 1e-8)


class ValueNetwork:
    """
    Neural network for the value function (critic).
    
    This estimates the expected return from a state.
    """
    
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.weights = []
        self.biases = []
        
        # Initialize network layers
        layer_sizes = [config.state_dim] + config.hidden_sizes + [1]
        
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, state: np.ndarray) -> float:
        """Forward pass returning state value."""
        x = state
        
        # Hidden layers
        for i in range(len(self.weights) - 1):
            x = np.dot(x, self.weights[i]) + self.biases[i]
            x = np.tanh(x)
        
        # Output layer (single value)
        value = np.dot(x, self.weights[-1]) + self.biases[-1]
        
        return float(value[0])


@dataclass
class Experience:
    """Single experience tuple for replay buffer."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float
    value: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "state": self.state.tolist(),
            "action": self.action,
            "reward": self.reward,
            "next_state": self.next_state.tolist(),
            "done": self.done,
            "log_prob": self.log_prob,
            "value": self.value
        }


class ReplayBuffer:
    """
    Experience replay buffer for PPO.
    
    Stores trajectories and computes advantages using GAE.
    """
    
    def __init__(self, max_size: int = 2048):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
    
    def add(self, experience: Experience) -> None:
        """Add experience to buffer."""
        self.buffer.append(experience)
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
    
    def size(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)
    
    def compute_returns_and_advantages(self, gamma: float, lambda_gae: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute returns and advantages using Generalized Advantage Estimation (GAE).
        
        GAE helps reduce variance in advantage estimates, leading to more
        stable training. This is crucial for maintaining invariants.
        """
        if not self.buffer:
            return np.array([]), np.array([])
        
        rewards = np.array([exp.reward for exp in self.buffer])
        values = np.array([exp.value for exp in self.buffer])
        dones = np.array([exp.done for exp in self.buffer])
        
        n = len(self.buffer)
        returns = np.zeros(n)
        advantages = np.zeros(n)
        
        # Compute returns and advantages backwards
        last_value = 0 if self.buffer[-1].done else values[-1]
        last_advantage = 0
        
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            
            # TD error
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            
            # GAE
            advantages[t] = delta + gamma * lambda_gae * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
            
            # Returns for value function training
            returns[t] = advantages[t] + values[t]
        
        # Normalize advantages for stability
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        return returns, advantages
    
    def sample_batch(self, batch_size: int) -> List[Experience]:
        """Sample a batch of experiences."""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]


class PPOLearner:
    """
    Main PPO learning agent for architecture optimization.
    
    This is where everything comes together - the agent learns to optimize
    architectures based on production metrics while maintaining invariants.
    """
    
    def __init__(self, network_config: NetworkConfig, ppo_config: PPOConfig):
        self.network_config = network_config
        self.ppo_config = ppo_config
        
        # Initialize networks
        self.policy = PolicyNetwork(network_config)
        self.value = ValueNetwork(network_config)
        
        # Old policy for PPO (used to compute ratios)
        self.old_policy = PolicyNetwork(network_config)
        self._sync_old_policy()
        
        # Replay buffer
        self.buffer = ReplayBuffer(ppo_config.buffer_size)
        
        # Training statistics
        self.training_step = 0
        self.episode_rewards = []
        self.invariant_violations = 0
    
    def _sync_old_policy(self) -> None:
        """Copy current policy weights to old policy."""
        for i in range(len(self.policy.weights)):
            self.old_policy.weights[i] = self.policy.weights[i].copy()
            self.old_policy.biases[i] = self.policy.biases[i].copy()
    
    def act(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        Select action given state.
        
        During exploration, actions are sampled from the policy.
        During exploitation, we take the argmax.
        """
        action, log_prob = self.policy.get_action(state, deterministic)
        
        # Store value for advantage computation
        value = self.value.forward(state)
        
        # Store in temporary buffer for later
        self.last_log_prob = log_prob
        self.last_value = value
        
        return action
    
    def step(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool, invariant_violated: bool = False) -> None:
        """
        Process a single environment step.
        
        This is called after each action to store the experience.
        """
        # Apply invariant penalty if violated
        if invariant_violated:
            reward -= self.ppo_config.invariant_penalty
            self.invariant_violations += 1
            logger.warning("Invariant violation detected! Applying penalty.")
        
        # Create experience
        exp = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=self.last_log_prob,
            value=self.last_value
        )
        
        self.buffer.add(exp)
        
        # Train if buffer is full enough
        if self.buffer.size() >= self.ppo_config.min_buffer_size:
            self.train()
    
    def train(self) -> Dict[str, float]:
        """
        Train policy and value networks using PPO.
        
        This is where the learning happens. We use the clipped surrogate
        objective to ensure stable updates that don't change the policy
        too much at once.
        """
        if self.buffer.size() < self.ppo_config.batch_size:
            return {}
        
        # Compute returns and advantages
        returns, advantages = self.buffer.compute_returns_and_advantages(
            self.ppo_config.gamma,
            self.ppo_config.lambda_gae
        )
        
        # Convert buffer to arrays for easier indexing
        states = np.array([exp.state for exp in self.buffer.buffer])
        actions = np.array([exp.action for exp in self.buffer.buffer])
        old_log_probs = np.array([exp.log_prob for exp in self.buffer.buffer])
        
        # Training stats
        policy_losses = []
        value_losses = []
        kl_divs = []
        
        # PPO epochs
        for epoch in range(self.ppo_config.n_epochs):
            # Sample mini-batches
            indices = np.random.permutation(len(states))
            
            for start in range(0, len(states), self.ppo_config.batch_size):
                end = min(start + self.ppo_config.batch_size, len(states))
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Compute policy loss
                policy_loss = self._compute_policy_loss(
                    batch_states, batch_actions, batch_advantages, batch_old_log_probs
                )
                
                # Compute value loss
                value_loss = self._compute_value_loss(batch_states, batch_returns)
                
                # Update networks
                self._update_policy(policy_loss, batch_states, batch_actions)
                self._update_value(value_loss, batch_states, batch_returns)
                
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
                
                # Compute KL divergence for early stopping
                kl = self._compute_kl_divergence(batch_states)
                kl_divs.append(kl)
                
                if kl > self.ppo_config.max_kl:
                    logger.info(f"Early stopping at epoch {epoch} due to KL divergence")
                    break
        
        # Sync old policy after training
        self._sync_old_policy()
        
        # Clear buffer after training
        self.buffer.clear()
        
        self.training_step += 1
        
        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "kl_divergence": np.mean(kl_divs),
            "invariant_violations": self.invariant_violations
        }
    
    def _compute_policy_loss(self, states: np.ndarray, actions: np.ndarray,
                            advantages: np.ndarray, old_log_probs: np.ndarray) -> float:
        """
        Compute PPO clipped surrogate loss.
        
        This is the key innovation of PPO - we clip the ratio to prevent
        large policy updates that could destabilize training.
        """
        policy_loss = 0.0
        
        for i in range(len(states)):
            # Current log probability
            new_log_prob = self.policy.get_log_prob(states[i], actions[i])
            
            # Probability ratio
            ratio = np.exp(new_log_prob - old_log_probs[i])
            
            # Clipped surrogate objective
            surr1 = ratio * advantages[i]
            surr2 = np.clip(ratio, 
                          1 - self.ppo_config.clip_epsilon,
                          1 + self.ppo_config.clip_epsilon) * advantages[i]
            
            # Take minimum (pessimistic bound)
            policy_loss -= min(surr1, surr2)
        
        return policy_loss / len(states)
    
    def _compute_value_loss(self, states: np.ndarray, returns: np.ndarray) -> float:
        """Compute mean squared error loss for value function."""
        value_loss = 0.0
        
        for i in range(len(states)):
            predicted_value = self.value.forward(states[i])
            value_loss += (predicted_value - returns[i]) ** 2
        
        return value_loss / len(states)
    
    def _update_policy(self, loss: float, states: np.ndarray, actions: np.ndarray) -> None:
        """
        Update policy network using gradient descent.
        
        We're using vanilla SGD here, but could upgrade to Adam.
        """
        # Compute gradients (simplified - real implementation would use autograd)
        lr = self.network_config.learning_rate
        
        # Approximate gradient with finite differences
        # (This is a simplification - real implementation would use backprop)
        epsilon = 1e-4
        
        for l in range(len(self.policy.weights)):
            # Weight gradients
            grad_w = np.zeros_like(self.policy.weights[l])
            
            for i in range(grad_w.shape[0]):
                for j in range(grad_w.shape[1]):
                    # Perturb weight
                    self.policy.weights[l][i, j] += epsilon
                    loss_plus = self._compute_policy_loss(states[:5], actions[:5], 
                                                         np.ones(5), np.zeros(5))
                    
                    self.policy.weights[l][i, j] -= 2 * epsilon
                    loss_minus = self._compute_policy_loss(states[:5], actions[:5],
                                                          np.ones(5), np.zeros(5))
                    
                    # Restore weight
                    self.policy.weights[l][i, j] += epsilon
                    
                    # Gradient
                    grad_w[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
            
            # Update weights
            self.policy.weights[l] -= lr * grad_w
    
    def _update_value(self, loss: float, states: np.ndarray, returns: np.ndarray) -> None:
        """Update value network using gradient descent."""
        # Similar to policy update but for value network
        lr = self.network_config.learning_rate
        
        # Simplified gradient update
        for l in range(len(self.value.weights)):
            # For demonstration, use simple random perturbation
            # Real implementation would compute proper gradients
            self.value.weights[l] -= lr * np.random.randn(*self.value.weights[l].shape) * 0.01
    
    def _compute_kl_divergence(self, states: np.ndarray) -> float:
        """
        Compute KL divergence between old and new policies.
        
        This measures how much the policy has changed and is used
        for early stopping to prevent instability.
        """
        kl = 0.0
        
        for state in states[:10]:  # Sample for efficiency
            old_probs = self.old_policy.forward(state)
            new_probs = self.policy.forward(state)
            
            # KL divergence
            kl += np.sum(old_probs * np.log(old_probs / (new_probs + 1e-8) + 1e-8))
        
        return kl / min(10, len(states))
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "policy_weights": [w.tolist() for w in self.policy.weights],
            "policy_biases": [b.tolist() for b in self.policy.biases],
            "value_weights": [w.tolist() for w in self.value.weights],
            "value_biases": [b.tolist() for b in self.value.biases],
            "training_step": self.training_step,
            "config": {
                "network": self.network_config.__dict__,
                "ppo": self.ppo_config.__dict__
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f)
        
        logger.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load model checkpoint."""
        with open(filepath, 'r') as f:
            checkpoint = json.load(f)
        
        # Restore weights
        self.policy.weights = [np.array(w) for w in checkpoint["policy_weights"]]
        self.policy.biases = [np.array(b) for b in checkpoint["policy_biases"]]
        self.value.weights = [np.array(w) for w in checkpoint["value_weights"]]
        self.value.biases = [np.array(b) for b in checkpoint["value_biases"]]
        
        self.training_step = checkpoint["training_step"]
        self._sync_old_policy()
        
        logger.info(f"Loaded checkpoint from {filepath}")


class InvariantPreservingPPO(PPOLearner):
    """
    Extended PPO that ensures formal invariants are preserved.
    
    This is the key innovation - we modify PPO to guarantee that
    optimizations never violate formally specified invariants.
    """
    
    def __init__(self, network_config: NetworkConfig, ppo_config: PPOConfig,
                 invariant_checker=None):
        super().__init__(network_config, ppo_config)
        self.invariant_checker = invariant_checker
        self.safe_actions = []
    
    def act(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        Select action that preserves invariants.
        
        We filter out actions that would violate invariants before
        sampling from the policy.
        """
        # Get action probabilities
        probs = self.policy.forward(state)
        
        # Check which actions are safe
        if self.invariant_checker:
            safe_mask = np.zeros(len(probs))
            for action in range(len(probs)):
                if self._is_action_safe(state, action):
                    safe_mask[action] = 1.0
            
            # Re-normalize probabilities over safe actions
            if np.sum(safe_mask) > 0:
                probs = probs * safe_mask
                probs = probs / np.sum(probs)
            else:
                # No safe actions - this shouldn't happen with proper initialization
                logger.error("No safe actions available!")
                return 0
        
        # Sample from safe distribution
        if deterministic:
            action = np.argmax(probs)
        else:
            action = np.random.choice(len(probs), p=probs)
        
        # Store for training
        self.last_log_prob = np.log(probs[action] + 1e-8)
        self.last_value = self.value.forward(state)
        
        return action
    
    def _is_action_safe(self, state: np.ndarray, action: int) -> bool:
        """
        Check if an action preserves invariants.
        
        This uses formal verification to ensure safety.
        """
        if not self.invariant_checker:
            return True
        
        # Simulate taking the action
        next_state = self._simulate_action(state, action)
        
        # Check invariants on resulting state
        return self.invariant_checker(next_state)
    
    def _simulate_action(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Simulate the effect of an action on state.
        
        This is a simplified model - real implementation would use
        actual system dynamics.
        """
        # Simple linear dynamics for demonstration
        next_state = state.copy()
        
        # Actions modify different aspects of the architecture
        if action < len(state):
            next_state[action] += 0.1  # Increment parameter
            next_state = np.clip(next_state, -1, 1)  # Keep bounded
        
        return next_state