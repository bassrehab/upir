"""
Architecture optimization using reinforcement learning.

This module implements ArchitectureLearner, which uses PPO to learn optimal
architectural configurations based on performance metrics and formal specifications.

Implementation based on:
- PPO paper: Schulman et al. (2017) https://arxiv.org/abs/1707.06347
- TD Commons disclosure: https://www.tdcommons.org/dpubs_series/8852/
- OpenAI Spinning Up: https://spinningup.openai.com/en/latest/

Author: Subhadip Mitra
License: Apache 2.0
"""

import copy
import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict

import numpy as np

from upir.core.specification import FormalSpecification
from upir.core.upir import UPIR
from upir.learning.ppo import PPO, PPOConfig

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """
    Single experience for replay buffer.

    Stores state-action-reward-next_state tuples for learning.

    Attributes:
        state: Encoded architecture state
        action: Action taken
        reward: Reward received
        log_prob: Log probability of action
        value: Value estimate
        done: Whether episode terminated
    """

    state: np.ndarray
    action: int
    reward: float
    log_prob: float
    value: float
    done: bool


class ArchitectureLearner:
    """
    Learn optimal architectures using PPO reinforcement learning.

    The learner encodes UPIR architectures as state vectors, uses PPO to
    select architectural modifications (actions), and learns from performance
    metrics (rewards) to optimize the architecture over time.

    State encoding extracts features like component count, latency, throughput.
    Actions represent architectural changes (adjust parallelism, change types, etc.).
    Rewards combine constraint satisfaction and performance improvements.

    Attributes:
        state_dim: Fixed dimension of state vectors
        action_dim: Number of possible actions
        ppo: PPO agent for policy learning
        experience_buffer: Replay buffer for experiences
        config: PPO configuration

    Example:
        >>> learner = ArchitectureLearner(state_dim=64, action_dim=40)
        >>> # After deploying architecture and collecting metrics:
        >>> optimized_upir = learner.learn_from_metrics(upir, metrics)

    References:
    - PPO: Policy gradient method with clipped objective
    - TD Commons: Architecture optimization approach
    - OpenAI Spinning Up: RL best practices
    """

    def __init__(
        self,
        state_dim: int = 64,
        action_dim: int = 40,
        config: PPOConfig = None,
        buffer_size: int = 1000
    ):
        """
        Initialize architecture learner.

        Args:
            state_dim: Dimension of state encoding (default 64)
            action_dim: Number of possible actions (default 40)
            config: PPO configuration (uses defaults if None)
            buffer_size: Maximum size of experience buffer
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or PPOConfig()

        # Initialize PPO agent
        self.ppo = PPO(state_dim=state_dim, action_dim=action_dim, config=self.config)

        # Experience buffer for training
        self.experience_buffer: Deque[Experience] = deque(maxlen=buffer_size)

        # Feature normalization stats (updated online)
        self.feature_stats = {
            "num_components_max": 100.0,
            "num_connections_max": 200.0,
            "avg_latency_max": 10000.0,  # 10 seconds in ms
            "total_throughput_max": 100000.0,  # 100k QPS
            "complexity_max": 1000.0,
        }

        logger.info(
            f"Initialized ArchitectureLearner: state_dim={state_dim}, "
            f"action_dim={action_dim}, buffer_size={buffer_size}"
        )

    def encode_state(self, upir: UPIR) -> np.ndarray:
        """
        Encode UPIR architecture as fixed-size state vector.

        Extracts architectural features and normalizes to [0, 1] range:
        - Number of components
        - Number of connections
        - Average component latency
        - Total throughput capacity
        - Deployment complexity score

        The state is padded to fixed size (state_dim) for consistent input.

        Args:
            upir: UPIR to encode

        Returns:
            State vector (state_dim,) with values in [0, 1]

        Example:
            >>> learner = ArchitectureLearner()
            >>> upir = UPIR(id="test", name="Test", description="Test")
            >>> state = learner.encode_state(upir)
            >>> state.shape
            (64,)
            >>> assert np.all((state >= 0) & (state <= 1))
        """
        if upir.architecture is None:
            # No architecture - return zero state
            return np.zeros(self.state_dim)

        arch = upir.architecture

        # Extract basic features
        num_components = len(arch.components)
        num_connections = len(arch.connections)

        # Compute average latency (if components have latency info)
        total_latency = 0.0
        latency_count = 0
        for comp in arch.components:
            if isinstance(comp, dict) and "latency_ms" in comp:
                total_latency += comp["latency_ms"]
                latency_count += 1
        avg_latency = total_latency / latency_count if latency_count > 0 else 0.0

        # Compute total throughput (if components have throughput info)
        total_throughput = 0.0
        for comp in arch.components:
            if isinstance(comp, dict) and "throughput_qps" in comp:
                total_throughput += comp["throughput_qps"]

        # Compute deployment complexity (simple heuristic)
        # Based on number of components and connections
        complexity = num_components * 10 + num_connections * 5

        # Normalize features to [0, 1]
        features = np.array([
            num_components / self.feature_stats["num_components_max"],
            num_connections / self.feature_stats["num_connections_max"],
            avg_latency / self.feature_stats["avg_latency_max"],
            total_throughput / self.feature_stats["total_throughput_max"],
            complexity / self.feature_stats["complexity_max"],
        ])

        # Clip to [0, 1] range
        features = np.clip(features, 0.0, 1.0)

        # Add per-component features (latency, throughput, parallelism)
        component_features = []
        for comp in arch.components[:10]:  # Max 10 components
            if isinstance(comp, dict):
                comp_latency = comp.get("latency_ms", 0.0)
                comp_throughput = comp.get("throughput_qps", 0.0)
                comp_parallelism = comp.get("parallelism", 1)

                component_features.extend([
                    comp_latency / self.feature_stats["avg_latency_max"],
                    comp_throughput / self.feature_stats["total_throughput_max"],
                    comp_parallelism / 100.0,  # Normalize to [0, 1] assuming max 100
                ])

        # Pad to fixed size
        all_features = np.concatenate([features, component_features])

        # Clip all features to [0, 1] range
        all_features = np.clip(all_features, 0.0, 1.0)
        if len(all_features) < self.state_dim:
            # Pad with zeros
            all_features = np.pad(
                all_features,
                (0, self.state_dim - len(all_features)),
                mode="constant"
            )
        else:
            # Truncate if too long
            all_features = all_features[:self.state_dim]

        return all_features.astype(np.float32)

    def decode_action(self, action: int, upir: UPIR) -> UPIR:
        """
        Decode action and apply architectural modification.

        Actions represent different architectural changes:
        - 0-9: Increase parallelism of component i
        - 10-19: Decrease parallelism of component i
        - 20-29: Change component type (e.g., batch -> streaming)
        - 30-39: Modify connection (add/remove)

        Args:
            action: Action index (0 to action_dim-1)
            upir: Current UPIR

        Returns:
            Modified UPIR with architectural change applied

        Example:
            >>> learner = ArchitectureLearner()
            >>> upir = UPIR(id="test", name="Test", description="Test")
            >>> modified = learner.decode_action(0, upir)
        """
        # Create copy to avoid modifying original
        modified_upir = copy.deepcopy(upir)

        if modified_upir.architecture is None:
            logger.warning("No architecture to modify")
            return modified_upir

        arch = modified_upir.architecture
        num_components = len(arch.components)

        if num_components == 0:
            logger.warning("No components to modify")
            return modified_upir

        # Decode action type and target
        if action < 10:
            # Increase parallelism of component (action % num_components)
            component_idx = action % num_components
            comp = arch.components[component_idx]
            current_parallelism = comp.get("parallelism", 1)
            new_parallelism = min(current_parallelism + 1, 100)
            comp["parallelism"] = new_parallelism
            logger.debug(
                f"Action {action}: Increase parallelism of {comp.get('name', f'comp_{component_idx}')} "
                f"from {current_parallelism} to {new_parallelism}"
            )

        elif action < 20:
            # Decrease parallelism of component
            component_idx = (action - 10) % num_components
            comp = arch.components[component_idx]
            current_parallelism = comp.get("parallelism", 1)
            new_parallelism = max(current_parallelism - 1, 1)
            comp["parallelism"] = new_parallelism
            logger.debug(
                f"Action {action}: Decrease parallelism of {comp.get('name', f'comp_{component_idx}')} "
                f"from {current_parallelism} to {new_parallelism}"
            )

        elif action < 30:
            # Change component type (simplified: toggle between batch/streaming)
            component_idx = (action - 20) % num_components
            comp = arch.components[component_idx]
            current_type = comp.get("type", "processor")
            if "streaming" in current_type.lower():
                new_type = "batch_processor"
            else:
                new_type = "streaming_processor"
            comp["type"] = new_type
            logger.debug(
                f"Action {action}: Change type of {comp.get('name', f'comp_{component_idx}')} "
                f"from {current_type} to {new_type}"
            )

        else:
            # Modify connection (simplified: toggle connection property)
            if len(arch.connections) > 0:
                connection_idx = (action - 30) % len(arch.connections)
                conn = arch.connections[connection_idx]
                # Toggle "batched" property
                current_batched = conn.get("batched", False)
                conn["batched"] = not current_batched
                logger.debug(
                    f"Action {action}: Toggle batched for connection "
                    f"{conn.get('from', '?')}->{conn.get('to', '?')} to {not current_batched}"
                )

        return modified_upir

    def compute_reward(
        self,
        metrics: Dict[str, float],
        spec: FormalSpecification,
        previous_metrics: Dict[str, float] = None
    ) -> float:
        """
        Compute reward from performance metrics and specification.

        Reward structure:
        - Base reward: 1.0
        - Constraint satisfaction: +0.1 per met constraint, -0.5 per violation
        - Performance improvement: +delta/target for improvements

        Rewards are clipped to [-1, 1] range for stability.

        Args:
            metrics: Current performance metrics
                - latency_p99: 99th percentile latency (ms)
                - throughput_qps: Queries per second
                - error_rate: Error rate (0-1)
                - cost: Deployment cost
            spec: Formal specification with constraints
            previous_metrics: Previous metrics for computing deltas

        Returns:
            Reward in [-1, 1]

        Example:
            >>> learner = ArchitectureLearner()
            >>> metrics = {"latency_p99": 100, "throughput_qps": 1000}
            >>> spec = FormalSpecification()
            >>> reward = learner.compute_reward(metrics, spec)
        """
        reward = 1.0  # Base reward

        # Check constraint satisfaction
        constraints_met = 0
        constraints_violated = 0

        # Check temporal properties for latency constraints
        for prop in spec.properties + spec.invariants:
            if prop.time_bound is not None:
                # Latency constraint
                target_latency = prop.time_bound
                actual_latency = metrics.get("latency_p99", float("inf"))

                if actual_latency <= target_latency:
                    constraints_met += 1
                    reward += 0.1
                else:
                    constraints_violated += 1
                    reward -= 0.5

        # Check throughput requirements (heuristic from predicates)
        has_throughput_req = any(
            "throughput" in prop.predicate.lower()
            for prop in spec.properties + spec.invariants
        )
        if has_throughput_req:
            # Assume target throughput of 1000 QPS (could be extracted from spec)
            target_throughput = 1000.0
            actual_throughput = metrics.get("throughput_qps", 0.0)

            if actual_throughput >= target_throughput:
                constraints_met += 1
                reward += 0.1
            else:
                constraints_violated += 1
                reward -= 0.5

        # Reward performance improvements over previous metrics
        if previous_metrics is not None:
            # Latency reduction (lower is better)
            prev_latency = previous_metrics.get("latency_p99", 0.0)
            curr_latency = metrics.get("latency_p99", 0.0)
            if prev_latency > 0:
                latency_delta = (prev_latency - curr_latency) / prev_latency
                reward += latency_delta  # Positive if latency reduced

            # Throughput increase (higher is better)
            prev_throughput = previous_metrics.get("throughput_qps", 0.0)
            curr_throughput = metrics.get("throughput_qps", 0.0)
            if prev_throughput > 0:
                throughput_delta = (curr_throughput - prev_throughput) / prev_throughput
                reward += throughput_delta  # Positive if throughput increased

        # Penalty for high error rate
        error_rate = metrics.get("error_rate", 0.0)
        if error_rate > 0.01:  # More than 1% errors
            reward -= error_rate * 10  # Heavy penalty

        # Clip reward to [-1, 1]
        reward = np.clip(reward, -1.0, 1.0)

        logger.debug(
            f"Computed reward: {reward:.3f} "
            f"(constraints_met={constraints_met}, violated={constraints_violated})"
        )

        return reward

    def learn_from_metrics(
        self,
        upir: UPIR,
        metrics: Dict[str, float],
        previous_metrics: Dict[str, float] = None
    ) -> UPIR:
        """
        Learn from performance metrics and return optimized architecture.

        Main entry point for architecture optimization. This method:
        1. Encodes current architecture as state
        2. Selects action using PPO policy
        3. Decodes action to modify architecture
        4. Computes reward from metrics
        5. Stores experience for learning
        6. Updates PPO policy when buffer is full
        7. Returns optimized architecture

        Args:
            upir: Current UPIR
            metrics: Performance metrics from deployment
            previous_metrics: Previous metrics for delta computation

        Returns:
            Optimized UPIR with modified architecture

        Example:
            >>> learner = ArchitectureLearner()
            >>> upir = UPIR(id="test", name="Test", description="Test")
            >>> metrics = {"latency_p99": 100, "throughput_qps": 1000}
            >>> optimized = learner.learn_from_metrics(upir, metrics)

        References:
        - PPO: Policy gradient with experience replay
        - TD Commons: Continuous optimization loop
        """
        # 1. Encode current architecture as state
        state = self.encode_state(upir)

        # 2. Select action using PPO policy
        action, log_prob, value = self.ppo.select_action(state)

        # 3. Decode action to modify architecture
        optimized_upir = self.decode_action(action, upir)

        # 4. Compute reward from metrics
        reward = self.compute_reward(
            metrics,
            upir.specification if upir.specification else FormalSpecification(),
            previous_metrics
        )

        # 5. Store experience
        # Check if this is terminal (could be based on convergence criteria)
        done = False  # For now, never terminal (continuous learning)

        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            log_prob=log_prob,
            value=value,
            done=done
        )
        self.experience_buffer.append(experience)

        # 6. Update policy if we have enough experiences
        if len(self.experience_buffer) >= self.config.batch_size:
            self._update_policy()

        logger.info(
            f"Learning step: action={action}, reward={reward:.3f}, "
            f"buffer_size={len(self.experience_buffer)}"
        )

        return optimized_upir

    def _update_policy(self):
        """
        Update PPO policy using experiences from buffer.

        Extracts states, actions, rewards from buffer, computes advantages
        using GAE, and performs PPO update.
        """
        # Extract experiences
        states = np.array([exp.state for exp in self.experience_buffer])
        actions = np.array([exp.action for exp in self.experience_buffer])
        old_log_probs = np.array([exp.log_prob for exp in self.experience_buffer])
        rewards = np.array([exp.reward for exp in self.experience_buffer])
        values = np.array([exp.value for exp in self.experience_buffer])
        dones = np.array([exp.done for exp in self.experience_buffer], dtype=np.float32)

        # Compute advantages using GAE
        advantages, returns = self.ppo.compute_gae(rewards, values, dones)

        # Update PPO policy
        metrics = self.ppo.update(states, actions, old_log_probs, returns, advantages)

        logger.info(
            f"Policy update: policy_loss={metrics['policy_loss']:.4f}, "
            f"value_loss={metrics['value_loss']:.4f}, "
            f"entropy={metrics['entropy']:.4f}"
        )

        # Clear buffer after update
        self.experience_buffer.clear()

    def __str__(self) -> str:
        """String representation."""
        return (
            f"ArchitectureLearner(state_dim={self.state_dim}, "
            f"action_dim={self.action_dim}, "
            f"buffer={len(self.experience_buffer)}/{self.experience_buffer.maxlen})"
        )

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"ArchitectureLearner(state_dim={self.state_dim}, "
            f"action_dim={self.action_dim}, "
            f"config={self.config})"
        )
