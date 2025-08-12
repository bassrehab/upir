#!/usr/bin/env python3
"""
Run actual PPO training on a simplified UPIR optimization problem.

This demonstrates PPO learning to optimize distributed system parameters
while maintaining constraints.
"""

import sys
import numpy as np
import json
from pathlib import Path
sys.path.append('/Users/subhadipmitra/Dev/upir')

from upir.learning.ppo import PPOLearner, PPOConfig, NetworkConfig

class UPIREnvironment:
    """
    Simplified environment for UPIR system optimization.
    
    State: [batch_size, worker_count, queue_depth] (normalized)
    Actions: Adjust one parameter up/down
    Reward: Throughput - latency_penalty - constraint_violations
    """
    
    def __init__(self):
        self.state_dim = 3
        self.action_dim = 6  # +/- for each parameter
        self.reset()
        
    def reset(self):
        """Reset to random initial configuration."""
        self.batch_size = np.random.randint(10, 50)
        self.workers = np.random.randint(5, 20)
        self.queue_depth = np.random.randint(100, 500)
        self.steps = 0
        return self.get_state()
    
    def get_state(self):
        """Get normalized state vector."""
        return np.array([
            self.batch_size / 100.0,
            self.workers / 50.0,
            self.queue_depth / 1000.0
        ])
    
    def step(self, action):
        """Execute action and return (next_state, reward, done)."""
        # Apply action
        if action == 0:
            self.batch_size = min(100, self.batch_size + 5)
        elif action == 1:
            self.batch_size = max(1, self.batch_size - 5)
        elif action == 2:
            self.workers = min(50, self.workers + 2)
        elif action == 3:
            self.workers = max(1, self.workers - 2)
        elif action == 4:
            self.queue_depth = min(1000, self.queue_depth + 50)
        elif action == 5:
            self.queue_depth = max(10, self.queue_depth - 50)
        
        # Calculate reward
        throughput = self.batch_size * self.workers * 10  # Simplified
        latency = self.queue_depth / (self.workers * 10)  # Simplified
        
        # Constraint: batch_size * workers must be >= 200 for min throughput
        constraint_violation = max(0, 200 - self.batch_size * self.workers) * 0.1
        
        reward = throughput / 1000.0 - latency / 10.0 - constraint_violation
        
        self.steps += 1
        done = self.steps >= 50  # Episode length
        
        return self.get_state(), reward, done
    
    def get_metrics(self):
        """Get current performance metrics."""
        throughput = self.batch_size * self.workers * 10
        latency = self.queue_depth / (self.workers * 10)
        return {
            'batch_size': self.batch_size,
            'workers': self.workers,
            'queue_depth': self.queue_depth,
            'throughput': throughput,
            'latency': latency
        }


def train_ppo_on_upir():
    """Train PPO to optimize UPIR system parameters."""
    
    # Configure PPO (using defaults from PPOConfig)
    config = PPOConfig()
    
    network_config = NetworkConfig(
        state_dim=3,
        action_dim=6,
        hidden_sizes=[64, 64],
        activation='tanh'
    )
    
    # Initialize
    learner = PPOLearner(config, network_config)
    env = UPIREnvironment()
    
    # Training loop
    episodes = 100
    results = []
    
    print("Starting PPO training on UPIR optimization problem...")
    print("="*60)
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        while True:
            # Get action from policy
            action, log_prob = learner.policy.get_action(state)
            value = learner.value.forward(state)
            
            # Execute action
            next_state, reward, done = env.step(action)
            
            # Store experience
            learner.buffer.step(state, action, reward, next_state, done, log_prob, value)
            
            episode_reward += reward
            episode_steps += 1
            state = next_state
            
            if done:
                break
        
        # Train at end of episode
        if episode > 0 and episode % 10 == 0:
            train_metrics = learner.train()
            
        # Log progress
        if episode % 10 == 0:
            metrics = env.get_metrics()
            print(f"Episode {episode:3d} | Reward: {episode_reward:7.2f} | "
                  f"Throughput: {metrics['throughput']:5.0f} | "
                  f"Latency: {metrics['latency']:5.2f}ms | "
                  f"Config: B={metrics['batch_size']:2d} W={metrics['workers']:2d}")
            
            results.append({
                'episode': episode,
                'reward': float(episode_reward),
                'throughput': metrics['throughput'],
                'latency': metrics['latency'],
                'batch_size': metrics['batch_size'],
                'workers': metrics['workers'],
                'queue_depth': metrics['queue_depth']
            })
    
    # Save results
    output_path = Path(__file__).parent.parent / 'data' / 'actual_ppo_results.json'
    with open(output_path, 'w') as f:
        json.dump({
            'experiment': 'actual_ppo_training',
            'episodes': episodes,
            'results': results
        }, f, indent=2)
    
    print("="*60)
    print(f"PPO training complete! Results saved to {output_path}")
    
    # Compare first vs last episode
    first = results[0]
    last = results[-1]
    
    print(f"\nImprovement from episode 0 to {episodes-1}:")
    print(f"  Throughput: {first['throughput']:.0f} → {last['throughput']:.0f} "
          f"({(last['throughput']/first['throughput']-1)*100:.1f}% increase)")
    print(f"  Latency: {first['latency']:.2f}ms → {last['latency']:.2f}ms "
          f"({(1-last['latency']/first['latency'])*100:.1f}% reduction)")
    print(f"  Reward: {first['reward']:.2f} → {last['reward']:.2f}")


if __name__ == "__main__":
    train_ppo_on_upir()