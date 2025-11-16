# RL Optimizer

Reinforcement learning-based architecture optimization.

---

## Overview

The `ArchitectureLearner` uses PPO (Proximal Policy Optimization) to optimize architectures from production metrics.

---

## Class Documentation

::: upir.learning.learner.ArchitectureLearner
    options:
      show_source: true
      show_root_heading: true
      show_category_heading: true

---

## Usage Example

```python
from upir import UPIR
from upir.learning.learner import ArchitectureLearner

# Create learner
learner = ArchitectureLearner(
    upir,
    learning_rate=0.0003,
    gamma=0.99
)

# Simulate production metrics
metrics = {
    "latency_p99": 85.0,
    "monthly_cost": 4500.0,
    "throughput_qps": 12000.0
}

# Learn to optimize
optimized_upir = learner.learn(
    metrics,
    episodes=100,
    steps_per_episode=50
)

# Compare
print(f"Original cost: ${upir.architecture.total_cost}")
print(f"Optimized cost: ${optimized_upir.architecture.total_cost}")
```

---

## See Also

- [PPO](ppo.md) - PPO implementation
