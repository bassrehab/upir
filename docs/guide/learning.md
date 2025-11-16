# Learning & Optimization

Optimize architectures from production metrics using reinforcement learning.

---

## Overview

UPIR uses PPO (Proximal Policy Optimization) to learn from production metrics and optimize architectures.

---

## Quick Start

```python
from upir.learning.learner import ArchitectureLearner

learner = ArchitectureLearner(upir)
optimized_upir = learner.learn(production_metrics, episodes=100)

print(f"Original cost: ${upir.architecture.total_cost}")
print(f"Optimized cost: ${optimized_upir.architecture.total_cost}")
```

---

## See Also

- [RL Optimizer API](../api/learning/learner.md) - Complete API reference
- [PPO API](../api/learning/ppo.md) - PPO implementation
