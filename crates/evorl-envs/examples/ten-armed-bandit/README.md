# 10-Armed Bandit Training Example

A comprehensive implementation of three classical multi-armed bandit algorithms: **Epsilon-Greedy**, **Upper Confidence Bound (UCB)**, and **Thompson Sampling**.

## Table of Contents

- [Overview](#overview)
- [The Multi-Armed Bandit Problem](#the-multi-armed-bandit-problem)
- [Algorithms](#algorithms)
  - [Epsilon-Greedy](#1-epsilon-greedy-default)
  - [Upper Confidence Bound (UCB)](#2-upper-confidence-bound-ucb)
  - [Thompson Sampling](#3-thompson-sampling)
- [Running the Example](#running-the-example)
- [Understanding the Output](#understanding-the-output)
- [Performance Comparison](#performance-comparison)
- [Implementation Details](#implementation-details)
- [References](#references)

## Overview

The multi-armed bandit problem is one of the simplest yet most fundamental problems in reinforcement learning. It explores the **exploration-exploitation trade-off**: the dilemma between trying new actions to discover their values (exploration) versus choosing the best known action to maximize reward (exploitation).

## The Multi-Armed Bandit Problem

Imagine you're in a casino facing 10 slot machines (the "arms"). Each machine pays out according to its own probability distribution. Your goal is to maximize your total winnings over 1000 plays.

**The catch**: You don't know which machines are good! You must learn through trial and error.

### Problem Formalization

- **Action space**: 10 discrete actions (pulling one of 10 arms)
- **State space**: Stateless (the optimal action doesn't depend on history)
- **Reward distribution**: Each arm returns a reward sampled from N(q*(a), 1)
  - q*(a) is the true expected reward for arm a
  - q*(a) ~ N(0, 1) (sampled once at initialization)
  - The agent must discover which arm has the highest q*(a)

## Algorithms

### 1. Epsilon-Greedy (Default)

**The simplest approach to balancing exploration and exploitation.**

#### Strategy
With probability ε (epsilon), select a **random** action to explore. Otherwise, select the action with the **highest estimated value** to exploit.

#### Pseudocode
```
if random() < ε:
    action = random_arm()
else:
    action = argmax(Q)
```

#### Characteristics
- **Exploration rate**: ε = 0.1 (10% of the time)
- **Update rule**: Incremental sample averaging
  ```
  Q(a) ← Q(a) + (1/n)[R - Q(a)]
  ```
- **Pros**: Simple, effective, well-understood
- **Cons**: Explores uniformly without considering which arms need more information

#### Expected Performance
- Average reward: ~1.15
- Optimal action rate: ~80% after 1000 steps

### 2. Upper Confidence Bound (UCB)

**A principled approach that systematically explores uncertain actions.**

#### Strategy
Select the action with the highest **upper confidence bound**:

```
UCB(a) = Q(a) + c × √(ln(t) / N(a))
```

Where:
- `Q(a)` = estimated action value
- `c` = exploration parameter (default: 2.0)
- `t` = total number of steps
- `N(a)` = number of times action a was selected

#### Key Insight
The confidence term `c × √(ln(t) / N(a))` is:
- **Large** for rarely-selected actions → encourages exploration
- **Small** for frequently-selected actions → encourages exploitation
- **Grows** slowly over time → maintains some exploration

#### Characteristics
- **Exploration parameter**: c = 2.0
- **No randomness**: Deterministic action selection
- **Pros**: Better performance, principled exploration
- **Cons**: Requires tuning c, slightly more complex

#### Expected Performance
- Average reward: ~1.33
- Optimal action rate: ~91% after 1000 steps

### 3. Thompson Sampling

**A Bayesian approach using probability distributions over reward expectations.**

#### Strategy
1. Maintain a **Beta distribution** Beta(α, β) for each arm
2. At each step, **sample** from each distribution
3. Select the arm with the **highest sample**

#### Reward Mapping
For continuous rewards from N(q*, 1):
- Positive rewards → increment α (successes)
- Negative rewards → increment β (failures)
- Magnitude determines update strength

#### Characteristics
- **Prior**: Beta(1, 1) for all arms (uniform)
- **Update**: Bayesian posterior updates
- **Pros**: Natural exploration/exploitation balance, strong empirical performance
- **Cons**: More computational overhead, requires probability distributions

#### Expected Performance
- Average reward: ~1.22
- Optimal action rate: ~84% after 1000 steps

## Running the Example

### Default (Epsilon-Greedy)
```bash
cargo run --example ten_armed_bandit_training
```

### UCB (Best Performance)
```bash
cargo run --example ten_armed_bandit_training --features ucb
```

### Thompson Sampling
```bash
cargo run --example ten_armed_bandit_training --features thompson
```

## Understanding the Output

### Progress Updates
Every 100 steps, you'll see:
```
Step  300: Steps: 300, Avg Reward: 0.800, Optimal Action: 55.3% (166/300)
```

This means:
- **Steps**: 300 actions have been taken
- **Avg Reward**: Average reward per step is 0.800
- **Optimal Action**: 55.3% of actions were optimal (166 out of 300)

### Final Results

```
Final Statistics:
  Steps: 1000, Avg Reward: 1.145, Optimal Action: 79.7% (797/1000)

Learned Q-values:
  Arm | Estimated Value
  -----------------------
    0 |           0.477
    1 |           0.122
    ...
    7 |           1.479  ← This is the optimal arm!
    ...
    9 |          -1.957
```

**Key Metrics:**
- **Average Reward**: Higher is better (theoretical maximum ≈ q*(optimal_arm))
- **Optimal Action %**: Percentage of times the agent selected the best arm
- **Q-values**: The agent's learned estimates of each arm's value

## Performance Comparison

After 1000 steps on the same problem instance:

| Algorithm | Avg Reward | Optimal Action % | Characteristics |
|-----------|------------|------------------|-----------------|
| **Epsilon-Greedy** | 1.15 | 80% | Simple, good baseline |
| **UCB** | 1.33 | 91% | **Best performance** |
| **Thompson Sampling** | 1.22 | 84% | Bayesian, no tuning |

**Winner**: UCB provides the best exploration-exploitation balance for this problem.

## Implementation Details

### Code Structure

The example is organized into three main modules conditionally compiled based on features:

```rust
#[cfg(not(any(feature = "ucb", feature = "thompson")))]
mod agent { /* Epsilon-Greedy implementation */ }

#[cfg(feature = "ucb")]
mod agent { /* UCB implementation */ }

#[cfg(feature = "thompson")]
mod agent { /* Thompson Sampling implementation */ }
```

### Agent Traits

All three agents implement the same interface:

```rust
impl Agent {
    fn new(...) -> Self;
    fn select_action(&mut self) -> usize;
    fn update(&mut self, action: usize, reward: f32);
    fn q_values(&self) -> &[f32; 10];
}
```

### Environment

The environment is initialized with a fixed seed for reproducibility:
- Each arm's true mean q*(a) is sampled from N(0, 1) once
- Each reward is sampled from N(q*(a), 1) when that arm is pulled

## References

1. **Sutton, R. S., & Barto, A. G. (2018).** *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. Chapter 2: Multi-armed Bandits, pages 25-36.

2. **Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002).** Finite-time Analysis of the Multiarmed Bandit Problem. *Machine Learning*, 47(2-3), 235-256.

3. **Thompson, W. R. (1933).** On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. *Biometrika*, 25(3/4), 285-294.

4. **Russo, D., Van Roy, B., Kazerouni, A., Osband, I., & Wen, Z. (2018).** A Tutorial on Thompson Sampling. *Foundations and Trends in Machine Learning*, 11(1), 1-96.

---

**Happy Learning!** 🎰

Try running all three algorithms and observe how they balance exploration and exploitation differently!