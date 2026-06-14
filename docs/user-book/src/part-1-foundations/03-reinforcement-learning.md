# Reinforcement Learning

Reinforcement learning (RL) is the study of how an **agent** learns to act in an
**environment** by trial and error, guided only by a scalar **reward** signal.
Unlike supervised learning, there is no labelled dataset — the agent must
discover good behaviour by interacting with the world and observing what happens.

## The Agent–Environment Loop

The basic setup, formalised by Sutton and Barto [[Sutton and Barto, 2018]](#bibliography), is:

```text
         action aₜ
  Agent ──────────────► Environment
    ▲                        │
    │   observation oₜ₊₁     │
    └────────────────────────┘
         reward rₜ₊₁
```

At each discrete timestep \\(t\\):

1. The agent observes the current state \\(s_t\\).
2. It selects an action \\(a_t\\) according to its **policy** \\(\pi(a \mid s)\\).
3. The environment transitions to a new state \\(s_{t+1}\\) and emits reward
   \\(r_{t+1}\\).
4. The agent updates its behaviour based on this experience.

The agent's goal is to maximise the **expected cumulative discounted reward**:

\\[
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}
\\]

where \\(\gamma \in [0, 1)\\) is a **discount factor** that makes immediate
rewards worth more than distant ones. When \\(\gamma = 0\\) the agent is
completely myopic; as \\(\gamma \to 1\\) the agent plans further into the future.

## Markov Decision Processes

The formal model underlying most RL is the **Markov Decision Process** (MDP),
a tuple \\((\mathcal{S}, \mathcal{A}, P, R, \gamma)\\):

- \\(\mathcal{S}\\) — state space
- \\(\mathcal{A}\\) — action space
- \\(P(s' \mid s, a)\\) — transition dynamics (probability of next state)
- \\(R(s, a)\\) — expected immediate reward
- \\(\gamma\\) — discount factor

The **Markov property** says that \\(P(s_{t+1} \mid s_t, a_t) = P(s_{t+1} \mid
s_0, a_0, \ldots, s_t, a_t)\\) — the future depends only on the current state,
not on the full history. This is an assumption, but one that holds exactly in
fully observed environments and approximately in many practical ones.

`rlevo`'s `Environment` trait encodes exactly this interface: `reset()` returns
the initial state, and `step(action)` returns the next observation and reward.

## Value Functions and the Bellman Equation

The **value function** \\(V^\pi(s)\\) is the expected return starting from state
\\(s\\) and following policy \\(\pi\\):

\\[
V^\pi(s) = \mathbb{E}_\pi\left[G_t \mid s_t = s\right]
\\]

The **action-value function** (Q-function) \\(Q^\pi(s, a)\\) gives the expected
return for taking action \\(a\\) in state \\(s\\) and then following \\(\pi\\):

\\[
Q^\pi(s, a) = \mathbb{E}_\pi\left[G_t \mid s_t = s, a_t = a\right]
\\]

Both satisfy a **Bellman equation** — a recursive consistency condition:

\\[
Q^\pi(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a)
             \sum_{a'} \pi(a' \mid s') Q^\pi(s', a')
\\]

The optimal Q-function \\(Q^*\\) satisfies the **Bellman optimality equation**:

\\[
Q^*(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a)
            \max_{a'} Q^*(s', a')
\\]

If you know \\(Q^*\\), the optimal policy is simply \\(\pi^*(s) = \arg\max_a
Q^*(s, a)\\) — always take the action with the highest Q-value.

## Tabular Methods: Q-Learning

When \\(\mathcal{S}\\) and \\(\mathcal{A}\\) are small and discrete, we can
represent \\(Q\\) as a table and update it directly. **Q-learning**
[[Watkins, 1989]](#bibliography) does this with the update rule:

\\[
Q(s_t, a_t) \leftarrow Q(s_t, a_t) +
  \alpha \left[ r_{t+1} + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t) \right]
\\]

where \\(\alpha \in (0, 1]\\) is a learning rate. Watkins and Dayan (1992) proved
that Q-learning converges to \\(Q^*\\) under mild conditions on the learning rate
schedule and exploration policy. Q-learning is an **off-policy** algorithm: it
updates toward the greedy policy regardless of which action was actually taken.

## Deep Reinforcement Learning

Tabular methods break down when the state space is large or continuous: a
CartPole observation is a 4-vector of real numbers, and a video game frame is
\\(84 \times 84 \times 4\\) pixels. The solution is to **approximate** \\(Q\\)
with a neural network \\(Q(s, a; \theta)\\) parameterised by weights \\(\theta\\).

**DQN** (Deep Q-Network) [[Mnih et al., 2015]](#bibliography) introduced two stabilising
ideas that made this work:

1. **Experience replay**: store transitions \\((s, a, r, s')\\) in a buffer and
   sample random mini-batches for training, breaking temporal correlation between
   updates.
2. **Target network**: maintain a separate, slowly-updated copy of the network
   \\(Q(s, a; \theta^-)\\) to compute the Bellman target, preventing the
   "moving target" instability where the network chases its own predictions.

DQN achieved human-level performance on 49 Atari games from raw pixels, a result
that launched the modern deep RL era.

Subsequent algorithms addressed DQN's limitations:

| Algorithm | Key contribution | Reference |
| --------- | ---------------- | --------- |
| Double DQN | Decoupled action selection and evaluation to reduce overestimation bias | [[van Hasselt et al., 2016]](#bibliography) |
| Dueling DQN | Separate value and advantage streams in the network | [[Wang et al., 2016]](#bibliography) |
| PPO | Stable policy gradient via clipped surrogate objective | [[Schulman et al., 2017]](#bibliography) |
| SAC | Off-policy actor-critic with maximum entropy objective | [[Haarnoja et al., 2018]](#bibliography) |
| TD3 | Twin critics + delayed policy updates for stability | [[Fujimoto et al., 2018]](#bibliography) |

## Policy Gradient Methods

An alternative to value-based methods is to directly optimise the policy. The
**policy gradient theorem** [[Sutton et al., 1999]](#bibliography) gives the gradient of the
expected return with respect to policy parameters \\(\theta\\):

\\[
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}
  \left[ \nabla_\theta \log \pi_\theta(a \mid s) \cdot Q^{\pi_\theta}(s, a) \right]
\\]

Williams (1992) [[Williams, 1992]](#bibliography) introduced the REINFORCE estimator
— a Monte Carlo approximation of this gradient — showing that the score function
\\(\nabla \log \pi\\) can be used to push the policy toward actions that produced
above-average returns, without needing a model of the environment.

Actor-critic methods (A2C, PPO, SAC) combine policy gradient with learned value
functions to reduce variance and improve sample efficiency.

> **Deeper reading.** Sutton and Barto, *Reinforcement Learning: An Introduction*
> (2nd ed., MIT Press, 2018) — freely available at
> [incompleteideas.net](http://incompleteideas.net/book/the-book-2nd.html) — is
> the definitive textbook. The DQN paper is Mnih et al. (2015), *Nature* 518,
> 529–533. For policy gradients, Williams (1992), *Machine Learning* 8, 229–256
> is the original; Schulman et al. (2015), arXiv:1502.05477 gives a unified
> treatment. The [Appendix B](../appendix-b-rl-algorithms/index.md) in this book
> gives pseudocode for DQN, PPO, and SAC as implemented in `rlevo`.

<a name="bibliography"></a>
*References on this page are collected in the [Bibliography](../bibliography.md).*
