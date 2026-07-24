# Reinforcement Learning

The evolutionary chapter scored a candidate all at once: hand a genome to the
fitness function, get one number back, done. Reinforcement learning changes the
rules of the game in a way that turns out to matter enormously. Here the score
arrives *drip by drip* — one scalar **reward** per timestep — and the agent must
commit to an action long before it learns whether that action was wise. A move
that looks brilliant now may doom you twenty steps later; a sacrifice now may pay
off only at the end of the episode.

That delay is the whole difficulty, and it has a name: the **credit assignment
problem**. When a reward finally lands, which of the many actions that led here
deserves the credit — or the blame? Hold onto that question, because every idea
in this chapter is a different machine for answering it. Value functions cache
credit so it can be looked up. The Bellman equation flows credit backward one
step at a time. Q-learning, DQN, and the policy-gradient methods are each a
particular way of estimating and propagating that credit at scale. This is also
the *gradient* half of `rlevo`'s thesis — the counterpart to the population path
of the last two chapters, and the side that learns from the fine structure of a
single trajectory rather than the coarse verdict on a whole one.

**Reinforcement learning** (RL), then, is the study of how an **agent** learns to
act in an **environment** by trial and error, guided only by that scalar reward.
Unlike supervised learning, there is no labelled dataset telling it the right
answer — the agent has to discover good behaviour by interacting with the world
and watching what comes back.

## The Agent–Environment Loop

Everything starts from one loop. The agent acts, the environment responds with an
observation and a reward, and the cycle repeats — the setup Sutton and Barto
formalised [[Sutton and Barto, 2018]](#bibliography):

```text
         action aₜ
  Agent ──────────────► Environment
    ▲                        │
    │   observation oₜ₊₁     │
    └────────────────────────┘
         reward rₜ₊₁
```

Unrolled, each discrete timestep \\(t\\) does four things:

1. The agent observes the current state \\(s_t\\).
2. It selects an action \\(a_t\\) according to its **policy** \\(\pi(a \mid s)\\).
3. The environment transitions to a new state \\(s_{t+1}\\) and emits reward
   \\(r_{t+1}\\).
4. The agent updates its behaviour based on this experience.

Notice what the agent is *not* handed: a per-step verdict on whether \\(a_t\\) was
good. All it gets is \\(r_{t+1}\\), which conflates the action just taken with the
consequences of everything before it. So we are careful about what "good"
means — the goal is not to grab the largest immediate reward but to maximise the
**expected cumulative discounted reward**, the *return*:

```math
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}
```

where \\(\gamma \in [0, 1)\\) is a **discount factor**. Read \\(\gamma\\) as a
credit-assignment dial: it sets how far into the future a present action is held
responsible for. At \\(\gamma = 0\\) the agent is completely myopic — it credits
each action with only the very next reward; as \\(\gamma \to 1\\) credit reaches
ever further ahead and the agent plans deep into the future.

> **In `rlevo`.** The reward signal is captured by the `Reward` trait in
> `rlevo::core::base`. It requires only that rewards form an additive monoid
> and can be converted to `f32` — exactly enough to accumulate the return
> \\(G_t\\) above and log it:
>
> ```rust
> pub trait Reward: Clone + Add<Output = Self> + Into<f32> + Debug {
>     fn zero() -> Self;   // additive identity
> }
> ```
>
> The built-in `ScalarReward(f32)` satisfies this for most environments. You
> only need a custom implementation when your reward has domain-specific
> structure — say a multi-component tuple you want to keep typed before
> scalarising it.

## Markov Decision Processes

Stated plainly, credit assignment over an arbitrarily long history is hopeless —
if the future could depend on *everything* that ever happened, there would be no
way to learn from finite experience. RL escapes this with one assumption that
makes the whole enterprise tractable. We model the world as a **Markov Decision
Process** (MDP), the tuple \\((\mathcal{S}, \mathcal{A}, P, R, \gamma)\\):

- \\(\mathcal{S}\\) — state space
- \\(\mathcal{A}\\) — action space
- \\(P(s' \mid s, a)\\) — transition dynamics (probability of the next state)
- \\(R(s, a)\\) — expected immediate reward
- \\(\gamma\\) — discount factor

> **In `rlevo`.** The \\(\mathcal{S}\\) and \\(\mathcal{A}\\) components map to
> two const-generic traits, where the const parameter `R` is the *rank* (number
> of axes) of the underlying tensor — `1` for a flat vector, `3` for an image:
>
> ```rust
> pub trait State<const R: usize>: Debug + Clone + Send + Sync {
>     fn shape() -> [usize; R];      // cardinality of each axis
>     fn is_valid(&self) -> bool;
> }
>
> pub trait Action<const R: usize>: Debug + Clone + Sized {
>     fn shape() -> [usize; R];
>     fn is_valid(&self) -> bool;
> }
> ```
>
> We split `State` from `Observation` on purpose, and the reason is exactly the
> credit-assignment story: a `State` holds the *full* information the Markov
> property needs, while an `Observation` is only what the agent *actually sees*,
> which may be partial. Producing one from the other is the environment's job —
> a dedicated [`Sensor`](https://docs.rs/rlevo-core/latest/rlevo_core/environment/trait.Sensor.html)
> trait, detailed in [State and Observation Spaces](reinforcement-learning/31-state.md)
> — not something a `State` value computes for itself. A fully observable
> environment's sensor hands back an observation that is just a flattened view of
> the state; a partially observable one projects out only the visible features.

The assumption itself is the **Markov property**: \\(P(s_{t+1} \mid s_t, a_t) =
P(s_{t+1} \mid s_0, a_0, \ldots, s_t, a_t)\\) — the future depends only on the
*current* state, not the full history. That is the lever. If the present state
summarises the past, then assigning credit no longer needs the whole trajectory;
the agent can reason one step at a time and let the recursion of the next section
carry credit backward. The property holds exactly in fully observed environments
and approximately in many practical ones — and where the agent's observation
hides part of the true state, the property fails *from its vantage point*, which
is precisely the partial-observability case the `State`/`Observation` split above
makes explicit.

`rlevo`'s `Environment` trait encodes this interface directly. `reset()` returns
an initial `Snapshot` carrying the first `Observation`, and `step(action)`
returns a `Snapshot` with the next `Observation`, the `Reward`, and a
`terminated` flag. The full state — the part that has to satisfy the Markov
property internally — lives inside the environment struct and is never exposed to
the agent; only the observation the environment's `Sensor` produces ever crosses
the boundary. That boundary keeps the agent honest about what it can actually
know, so an agent you train cannot accidentally cheat by reading state it would
not have at deployment.

## Value Functions and the Bellman Equation

Here is the central machine. Rather than reason about credit from scratch every
time, the agent learns to *cache* it. The **value function** \\(V^\pi(s)\\) is the
expected return from state \\(s\\) when following policy \\(\pi\\) — credit,
pre-computed and stored against a state:

```math
V^\pi(s) = \mathbb{E}_\pi\left[G_t \mid s_t = s\right]
```

The **action-value function** (Q-function) \\(Q^\pi(s, a)\\) is finer-grained:
the expected return for taking action \\(a\\) in state \\(s\\) and following
\\(\pi\\) thereafter. This is the quantity we most want, because it scores
*actions* — exactly the thing credit assignment is about:

```math
Q^\pi(s, a) = \mathbb{E}_\pi\left[G_t \mid s_t = s, a_t = a\right]
```

What makes value functions learnable rather than merely definable is that they
are self-consistent across a single step. Both satisfy a **Bellman equation** — a
recursive condition tying the value here to the value one step ahead:

```math
Q^\pi(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a)
             \sum_{a'} \pi(a' \mid s') Q^\pi(s', a')
```

This recursion *is* credit assignment made mechanical: the value of acting now is
the immediate reward plus the discounted value of where you land, so credit flows
backward one link at a time instead of waiting for the episode to end. The best
possible Q-function, \\(Q^*\\), satisfies the **Bellman optimality equation**,
which simply replaces "average over the policy's next actions" with "assume the
best next action":

```math
Q^*(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a)
            \max_{a'} Q^*(s', a')
```

And once you hold \\(Q^*\\), acting optimally is trivial — there is nothing left
to plan. The optimal policy just reads it off greedily, \\(\pi^*(s) = \arg\max_a
Q^*(s, a)\\): in every state, take the action with the highest Q-value. All the
difficulty, then, collapses into *estimating* \\(Q^*\\) from experience, and that
is what the remaining methods do.

## Tabular Methods: Q-Learning

When \\(\mathcal{S}\\) and \\(\mathcal{A}\\) are small and discrete, the simplest
estimator is the most literal one: store \\(Q\\) as a table — one cell per
(state, action) — and nudge each cell toward its own Bellman target as experience
arrives. **Q-learning** [[Watkins, 1989]](#bibliography) does exactly that:

```math
Q(s_t, a_t) \leftarrow Q(s_t, a_t) +
  \alpha \left[ r_{t+1} + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t) \right]
```

Read the bracket as *surprise*: it is the one-step bootstrapped target
\\(r_{t+1} + \gamma \max_a Q(s_{t+1}, a)\\) minus what we currently believe,
\\(Q(s_t, a_t)\\). This **temporal-difference error** is the unit of credit being
assigned, and \\(\alpha \in (0, 1]\\) sets how hard each new surprise moves the
estimate. Watkins and Dayan (1992) proved this converges to \\(Q^*\\) under mild
conditions on the learning-rate schedule and exploration. Q-learning is also
**off-policy**: the \\(\max\\) means it updates toward the *greedy* policy no
matter which action the agent actually took to explore — it can learn the optimal
value while behaving sub-optimally to gather experience.

<!-- todo! rlevo::envs::Environment walk-through and tutorial -->

## Deep Reinforcement Learning

A table works only as long as you can enumerate states, and that ceiling is low.
A CartPole observation is a 4-vector of real numbers — already uncountable; an
Atari frame is \\(84 \times 84 \times 4\\) pixels. There is no cell to look up,
and even if there were, the agent could never visit them all. The escape is a
second idea layered on top of the first: stop storing \\(Q\\) and start
*generalising* it. We **approximate** \\(Q\\) with a neural network \\(Q(s, a;
\theta)\\) of weights \\(\theta\\), so that experience in one state teaches the
agent about the unseen states around it.

The catch is that bootstrapping and function approximation do not naturally get
along — the network is now chasing a target computed from its own
ever-shifting weights, and the whole thing can spiral. **DQN** (Deep Q-Network)
[[Mnih et al., 2015]](#bibliography) is, at heart, two tricks for keeping that
bootstrap stable:

1. **Experience replay** stores transitions \\((s, a, r, s')\\) in a buffer and
   trains on random mini-batches drawn from it. Sampling out of order breaks the
   temporal correlation between consecutive updates, which would otherwise bias
   the network toward whatever it saw most recently.
2. **A target network** keeps a separate, slowly-updated copy of the weights,
   \\(Q(s, a; \theta^-)\\), to compute the Bellman target. Freezing the target
   for a while stops the "moving target" instability where the network forever
   chases its own predictions. Mnih et al. copied \\(\theta\\) into
   \\(\theta^-\\) wholesale every \\(C\\) parameter updates; later work blends
   instead, \\(\theta^- \leftarrow \tau\theta + (1 - \tau)\theta^-\\). We treat
   those as one mechanism rather than two, and give it one type,
   `TargetUpdate`: its cadence decides *when* an update fires and its
   \\(\tau\\) decides *how far* the target moves, so \\(\tau = 1.0\\) recovers
   the original hard copy as a degenerate case of the blend. Every off-policy
   agent in `rlevo` — DQN, C51, QR-DQN, DDPG, TD3, SAC — configures its target
   through that single field, and the cadence counts gradient updates, not
   environment steps.

With those two stabilisers, DQN reached human-level play on 49 Atari games from
raw pixels — the result that launched the modern deep-RL era. Everything since
has chipped away at its limitations:

| Algorithm | Key contribution | Reference |
| --------- | ---------------- | --------- |
| Double DQN | Decouples action selection from evaluation to curb overestimation bias | [[van Hasselt et al., 2016]](#bibliography) |
| Dueling DQN | Splits the network into separate value and advantage streams | [[Wang et al., 2016]](#bibliography) |
| PPO | Stable policy gradient via a clipped surrogate objective | [[Schulman et al., 2017]](#bibliography) |
| SAC | Off-policy actor-critic with a maximum-entropy objective | [[Haarnoja et al., 2018]](#bibliography) |
| TD3 | Twin critics plus delayed policy updates for stability | [[Fujimoto et al., 2018]](#bibliography) |

## Policy Gradient Methods

Everything so far assigns credit *through a value function*: learn \\(Q\\), then
act greedily. There is a second route that skips the middleman entirely — adjust
the policy *directly*, pushing up the probability of actions that turned out
better than expected. Instead of asking "what is each action worth?" it asks
"which way should I tilt my parameters to earn more return?" The **policy gradient
theorem** [[Sutton et al., 1999]](#bibliography) gives that direction with respect
to the policy parameters \\(\theta\\):

```math
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}
  \left[ \nabla_\theta \log \pi_\theta(a \mid s) \cdot Q^{\pi_\theta}(s, a) \right]
```

Williams (1992) [[Williams, 1992]](#bibliography) turned this into the REINFORCE
estimator — a Monte Carlo approximation that needs no model of the environment.
The score function \\(\nabla \log \pi\\) is a knob on each action's probability,
and weighting it by the return pushes the policy toward whatever produced
above-average outcomes. It is credit assignment again, now applied straight to the
policy's parameters rather than to a table of values.

The two routes are not rivals so much as ingredients. Actor-critic methods (A2C,
PPO, SAC) *fuse* them: a **critic** learns a value function to supply a
low-variance estimate of how good each action was, and an **actor** follows the
policy gradient using that estimate in place of the noisy Monte Carlo return.
That combination — value-based credit feeding policy-based improvement — is what
makes modern deep RL both stable and sample-efficient, and it is the gradient
engine that `rlevo` later sets *alongside* evolution in
[Why Combine Them?](50-why-combine.md).

> **Deeper reading.** Sutton and Barto, *Reinforcement Learning: An Introduction*
> (2nd ed., MIT Press, 2018) — freely available at
> [incompleteideas.net](http://incompleteideas.net/book/the-book-2nd.html) — is
> the definitive textbook. The DQN paper is Mnih et al. (2015), *Nature* 518,
> 529–533. For policy gradients, Williams (1992), *Machine Learning* 8, 229–256
> is the original; Schulman et al. (2015), arXiv:1502.05477 gives a unified
> treatment. [Appendix B](../appendix-b-rl-algorithms/index.md) gives pseudocode
> for DQN, PPO, and SAC as implemented in `rlevo`.

<a name="bibliography"></a>
*References on this page are collected in the [Bibliography](../bibliography.md).*

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
