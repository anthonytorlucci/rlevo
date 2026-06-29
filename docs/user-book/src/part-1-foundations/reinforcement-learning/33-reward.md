# Rewards

Everything an RL agent learns, it learns from one channel: the **reward**. The
[reinforcement-learning chapter](30-reinforcement-learning.md) defined the
agent's objective as the expected discounted return

```math
G_t = \sum_{k=0}^{\infty} \gamma^k\, r_{t+k+1},
```

so the reward \\(r\\) is the atom out of which that whole sum is built. You might
expect the type representing it to be elaborate. It is the opposite — `rlevo`'s
`Reward` trait is the smallest trait in the core crate. The interesting part is
*which* three capabilities it demands, and why those three are exactly enough.

That minimalism is the spine of the chapter. We start from the bare trait and the
`ScalarReward` workhorse that satisfies it, watch those three capabilities do real
work accumulating a return, follow a reward out of a step inside a `Snapshot`,
and only then meet the two places the type system earns its keep — the
terminated-vs-truncated distinction that a single `done` flag would corrupt, and
the structured rewards a minimal trait still accommodates without ceremony.

## The `Reward` trait

From `rlevo::core::base`:

```rust
pub trait Reward: Clone + Add<Output = Self> + Into<f32> + Debug {
    fn zero() -> Self;   // additive identity
}
```

Read the supertraits as a specification of what the rest of the library needs to
do with a reward, and nothing more:

- **`Add<Output = Self>` + `zero()`** — together these make rewards a *monoid*:
  you can add two of them, and there is an identity element. This is precisely
  what you need to accumulate a return — fold a sequence of rewards starting
  from `zero()`.
- **`Into<f32>`** — at some point a reward has to become a number a network can
  consume and a discount factor can scale. This bound guarantees that collapse
  to a scalar is always available, without committing the reward's *internal*
  representation to being a single float.
- **`Clone` + `Debug`** — rewards get stored in replay buffers (cloned) and
  logged (printed).

That is the entire contract. Note what is *absent*: no ordering, no
multiplication, no assumption that a reward is one number. A reward only has to
be summable and, ultimately, scalarisable.

> **The monoid is the point.** Modelling reward as "a thing you can add, with a
> zero" rather than hard-coding `f32` means return accumulation is generic over
> the reward type. The code that sums a trajectory doesn't care whether the
> reward is a scalar or a typed multi-component bundle — it just folds with `+`
> from `zero()`. Rust's trait system lets the *algebraic structure* be the
> interface.

## `ScalarReward`: the workhorse

Most environments emit a single number per step, and for them `rlevo::core::reward`
provides `ScalarReward` — a transparent wrapper around an `f32` that satisfies
the trait:

```rust
pub struct ScalarReward(pub f32);

impl Reward for ScalarReward {
    fn zero() -> Self { Self(0.0) }
}

impl Add for ScalarReward {
    type Output = Self;
    fn add(self, other: Self) -> Self { Self(self.0 + other.0) }
}

impl From<ScalarReward> for f32 { fn from(r: ScalarReward) -> f32 { r.0 } }
impl From<f32> for ScalarReward { fn from(v: f32) -> Self { Self(v) } }
```

It is `Copy`, `Default`, and `Serialize`/`Deserialize`, so it drops into replay
buffers and records without ceremony. You construct it either with the
tuple form where brevity helps inside an environment (`ScalarReward(1.5)`) or
with the more readable `ScalarReward::new(1.5)` from outside, and pull the value
back out with `.value()` or `f32::from(r)`. The `From<f32>`/`Into<f32>` pair in
both directions means it interoperates with raw floats wherever that is
convenient.

For the overwhelming majority of environments, `ScalarReward` is all you ever
touch. You reach past it only when a reward has structure you want to keep typed
— more on that at the end of the chapter.

## Accumulating a return

The monoid bounds exist so this kind of code is type-generic, not float-specific.
Summing the total reward of a trajectory is a plain fold from the identity:

```rust
let total_reward = experiences
    .iter()
    .fold(R::zero(), |acc, exp| acc + exp.reward.clone());
```

That works for *any* `R: Reward`. To turn an *undiscounted* sum into the
discounted return \\(G_t\\), the `Into<f32>` bound carries each reward into the
arithmetic where the \\(\gamma^k\\) factors live. The two bounds split the labour
cleanly: the monoid accumulates, the scalar conversion meters out the discount
and feeds the value network.

## How a reward leaves the environment

A reward never travels alone — it rides out of every step inside a `Snapshot`,
the per-step result type:

```rust
pub trait Snapshot<const R: usize>: Debug {
    type RewardType: Reward;
    fn reward(&self) -> &Self::RewardType;
    fn status(&self) -> EpisodeStatus;
    // observation(), is_done(), is_terminated(), is_truncated() ...
}
```

`Environment::step(action)` returns one of these, bundling the next observation,
the reward for the transition, and the episode status. The associated
`RewardType: Reward` bound is where the trait from the top of this chapter gets
pinned to a concrete type — most environments set `type RewardType = ScalarReward`.
Because it is an associated type rather than a hard-coded field, an environment
with a richer reward simply names a different type and everything downstream
still compiles.

## Terminated vs. truncated: a reward subtlety

Here is a place where a sloppy reward implementation quietly corrupts learning,
and where `rlevo` uses the type system to stop you. An episode can end two
different ways, and `EpisodeStatus` keeps them distinct:

```rust
pub enum EpisodeStatus {
    Running,
    Terminated,  // reached a terminal MDP state (goal, failure)
    Truncated,   // hit an external step limit
}
```

The difference matters for the value target. When an episode **terminates** —
the pole fell, the agent reached the goal — there genuinely is no future, so the
bootstrap value of the next state is zero:

```math
\text{target} = r_{t+1}.
```

When an episode is merely **truncated** — you cut it off at 500 steps — the world
*would have continued*, so throwing away the future value biases the target
downward. The correct target still bootstraps:

```math
\text{target} = r_{t+1} + \gamma\, V(s_{t+1}).
```

Conflate the two and you systematically under-value states near your time limit.
`rlevo` refuses to let you conflate them: `Terminated` and `Truncated` are
separate enum variants, and `Snapshot` exposes `is_terminated()` and
`is_truncated()` as distinct predicates (`is_done()` is just their `or`). The
distinction is carried, not collapsed — `rlevo`'s PPO rollout buffer, for
instance, records `terminated` and `truncated` as separate per-step flags so its
advantage estimation can treat them correctly.

> **In `rlevo`.** The lesson the type encodes: "the episode is over" is not one
> boolean. Reaching for a single `done` flag is the bug; the three-variant
> `EpisodeStatus` makes the safe path the default one.

## Shaped and multi-component rewards

Real environments — especially locomotion — rarely have a single clean reward.
A walker's step reward might be a sum of a forward-progress term, a control-cost
penalty, and a "still alive" bonus. You usually want to *log* those components
separately (to see which one the agent is gaming) even though `step` still emits
a single scalar.

`rlevo` handles this with `SnapshotMetadata`, an optional sidecar on a snapshot
that names reward components without changing the reward type:

```rust
pub struct SnapshotMetadata {
    pub components: BTreeMap<&'static str, f32>,   // "ctrl", "healthy", "goal"
    pub positions:  BTreeMap<&'static str, [f32; 3]>,
}
```

The keys are `&'static str` constants defined per-environment, so there are no
magic strings at the call sites. The scalar the agent optimises stays a clean
`ScalarReward`; the decomposition lives alongside it purely for analysis.

When the reward's *structure itself* needs to survive — a vector-valued reward
you intend to scalarise differently per experiment, or a typed bundle you want
to keep distinct until the last moment — you implement `Reward` directly on your
own type instead of using `ScalarReward`. As long as it adds, has a zero, and
can become an `f32`, the entire training stack accepts it unchanged. That is the
payoff of keeping the trait minimal: the common case is one line
(`type RewardType = ScalarReward`), and the exotic case is still just a trait
impl.

## Putting it together

- A **`Reward`** is a monoid (`+` and `zero()`) that can collapse to `f32`
  (`Into<f32>`) — exactly the structure needed to accumulate and then discount a
  return, and nothing more.
- **`ScalarReward`** is the single-`f32` concrete type the vast majority of
  environments use.
- Rewards leave each step inside a **`Snapshot`**, whose `RewardType` associated
  type pins the abstract trait to a concrete one.
- **`EpisodeStatus`** keeps `Terminated` and `Truncated` separate so value
  targets bootstrap correctly — a type-level guard against a classic RL bug.
- **`SnapshotMetadata`** logs shaped-reward components without polluting the
  optimised scalar; custom `Reward` impls cover genuinely structured rewards.

We now have the three nouns of the agent's interface — state, action, reward.
The next chapter brings them together under the **`Environment`** trait, the
`reset`/`step` protocol that turns this vocabulary into an interactive loop.

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
