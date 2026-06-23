# State Constraints

In the previous chapter, the Grid Agent showed you how
[`State`](https://docs.rs/rlevo-core/latest/rlevo_core/base/trait.State.html) can
describe a whole world — even when that world was just a discrete grid of tiles.
That was a deliberately gentle start. Real robotics, though, rarely fits on a
grid: it lives in continuous space, carries precise physical units, and obeys
constraints that couple several quantities at once.

So let's deepen the story. We keep exactly the same
[`rlevo-core`](https://docs.rs/rlevo-core/latest/rlevo_core/index.html) traits you
already know, and turn them loose on a harder question: **what does it mean for a
state to be "legal" in a physical space, and where do we enforce it?**

Our vehicle is a `RobotPose` — a 2D robot in a 1000 mm × 1000 mm workspace whose
orientation is bounded to \\([-180°, 180°]\\). You can run the complete program at
any point:

```bash
cargo run -p rlevo-examples --example ch00_state_constraints
```

## Continuous quantities as fixed-point integers

A robot's position is physically continuous, but the moment you put it in software
you have to *choose* a representation. The obvious reach is for floating point —
and it's worth pausing on why we don't. Floats make equality treacherous
(\\(0.1 + 0.2 \neq 0.3\\)) and can leak non-determinism into a pipeline you'd like
to reproduce exactly.

`rlevo` never forces a representation on you, so `RobotPose` makes a pragmatic
call: it stores each quantity as a **fixed-point integer**. Position lives in
millimetres and orientation in millidegrees (where 1000 mdeg = 1°). That buys us
the best of both worlds — the resolution of a continuous space with the
reliability of values that are exact, `Hash`-able, and `Eq`-comparable, which a
floating-point pose could never be.

```rust,no_run
{{#rustdoc_include ../../../crates/rlevo-examples/examples/book/ch00_state_constraints.rs:state}}
```

Here's the part worth noticing: even though the underlying units changed from the
grid's "cells" to fixed-point millimetres, the **trait surface is identical**. It
is still a rank-1 `State<1>` whose `shape()` is `[3]`. That consistency is the
whole point — you can swap a toy game world for a high-precision robot model
without rewriting a line of your core logic.

Now compare this `is_valid` with the grid agent's. There, validity simply meant
*inside the grid*. Here it is a **box constraint** over three coupled axes — two
positional bounds and one angular bound — and it is the single predicate that
defines the entire admissible state space. Keep that in mind, because everything
else on this page exists to keep states on the right side of it. And unlike the
grid agent, `observe` here performs *full* observability: the observation is a
direct projection of the pose, because this robot perceives its complete state.

## Validity enforced at construction

In the Grid Agent we used a **check-and-discard** discipline: compute a candidate
move, test it, and ignore it if it lands out of bounds. That works beautifully for
discrete tiles. But it isn't the only tool we have, and for high-stakes systems
it often isn't the right one.

The stricter discipline — and the one `RobotPose` demonstrates — is to **make an
invalid state unrepresentable by construction**. Rather than letting a bad pose
exist and hoping someone checks it, we validate at the door with a fallible
constructor:

```rust,no_run
{{#rustdoc_include ../../../crates/rlevo-examples/examples/book/ch00_state_constraints.rs:construction}}
```

`new` validates against `is_valid` and returns an `Option`: `Some(pose)` only when
every constraint holds, `None` otherwise. This is the builder pattern you'll meet
throughout `rlevo-environments` — a fallible constructor is the single chokepoint
where the invariant is checked *once*, so any `RobotPose` a downstream algorithm
receives is already known good. The one escape hatch is `new_unchecked`, for the
hot path where you have *already* guaranteed validity and want to skip the check
for speed. Notice that it's a named, deliberate opt-out rather than a silent one —
you choose safety or speed, and the choice is visible at the call site.

## Distance as a reward signal

A constrained state space is only half of an environment. The other half is the
**reward** — the feedback that tells an agent how well it's doing. This is a good
moment to see where a `State` hands off to that idea.

`RobotPose` exposes a Euclidean distance between two poses, deliberately ignoring
orientation:

```rust,no_run
{{#rustdoc_include ../../../crates/rlevo-examples/examples/book/ch00_state_constraints.rs:distance}}
```

On its own that's just a measurement. The example *shapes* it into a reward by
normalising against the workspace diagonal,
\\(\sqrt{1000^2 + 1000^2} \approx 1414\\) mm, and inverting it, so a pose nearer
the goal scores higher:

```math
r = 1 - \min\!\left(\frac{d}{d_{\max}},\, 1\right)
```

That gives the agent a smooth gradient to climb. This is the seam where a `State`
connects to the **reward** concept developed in
[Part I](part-1-foundations/reinforcement-learning/33-reward.md) — a single scalar,
derived from the state, that turns a physical measurement into a goal.

One orientation note before you go further. What we just built is a *maximisation*
signal: reward climbs as the robot nears the goal, exactly as most RL
practitioners expect. rlevo's evolutionary half runs the **same** direction — the
optimisation stack ([`FitnessEvaluable`](https://docs.rs/rlevo-core/latest/rlevo_core/fitness/trait.FitnessEvaluable.html)
and the algorithms in `rlevo-evolution`) is maximise-native: higher fitness is
better. So when this reward becomes a fitness for an evolutionary optimiser — the
"evolutionary" in evolutionary deep RL — it flows straight through. A policy-return
objective like `RolloutFitness` declares `ObjectiveSense::Maximize`, and the
return is used as-is; only a genuine *cost* (a loss, an error) declares
`ObjectiveSense::Minimize`, and even then the harness reconciles direction at one
chokepoint, so you never hand-negate. We surface it here because earlier versions
of rlevo minimised, and any habit of "negating a reward into a cost" is now exactly
the bug to avoid.

## Keeping a state valid under accumulation

The box constraint introduces a wrinkle the grid agent never had to face — and
it's a classic robotics gotcha, so let's slow down on it. Apply angular actions
repeatedly and the orientation drifts past \\(\pm 180°\\). Strictly read,
`is_valid` would then reject the state — even though the *physical* heading is
perfectly legal, because angles wrap around.

In the Grid Agent a step out of bounds was discarded, full stop, because walking
off the edge is a genuine physical impossibility. An angle past \\(180°\\) is a
different animal: it isn't illegal, it's merely *unnormalised*. So the right
response is not to forbid the action but to **normalise the state back into range**:

```rust,no_run
{{#rustdoc_include ../../../crates/rlevo-examples/examples/book/ch00_state_constraints.rs:normalize}}
```

Wrapping a \\(270°\\) raw heading to \\(-90°\\) keeps the state inside the
admissible set without losing any information — the robot can still perform full
rotations, and its internal representation stays clean and canonical. Think of it
as the angular analogue of the grid agent rejecting a step off the edge: same
invariant, but where an off-grid position is truly illegal, an over-range angle
just needs canonicalising.

## What you should see

When you run the example, you aren't only watching a robot move — you're watching
the same `is_valid` predicate enforced from several angles: valid construction,
rejection of six out-of-bounds cases, distance-based reward shaping, orientation
normalisation, a verified trajectory of waypoints, and a short agent loop in which
any move that would leave the workspace is blocked. Each of those is one
consequence of that single predicate on the state.

Step back and look at the Grid Agent and the Robot Pose together, and you've now
met the two enforcement disciplines `rlevo` leans on everywhere:
**check-and-discard** at the call site, and **validate-at-construction** behind a
fallible builder. [Part I](part-1-foundations/reinforcement-learning/31-state.md)
formalises the state and observation seams, and
[Part II](part-2-guided-tour/00-overview.md) puts a real environment and a real
algorithm around them.

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
