# State Constraints
In our previous chapter, we used the Grid Agent to see how [`State`](https://docs.rs/rlevo-core/latest/rlevo_core/base/trait.State.html) can define a world—even if that world was simplified into a discrete grid of tiles. It was a great starting point, but real-world robotics often requires us to move beyond "tiles" and deal with continuous spaces, precise units, and complex constraints.

This page deepens the story of [`State`](https://docs.rs/rlevo-core/latest/rlevo_core/base/trait.State.html). We keep the same [`rlevo-core`](https://docs.rs/rlevo-core/latest/rlevo_core/index.html) traits as before, but we’ll see how they handle a much more nuanced problem: **how do we define what is "legal" in a physical space?**

The vehicle here is a `RobotPose` — a robot moving in a 1000 mm x 1000 mm workspace with an orientation bounded between -180° and 180°. You can run the full example here:

<!--
The [Grid Agent](grid-agent.md) introduced `State` as the full description of a
world, with `is_valid` guarding a single invariant — the agent must stay inside
the grid. That example kept things deliberately coarse: integer cells, a discrete
heading, an out-of-bounds check that simply rejected the offending move. This
page deepens the `State` story. It keeps the same `rlevo-core` traits but pushes
on the part the grid agent only touched lightly: **what it means for a state to
be valid, and how that invariant is enforced at the boundaries of a continuous
workspace**.

The vehicle is `RobotPose` — a 2D robot in a 1000 mm × 1000 mm workspace whose
orientation is bounded to \\([-180°, 180°]\\). The complete program is the
`ch00_state_constraints` example:
-->

```bash
cargo run -p rlevo-examples --example ch00_state_constraints
```

## Continuous quantities as fixed-point integers

While a robot's position is physically "continuous," representing it in software involves choices. You could use floating-point numbers, but those can introduce non-deterministic behavior or issues with equality checks ($0.1 + 0.2 \neq 0.3$).

`rlevo` doesn’t force a specific type on you, so RobotPose makes a practical choice: it stores measurements as fixed-point integers. By using millimeters for position and millidegrees for orientation (where $1000$ mdeg = 1°), we get the best of both worlds: the precision of continuous space with the reliability of exact, `Hash`-able, and `Eq`-comparable integers.

Even though the underlying data types have changed from the grid's "cells" to "fixed-point units," notice that the **trait surface remains identical**. It is still a rank-1 `State<1>` with a shape of `[3]`. This consistency allows you to swap a simple game world for a high-precision robot model without rewriting your core logic.

<!--
A pose is naturally continuous — a position and an angle. `rlevo` does not force
a representation on you, and `RobotPose` makes a pragmatic choice: store each
quantity as a fixed-point integer. Position is held in **millimetres** and
orientation in **millidegrees** (1000 mdeg = 1°), so the whole state is exact,
`Hash`-able, and `Eq`-comparable — properties a floating-point pose could not
offer. The trait surface is identical to the grid agent's: a rank-1 `State<1>`
whose `shape()` is `[3]`.

```rust,no_run
{{#rustdoc_include ../../../crates/rlevo-examples/examples/book/ch00_state_constraints.rs:state}}
```

Compare this with the grid agent's `is_valid`. There, validity meant *inside the
grid*. Here it is a **box constraint** over three coupled axes — two positional
bounds and one angular bound — and it is the single predicate that defines the
admissible state space. Everything else in the example exists to keep states on
the right side of this predicate. As before, `observe` performs full
observability: the observation is a direct projection of the pose, because this
agent perceives its complete state.
-->

## Validity enforced at construction

In the Grid Agent example, we used a "check-and-discard" approach: the system calculates a move, and if it’s out of bounds, we simply ignore it. That works well for discrete tiles.

However, in many high-stakes systems, we want to **make an invalid state unrepresentable by construction**. This is the philosophy behind `RobotPose` (and `rlevo` in general). Instead of allowing a pose to exist and then checking if it's valid, we use a fallible constructor:

```rust,no_run
{{#rustdoc_include ../../../crates/rlevo-examples/examples/book/ch00_state_constraints.rs:construction}}
```

By using the **Builder Pattern**, we ensure that if you have a `RobotPose` object in your hand, it is *guaranteed* to be valid. The only exception is `new_unchecked`, which we provide for "hot paths" where you already know the data is safe and want to skip the safety check for performance. This gives you a clear, named choice between safety and speed.

<!--
The grid agent computed a candidate successor and left the caller to test
`is_valid` and discard it. That is one discipline. `RobotPose` demonstrates a
stricter one: **make an invalid state unrepresentable by construction**. The
`new` constructor validates against `is_valid` and returns an `Option` —
`Some(pose)` only when every constraint holds, `None` otherwise:

```rust,no_run
{{#rustdoc_include ../../../crates/rlevo-examples/examples/book/ch00_state_constraints.rs:construction}}
```

This is the builder pattern used throughout `rlevo-environments`: a fallible
constructor is the chokepoint where the invariant is checked once, so any
`RobotPose` a downstream algorithm receives is already known good. The escape
hatch `new_unchecked` exists for the hot path where the caller has already
guaranteed validity and wants to skip the check — a deliberate, named opt-out
rather than a silent one.
-->

## Distance as a reward signal

A state space is just half of the story; the other half is the reward—the feedback that tells the agent how well it's performing.

In this example, we take a raw physical property (Euclidean distance) and "shape" it into a reward. By normalizing the distance against the workspace dimensions and inverting it, we create a gradient that an agent can climb: \\(r=1-\min (\frac{d}{d_\max },1)\\)

This is the bridge where our `State` logic connects to the Reward concepts in [Part I](part-1-foundations/reinforcement-learning/33-reward.md). We turn a physical measurement into a mathematical goal for the agent.

<!--
A constrained state space is only half of an environment; the other half is the
**reward** that tells an agent how well it is doing. `RobotPose` exposes a
Euclidean distance between poses, ignoring orientation:

```rust,no_run
{{#rustdoc_include ../../../crates/rlevo-examples/examples/book/ch00_state_constraints.rs:distance}}
```

The example turns this into a shaped reward by normalising against the
workspace diagonal \\(\sqrt{1000^2 + 1000^2} \approx 1414\\) mm and inverting it,
so that a pose nearer the goal scores higher: \\(r = 1 - \min(d / d_{\max}, 1)\\).
This is the seam where a `State` connects to the **reward** concept developed in
[Part I](part-1-foundations/reinforcement-learning/33-reward.md) — a single
scalar derived from the state that an agent can climb.
-->

## Keeping a state valid under accumulation

One of the trickiest problems in robotics is "wrapping" logic—specifically, how angles behave. If a robot turns 190 degrees, it is still physically facing "backwards," but its raw counter might now exceed our $180^\circ$ boundary.

In the Grid Agent, a step out of bounds was discarded because it was a physical impossibility. But for an angle, exceeding $180^\circ$ isn't an error; it’s just an unnormalized value. Instead of rejecting the action, we normalize the state back into the valid range:

```rust,no_run
{{#rustdoc_include ../../../crates/rlevo-examples/examples/book/ch00_state_constraints.rs:normalize}}
```

This ensures that the robot’s "internal" representation stays clean and consistent without sacrificing its ability to perform full rotations.

<!--
The box constraint creates a problem the grid agent never faced. Repeatedly
applying angular actions makes orientation drift past \\(\pm 180°\\), which
`is_valid` would then reject — even though the *physical* heading is perfectly
legal, since angles wrap. The fix is to **normalise back into the valid range**
rather than to forbid the action:

```rust,no_run
{{#rustdoc_include ../../../crates/rlevo-examples/examples/book/ch00_state_constraints.rs:normalize}}
```

Wrapping a \\(270°\\) raw heading to \\(-90°\\) keeps the state inside the
admissible set without losing information. This is the angular analogue of the
grid agent rejecting a step off the edge — but where a position off the grid is
genuinely illegal, an angle past \\(180°\\) is merely *unnormalised*, so the
right response is to canonicalise rather than reject.
-->

## What you should see

When you run this example, you aren't just seeing a robot move; you are seeing two different philosophies of safety in action: **check-and-discard** (for things like physical boundaries) and **validate-at-construction** (for ensuring internal consistency). These are core ideas that permeate throughout the library.

By looking at both the Grid Agent and the Robot Pose, you’ve now seen how rlevo handles everything from simple logic to complex, real-world constraints. [Part I](part-1-foundations/reinforcement-learning/31-state.md) formalizes these state boundaries, and [Part II](part-2-guided-tour/00-overview.md) will show you how we put a full algorithm to work within these rules.

<!--
Running the example walks through the same invariant from several angles: valid
construction, rejection of six out-of-bounds cases, distance-based reward
shaping, orientation normalisation, a verified trajectory of waypoints, and a
short agent loop in which moves that would leave the workspace are blocked. Each
section is one consequence of the single `is_valid` predicate defined on the
state.

Between this page and the grid agent you have seen the two enforcement
disciplines `rlevo` uses throughout: **check-and-discard** at the call site, and
**validate-at-construction** behind a fallible builder. [Part I](part-1-foundations/reinforcement-learning/31-state.md)
formalises the state and observation seams, and [Part II](part-2-guided-tour/00-overview.md)
puts a real environment and a real algorithm around them.
-->

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
