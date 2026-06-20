# State Constraints

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

```bash
cargo run -p rlevo-examples --example ch00_state_constraints
```

## Continuous quantities as fixed-point integers

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

## Validity enforced at construction

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

## Distance as a reward signal

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

## Keeping a state valid under accumulation

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

## What you should see

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

---

*Co-Authored-By: Anthropic Claude Opus 4.8*\
*Reviewed-By: (Human) Anthony Torlucci*
