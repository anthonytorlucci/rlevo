# Grid Agent

Welcome to your first encounter with `rlevo`. Before we can write the next 
ground-breaking algorithm or build the most advanced robotic manipulation 
environment, we'll walk through a few core building blocks in 
[`rlevo-core`](https://docs.rs/rlevo-core/latest/rlevo_core/), namely **state**, 
**observation**, **actions**, and the **tensor bridge** using a simple "grid 
agent" as our guide.

This agent navigates a bounded world by turning and stepping forward, mirroring 
the Minigrid-style environments found in `rlevo::envs::grids`. 

To see everything in action, you can run the full example:

```bash
cargo run -p rlevo-examples --example ch00_grid_agent
```

*Note: The code below is the exact implementation used in our CI suite; what 
you read here is exactly what you will see when you run it.*

## Dimensional safety through const generics

One of our core design goals in `rlevo` is catching errors as early as 
possible. To achieve this, we use a constant generic `R` to represent the 
tensor rank across all primary traits (`State<R>`, `Observation<R>`, and 
`Action<R>`).

Think of `R` as your safety rail: if you define a state as a flat vector 
(rank-1), the compiler will catch any attempt to treat it as a matrix before a 
single tensor is ever allocated. In our grid agent example, everything is 
rank-1, so you'll see these traits instantiated at `R = 1`. Each trait exposes 
this rank through the `RANK` constant and a `shape()` method (e.g., `[3]` for 
a state containing $x$, $y$, and heading).

<!--
The first thing to notice is that the core traits — `State<R>`, `Observation<R>`,
and `Action<R>` — all carry a const generic `R`, the **tensor rank** of the value.
We mentioned this in the [introduction](#introduction.md), but we will unpack that 
idea a bit more here. A rank-1 value is a flat vector, a rank-2 value a matrix, 
and so on. The rank is fixed at the type level: a `State<1>` and its `Observation<1>` 
must agree on rank, and the compiler rejects any mismatch before a single 
tensor is allocated. The grid agent is entirely rank-1, so every trait below is 
instantiated at `R = 1`.

Each trait exposes the rank as the associated constant `RANK` and a `shape()`
that returns the per-axis extents as `[usize; R]`. For a value that flattens to
three numbers, `shape()` is `[3]` and `RANK` is `1`.
-->

## Observation — what the agent perceives

While the world has many properties, an [`Observation`](https://docs.rs/rlevo-core/latest/rlevo_core/base/trait.Observation.html) represents only what the 
agent is actually permitted to "see." In our grid example, the agent’s view is 
strictly egocentric: it knows its $(x, y)$ position and the direction it is 
facing.

Notice that we intentionally omit the grid's total width and height from the 
`Observation`. Those are properties of the world environment, not the agent's 
immediate perception. By keeping the observation "lean," we allow for easier 
scaling to complex environments where an agent's view might be limited by 
sensors.

<!--
An [`Observation`](https://docs.rs/rlevo-core) is the agent's *view* of the world.
The grid agent sees its position and the direction it faces — nothing more:

```rust,no_run
{{#rustdoc_include ../../../crates/rlevo-examples/examples/book/ch00_grid_agent.rs:observation}}
```

The `facing` field is a four-way compass direction. It is part of what the agent
perceives because the world is **egocentric**: the agent turns relative to its
current heading rather than naming an absolute direction, so it must know which
way it is pointing. Crucially, the grid's width and height are *absent* — those
are properties of the world, not of the agent's perception, and they belong on
the state instead.
-->

## TensorConvertible — the bridge to a network

An observation is only useful to a neural network once it becomes a tensor. The [`TensorConvertible`](https://docs.rs/rlevo-core/latest/rlevo_core/base/trait.TensorConvertible.html) trait acts as the bridge here. It defines a strict round-trip: `to_tensor` encodes your data for the backend, and `from_tensor` decodes it back into your custom type.

In the code below, you'll see us pack the `facing` enum into a single byte index during encoding. We then enforce a strict check in `from_tensor` to ensure the resulting value is a valid direction. Because this trait is generic over any backend `B`, the same logic works seamlessly whether you are running on CPU (flex) or GPU (wgpu) backend.

<!--
An observation becomes useful to a neural policy only once it is a tensor.
[`TensorConvertible<R, B>`](https://docs.rs/rlevo-core) defines that round-trip
for a backend `B`: `to_tensor` encodes the value, and `from_tensor` decodes it
back, returning a `TensorConversionError` if the shape or contents are invalid.

```rust,no_run
{{#rustdoc_include ../../../crates/rlevo-examples/examples/book/ch00_grid_agent.rs:tensor}}
```

The encoding flattens `(x, y, facing)` into three `f32` values, packing the
`facing` enum into a single byte index. `from_tensor` is the deliberate inverse:
it checks the shape is `[3]`, reads the three values back, and rejects any byte
that does not name a valid direction. The example exercises this contract by
encoding an observation, decoding it, and asserting the result is unchanged —
the round-trip that any policy relies on every step.

The trait is generic over the backend `B`, so the same implementation serves the
CPU `ndarray` backend, the `wgpu` GPU backend, or the `Flex` backend the example
uses, with no change to the conversion logic.
-->

## State — the full world
If the observation is what the agent perceives, the [`State`](https://docs.rs/rlevo-core/latest/rlevo_core/base/trait.State.html) is what the world is. The `State` trait houses the "hidden" information, such as the grid's boundaries.

We use the `State` trait to tie three responsibilities together:

1. It defines which `Observation` it projects into.
2. The `observe()` method performs that projection (deliberately dropping the bounds).
3. The `is_valid` method enforces the physical invariants of the world — for instance, ensuring a coordinate is only "valid" if it sits within the grid's dimensions.

This distinction between state and observation is the primary seam rlevo uses to model partially observable problems.

<!--
Where the observation is what the agent *sees*, the
[`State`](https://docs.rs/rlevo-core) is what the world *is*. It owns the
information the agent must not see — here, the grid bounds:

```rust,no_run
{{#rustdoc_include ../../../crates/rlevo-examples/examples/book/ch00_grid_agent.rs:state}}
```

`State` ties three responsibilities together. Its associated type `Observation`
names the perception it yields; `observe` performs the projection, deliberately
dropping `width` and `height` so the bounds stay hidden; and `is_valid` encodes
the world's invariant — a position is legal only while it sits inside the bounds.
This separation between full state and partial observation is the seam `rlevo`
uses to model partially observable problems throughout the library.

The trait also provides `numel`, a default method that multiplies the entries of
`shape()` to report the total element count — `3` for this state.
-->

## DiscreteAction — one choice from a finite set

We categorize actions into a small hierarchy to make them easier to work with. The simplest form is a [`DiscreteAction`](https://docs.rs/rlevo-core/latest/rlevo_core/action/trait.DiscreteAction.html): a single choice from a finite, index-addressable set (like *left*, *right*, or *forward*).

To ensure this works perfectly with neural networks, we define the relationship between your high-level action types and integer indices as a strict bijection. This guarantees that when a network emits an integer or performs an `argmax` over logits, it resolves to a unique, unambiguous action every time.

<!--
Actions come in a small hierarchy. The simplest is a
[`DiscreteAction`](https://docs.rs/rlevo-core/latest/rlevo_core/action/trait.DiscreteAction.html): a single choice drawn from a
finite, index-addressable set. The grid agent's movement is exactly this — turn
left, turn right, or step forward:

```rust,no_run
{{#rustdoc_include ../../../crates/rlevo-examples/examples/book/ch00_grid_agent.rs:discrete}}
```

The contract is a bijection between variants and the integers \\([0,
\texttt{ACTION\_COUNT})\\): `from_index` and `to_index` must invert one another.
That round-trip is what lets a network emit an integer (or an
\\(\texttt{argmax}\\) over logits) and have it resolve unambiguously to an action.
From those two methods the trait derives `enumerate`, which lists every action,
and `random`, which samples one uniformly — both used in the example to walk the
full action set.
-->

## MultiDiscreteAction — independent sub-choices

Often, a controller needs to decide multiple things at once—perhaps choosing both a movement direction and an interaction command. Instead of flattening these into one giant list of possibilities (which can lead to a "combinatorial explosion"), we use [`MultiDiscreteAction<R>`](https://docs.rs/rlevo-core/latest/rlevo_core/action/trait.MultiDiscreteAction.html).

By treating these as independent sub-dimensions, your policy can choose the movement and the interaction separately. In our example, this results in a shape of `[3, 2]`, giving the agent six possible combinations while keeping the underlying logic clean and modular.

<!--
Many controllers decide several things at once. A
[`MultiDiscreteAction<R>`](https://docs.rs/rlevo-core) models exactly that: `R`
independent sub-dimensions, each indexed on its own axis. The grid agent's
compound action pairs a movement with an optional interaction:

```rust,no_run
{{#rustdoc_include ../../../crates/rlevo-examples/examples/book/ch00_grid_agent.rs:multidiscrete}}
```

Here `shape()` is `[3, 2]` — three movements times two interactions, six
combinations in all — and the index *array* `[movement, interact]` replaces the
single scalar of a `DiscreteAction`. Because the axes are independent, a policy
can choose movement and interaction separately rather than enumerating the full
product space, which is the practical reason to reach for `MultiDiscreteAction`
over a flattened `DiscreteAction`.
-->

## Putting it together — the transition

While the traits above define the *structure* of the problem, the transition function provides the *physics*.

The `step` function defines how the world changes when an action is taken. Crucially, `step` is only responsible for the dynamics: it calculates where a move would land you. It doesn't check if that spot is valid. Instead, we use the `is_valid` method on the resulting state to decide whether to accept the movement or discard it (e.g., if the agent tried to walk through a wall).

This separation—dynamics propose, validity disposes—is exactly how the real-world environments in `rlevo-environments` operate.

<!--
The traits describe *structure*; a transition function supplies the *dynamics*.
Given a state and a movement, `step` returns the candidate successor — turning
leaves the position fixed, while a forward step displaces it by the unit vector
for the current facing:

```rust,no_run
{{#rustdoc_include ../../../crates/rlevo-examples/examples/book/ch00_grid_agent.rs:transition}}
```

Note what `step` does *not* do: it never checks the bounds. It may return an
out-of-bounds successor, and the caller is expected to call `is_valid` on the
result and decide whether to accept or discard it. This division — dynamics
propose, the validity invariant disposes — is the same one the real
`rlevo-environments` grids follow. The example drives a short scripted sequence
through this loop, accepting each legal move and rejecting the final step that
would walk off the top edge.
-->

## What you should see

When you run the example, you'll see the system print each component in turn: the state and its rank, the tensor round-trip, the action tables, and finally the egocentric walk. The final line will show `valid=false` as the agent hits the boundary — a perfect demonstration of our "propose and dispose" flow.

With these pieces—state, observation, tensor bridge, and actions—you now have the full vocabulary required by `rlevo`.

<!--
Running the example prints each component in turn: the state and its rank, the
observation tensor round-trip, the discrete and multi-discrete action tables,
and finally the egocentric walk, the last line of which reports
`valid=false` as the agent reaches the boundary and its state is left unchanged.

With these pieces in hand — state, observation, tensor bridge, and actions — you
have the full vocabulary `rlevo` uses to describe a problem. Part I develops each
of them in depth, and Part II puts them to work driving real algorithms.
-->

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
