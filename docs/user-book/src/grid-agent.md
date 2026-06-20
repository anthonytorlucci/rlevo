# Grid Agent

This is the first encounter with `rlevo`'s building blocks. Before any algorithm
runs, every problem in the library is described through a small set of traits in
[`rlevo-core`](https://github.com/anthonytorlucci/rlevo/tree/main/crates/rlevo-core):
a **state**, the **observation** an agent perceives of it, the **actions** it may
take, and the **tensor bridge** that carries those values into a neural network.
This page introduces each of them through a single worked example — an
egocentric grid agent that turns and steps through a bounded world, mirroring the
Minigrid-style environments shipped in `rlevo-environments::grids`.

The complete program is the `ch00_grid_agent` example. Run it with:

```bash
cargo run -p rlevo-examples --example ch00_grid_agent
```

The code shown below *is* that example — compiled and run in CI — so what you
read is what runs.

## Dimensional safety through const generics

The first thing to notice is that the core traits — `State<R>`, `Observation<R>`,
and `Action<R>` — all carry a const generic `R`, the **tensor rank** of the value.
A rank-1 value is a flat vector, a rank-2 value a matrix, and so on. The rank is
fixed at the type level: a `State<1>` and its `Observation<1>` must agree on rank,
and the compiler rejects any mismatch before a single tensor is allocated. The
grid agent is entirely rank-1, so every trait below is instantiated at `R = 1`.

Each trait exposes the rank as the associated constant `RANK` and a `shape()`
that returns the per-axis extents as `[usize; R]`. For a value that flattens to
three numbers, `shape()` is `[3]` and `RANK` is `1`.

## Observation — what the agent perceives

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

## TensorConvertible — the bridge to a network

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

## State — the full world

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

## DiscreteAction — one choice from a finite set

Actions come in a small hierarchy. The simplest is a
[`DiscreteAction`](https://docs.rs/rlevo-core): a single choice drawn from a
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

## MultiDiscreteAction — independent sub-choices

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

## Putting it together — the transition

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

## What you should see

Running the example prints each component in turn: the state and its rank, the
observation tensor round-trip, the discrete and multi-discrete action tables,
and finally the egocentric walk, the last line of which reports
`valid=false` as the agent reaches the boundary and its state is left unchanged.

With these pieces in hand — state, observation, tensor bridge, and actions — you
have the full vocabulary `rlevo` uses to describe a problem. Part I develops each
of them in depth, and Part II puts them to work driving real algorithms.

---

*Co-Authored-By: Anthropic Claude Opus 4.8*\
*Reviewed-By: (Human) Anthony Torlucci*
