# Grid Agent

Welcome to your first encounter with `rlevo`. Before we write a new algorithm or
build a complex environment, we'll walk through the core building blocks in
[`rlevo-core`](https://docs.rs/rlevo-core/latest/rlevo_core/) — **state**,
**observation**, **actions**, and the **tensor bridge** — using a simple "grid
agent" as our guide.

This agent navigates a bounded world by turning and stepping forward, mirroring
the Minigrid-style environments you'll find in `rlevo::envs::grids`.

To see everything in action, you can run the full example:

```bash
cargo run -p rlevo-examples --example ch00_grid_agent
```

*Note: every code block below is pulled directly from the implementation our CI
suite runs, so what you read here is exactly what you'll see when you run it.*

## Dimensional safety through const generics

One of our core design goals in `rlevo` is catching mistakes as early as
possible. To that end, we thread a const generic `R` — the **tensor rank** — through
all of the primary traits (`State<R>`, `Observation<R>`, and `Action<R>`).

Think of `R` as your safety rail. A rank-1 value is a flat vector, a rank-2 value
a matrix, and so on; the rank is fixed at the type level. If you define a state as
a flat vector and later try to treat it as a matrix, the compiler stops you before
a single tensor is ever allocated — a `State<1>` and its `Observation<1>` must
agree on rank, and any mismatch is a compile error rather than a runtime surprise.

Each trait exposes its rank two ways: the associated constant `RANK`, and a
`shape()` method that returns the per-axis extents as `[usize; R]`. The grid agent
is entirely rank-1, so every trait below is instantiated at `R = 1`. For a state
that flattens to three numbers — \\(x\\), \\(y\\), and a heading — `shape()` is `[3]`
and `RANK` is `1`.

## Observation — what the agent perceives

The world has many properties, but an
[`Observation`](https://docs.rs/rlevo-core/latest/rlevo_core/base/trait.Observation.html)
captures only what the agent is permitted to *see*. The grid agent's view is
strictly egocentric: it knows its \\((x, y)\\) position and the direction it faces,
and nothing more.

```rust,no_run
{{#rustdoc_include ../../../crates/rlevo-examples/examples/book/ch00_grid_agent.rs:observation}}
```

The `facing` field is a four-way compass direction, and it is part of what the
agent perceives precisely *because* the world is egocentric: the agent turns
relative to its current heading rather than naming an absolute direction, so it
must know which way it is pointing. Notice what we leave out — the grid's width
and height. Those are properties of the world, not of the agent's immediate
perception, so they belong on the state instead. Keeping the observation lean is
what lets us scale to richer environments later, where an agent's view might be
limited to a handful of sensors.

## TensorConvertible — the bridge to a network

An observation is only useful to a neural policy once it becomes a tensor. The
[`TensorConvertible`](https://docs.rs/rlevo-core/latest/rlevo_core/base/trait.TensorConvertible.html)
trait is that bridge. It defines a strict round-trip for a backend `B`: `to_tensor`
encodes your value for the network, and `from_tensor` decodes it back into your
custom type, returning a `TensorConversionError` if the shape or contents are
invalid.

```rust,no_run
{{#rustdoc_include ../../../crates/rlevo-examples/examples/book/ch00_grid_agent.rs:tensor}}
```

Here we flatten `(x, y, facing)` into three `f32` values, packing the `facing`
enum into a single byte index on the way out. `from_tensor` is the deliberate
inverse: it checks the shape is `[3]`, reads the three values back, and rejects
any byte that does not name a valid direction. The example exercises this contract
end to end — it encodes an observation, decodes it, and asserts the result is
unchanged, the same round-trip a policy depends on every step. Because the trait
is generic over `B`, that one implementation serves the CPU `ndarray` backend, the
`wgpu` GPU backend, or the `Flex` backend the example uses, with no change to the
conversion logic.

## State — the full world

If the observation is what the agent perceives, the
[`State`](https://docs.rs/rlevo-core/latest/rlevo_core/base/trait.State.html) is
what the world *is*. The `State` trait owns the information the agent must not
see — here, the grid's boundaries.

```rust,no_run
{{#rustdoc_include ../../../crates/rlevo-examples/examples/book/ch00_grid_agent.rs:state}}
```

We use `State` to tie three responsibilities together:

1. Its associated type names which `Observation` it projects into.
2. The `observe()` method performs that projection, deliberately dropping `width`
   and `height` so the bounds stay hidden.
3. The `is_valid` method enforces the world's invariants — a position is legal
   only while it sits inside the grid's dimensions.

The trait also gives you `numel` for free: a default method that multiplies the
entries of `shape()` to report the total element count — `3` for this state. This
distinction between full state and partial observation is the primary seam `rlevo`
uses to model partially observable problems throughout the library.

## DiscreteAction — one choice from a finite set

We sort actions into a small hierarchy to make them easier to work with. The
simplest form is a
[`DiscreteAction`](https://docs.rs/rlevo-core/latest/rlevo_core/action/trait.DiscreteAction.html):
a single choice drawn from a finite, index-addressable set. The grid agent's
movement is exactly this — turn left, turn right, or step forward.

```rust,no_run
{{#rustdoc_include ../../../crates/rlevo-examples/examples/book/ch00_grid_agent.rs:discrete}}
```

We define the relationship between your action variants and the integers
\\([0, \texttt{ACTION\\_COUNT})\\) as a strict bijection: `from_index` and `to_index`
must invert one another. That round-trip is what lets a network emit an integer —
or an \\(\operatorname{argmax}\\) over logits — and have it resolve to a unique,
unambiguous action every time. From those two methods the trait derives
`enumerate`, which lists every action, and `random`, which samples one uniformly;
the example uses both to walk the full action set.

## MultiDiscreteAction — independent sub-choices

Often a controller needs to decide several things at once — perhaps a movement
direction *and* an interaction command. Flattening these into one giant list of
possibilities invites a combinatorial explosion, so instead we reach for
[`MultiDiscreteAction<R>`](https://docs.rs/rlevo-core/latest/rlevo_core/action/trait.MultiDiscreteAction.html):
`R` independent sub-dimensions, each indexed on its own axis.

```rust,no_run
{{#rustdoc_include ../../../crates/rlevo-examples/examples/book/ch00_grid_agent.rs:multidiscrete}}
```

Here `shape()` is `[3, 2]` — three movements times two interactions, six
combinations in all — and the index *array* `[movement, interact]` replaces the
single scalar of a `DiscreteAction`. Because the axes are independent, your policy
chooses movement and interaction separately rather than enumerating the full
product space, which is the practical reason to reach for `MultiDiscreteAction`
over a flattened `DiscreteAction`.

## Putting it together — the transition

The traits above define the *structure* of the problem; a transition function
supplies the *dynamics*. Given a state and a movement, `step` returns the candidate
successor — turning leaves the position fixed, while a forward step displaces it by
the unit vector for the current facing.

```rust,no_run
{{#rustdoc_include ../../../crates/rlevo-examples/examples/book/ch00_grid_agent.rs:transition}}
```

Notice what `step` does *not* do: it never checks the bounds. It may hand back an
out-of-bounds successor, and the caller is expected to call `is_valid` on the
result and decide whether to accept or discard it — for instance, refusing a step
that would walk the agent through a wall. This division — dynamics propose, the
validity invariant disposes — is exactly how the real environments in
`rlevo-environments` operate. The example drives a short scripted sequence through
this loop, accepting each legal move and rejecting the final step that would walk
off the top edge.

## What you should see

When you run the example, it prints each component in turn: the state and its
rank, the observation tensor round-trip, the discrete and multi-discrete action
tables, and finally the egocentric walk. The last line reports `valid=false` as
the agent reaches the boundary and its state is left unchanged — a compact
demonstration of the propose-and-dispose flow.

With these pieces in hand — state, observation, tensor bridge, and actions — you
now have the full vocabulary `rlevo` uses to describe a problem. Part I develops
each of them in depth, and Part II puts them to work driving real algorithms.

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
