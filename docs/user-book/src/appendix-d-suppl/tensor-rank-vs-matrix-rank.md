# Tensor Rank vs. Matrix Rank vs. Dimensionality

The word **rank** is overloaded. In `rlevo` it appears as the const generic on
`State<R>`, `Observation<R>`, and `Action<AR>` — and it means something very
specific there. But the same word means something *different* in linear algebra,
and something different again in everyday "how big is this space?" talk. Three of
those senses collide precisely in the discussion of partially-observable
environments ([issue #62](https://github.com/anthonytorlucci/rlevo/issues/62)),
where it is easy to conclude that an observation space has a "lower rank" than its
state space and reach for the wrong tool. This page pulls the three senses apart.

## Three senses of "rank"

### 1. Tensor order (ndim) — the one `rlevo` tracks

The `const R` in `State<R>` / `Observation<R>` is the **number of axes** a tensor
has — what NumPy calls `ndim` and Burn writes as the `R` in `Tensor<B, R>`. It is
the count of indices you need to address a single element, *not* the size of any
axis:

- a scalar is order $0$;
- a flat vector $v \in \mathbb{R}^{n}$ is order $1$ — one index, $v[i]$;
- a matrix / greyscale image $M \in \mathbb{R}^{h \times w}$ is order $2$ — two
  indices, $M[i][j]$;
- an RGB image is order $3$ — $\text{img}[c][i][j]$.

Crucially, the *size* of each axis is irrelevant to the order. A length-$1000$
vector and a length-$2$ vector are **both order 1**. This is the only "rank" the
type system reasons about: it stops you from feeding a rank-2 observation where a
rank-1 one is expected, but it says nothing about how many values live along each
axis. (See also the [tensor conventions](index.md#tensor-conventions-in-rlevo)
note.)

### 2. Matrix rank — a property of the *values*, invisible to the types

In linear algebra, the **rank** of a matrix $A \in \mathbb{R}^{m \times n}$ is the
dimension of its column space — the number of linearly independent columns,
$\operatorname{rank}(A) \le \min(m, n)$. This is a fact about the *numbers inside*
the matrix, not about its shape. The matrix

```math
A = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}
```

is order-2 and shape $2 \times 2$, yet has **matrix rank 1** (the second column is
twice the first). A linear map $y = Ax$ with $\operatorname{rank}(A) = r$ collapses
its input onto an $r$-dimensional subspace, losing information along the null
space. The type system cannot see this at all: $A$ is just a `Tensor<B, 2>`
whatever its column space looks like.

### 3. Cardinality / dimensionality — the size of an axis

The third sense is the everyday one: "how big is the space?" — the number of values
an axis (or the whole space) can take. In `rlevo` this is the content of
`shape()`: a rank-1 observation with `shape() == [16]` has $16$ slots along its
single axis. Critically, **`Observation::shape()` is independent of
`State::shape()`** — a rank-1 state of shape $[n]$ can already produce a rank-1
observation of shape $[m]$ with $m < n$. Shrinking an axis is a *dimensionality*
reduction at **constant tensor order**.

### Side by side

| Sense | Means | A property of… | Where it lives in `rlevo` |
| --- | --- | --- | --- |
| **Tensor order (ndim)** | number of axes / indices | the container's shape | the `const R`/`SR`/`AR` on `State`/`Observation`/`Action` — the **only** "rank" the types track |
| **Matrix rank** | dimension of a linear map's column space | the *values* of a matrix | nowhere in the types; a numerical fact about a `Tensor<B, 2>` |
| **Cardinality / dimensionality** | size of an axis (count of values) | the container's shape | `shape()` entries (and `numel()`) — independent between state and observation |

A clean way to keep them straight: **tensor order** counts the axes,
**dimensionality** counts the values *along* an axis, and **matrix rank** counts
the independent directions *inside* the numbers. Reducing one says nothing about
the others.

## Why this matters for partial observability

A POMDP is defined by an *information* deficit: the observation does not pin down
the state. Newcomers often translate "the observation carries less information" into
"the observation has lower rank" — but which rank? Almost always the reduction is in
**dimensionality** (a smaller axis) or in **matrix rank** (a lossy linear/stochastic
projection), both at **constant tensor order**. Genuinely changing the *tensor
order* — the `const R` — is a much rarer thing, and it has a different name:
a **modality change**.

The canonical POMDP benchmarks make the distinction concrete. Reading the "rank"
column carefully shows that nearly all of them keep the tensor order fixed:

| Example | State (order / shape) | Obs (order / shape) | Reduction mechanism | Tensor-order change? |
| --- | --- | --- | --- | --- |
| **Tiger** | 1 / $[2]$ | 1 / $[2]$ | stochastic aliasing (noisy emission) | no |
| **PO grid world** | 1 / $[N\cdot M]$ | 1 / $[16]$ | combinatorial compression (local percept) | no |
| **RockSample** | 1 / $[\text{pos}\cdot 2^k]$ | 1 / $[3]$ | exponential collapse + distance-decayed sensor | no |
| **LQG** | 1 / $[n]$ | 1 / $[m]$, $m<n$ | **matrix**-rank-$m$ linear projection $y = Cx + \varepsilon$ | **no** |
| **Contact manipulation** | 1 / $[n]$ | 1 / $[m]$, $m\ll n$ | nonlinear proprioceptive projection $y = h(x)+\varepsilon$ | no |
| **Atari** | 1 / $[\text{ram}]$ | **2–3** / $[H, W, C]$ | **modality** change (RAM → pixels) | **yes** |

The instructive trap is **LQG** — the example most often cited as the "cleanest"
statement of a rank-deficient observation. Its emission matrix $C \in
\mathbb{R}^{m \times n}$ with $m < n$ genuinely *is* rank-deficient — but that is
**matrix rank**. Both $x \in \mathbb{R}^{n}$ and $y \in \mathbb{R}^{m}$ are
**order-1 tensors**; the Kalman filter and the separation principle live entirely
in that constant-order, dimensionality-reducing regime. Nothing about LQG changes
`ndim`.

Only the last row — **Atari**, a compact emulator-RAM state (order 1) observed as a
pixel image (order 2 or 3) — actually changes the tensor order. That, and only
that, is a case the const generics must be allowed to differ for.

## How `rlevo` handles each case

The three senses map onto three different mechanisms. Since
[ADR 0047](https://github.com/anthonytorlucci/rlevo/blob/main/docs/adr/0047-sensor-relocates-emission-model-to-environment.md)
(superseding [ADR 0019](https://github.com/anthonytorlucci/rlevo/blob/main/docs/adr/0019-observable-projection-trait.md)),
the mechanism that turns a state into an observation is the environment's
[`Sensor<OR, AR, SR>`](https://docs.rs/rlevo-core/latest/rlevo_core/environment/trait.Sensor.html),
not a method on `State` — but the three-way split below is unchanged by that
move, because it was always about *which rank is doing the reducing*, not about
which trait holds the reduction:

- **Dimensionality / stochastic reduction** (Tiger, grid worlds, RockSample, LQG,
  contact manipulation) is handled by a **`Sensor`** whose `observe`/`observe_reset`
  return a smaller-`shape`, same-order observation. Because `Observation::shape()`
  is independent of `State::shape()`, no special machinery is needed — this is
  the flavour of partial observability `rlevo` models directly, and `R == SR`
  still holds.
- **Matrix-rank reduction** is just a *kind* of dimensionality/information loss; it
  too is expressed inside a `Sensor` (or inside a learned encoder). It never
  forces a tensor-order change.
- **Modality change** (Atari) is the one case that alters the tensor order, and
  it is directly expressible by `Sensor` itself: `Sensor<OR, AR, SR>` already
  carries `OR` as a rank independent of `SR`, so an environment's sensor can
  return an observation of any order at all — no companion trait is *required*
  to change tensor order any more. Where that projection happens to be a pure
  function of state (no world context needed), though, it is common — and DRY —
  for the sensor's body to delegate to a helper: the
  [`Observable<OR>`](https://docs.rs/rlevo-core/latest/rlevo_core/state/trait.Observable.html)
  trait, via `state.project()`. `rlevo`'s `pixel_grid` environment is the
  reference for that pattern — `PixelGridState: Observable<3>` holds the
  projection body, and `PixelGridEnv: Sensor<3, 1, 1>` simply forwards to
  `next_state.project()`. Before ADR 0047, `Observable` was the *only* escape
  from `State::observe()`'s same-rank binding (ADR 0019); now it is a
  convenience for keeping a state-pure projection in one place, not the
  mechanism that makes the rank change possible. The
  [Environments chapter](../part-1-foundations/reinforcement-learning/34-environment.md)
  walks through how an environment wires a `Sensor` up.

The one-line takeaway: **partial observability is almost always a dimensionality
or matrix-rank reduction at constant tensor order — handled inside a `Sensor`.
A modality change is directly expressible by `Sensor`'s own `OR` parameter;
`Observable<OR>` remains a convenience helper for the case where that projection
is a pure function of state.**

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
