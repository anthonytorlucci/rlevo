# Glossary: rank vs components, space vs value, type-level vs runtime

**Why this exists.** Four separate defects in this repo (#100, #253, a stale
`from_slice` doc comment, and a self-contradiction inside `docs/rules.md`)
were one confusion wearing different hats: **tensor rank is not a component
count**. There was no single page a contributor could read to get the
vocabulary right, so the conflation kept being re-derived at each new call
site. This chapter is that page — not tidying, a deliverable in its own right
(issue #376).

**Key source of truth.** [`docs/rules.md`](https://github.com/anthonytorlucci/rlevo/blob/main/docs/rules.md)
§3 and §7, [ADR 0038](https://github.com/anthonytorlucci/rlevo/blob/main/docs/adr/0038-continuous-action-components-const.md),
[ADR 0053](https://github.com/anthonytorlucci/rlevo/blob/main/docs/adr/0053-bounded-action-per-component-bounds.md),
[ADR 0047](https://github.com/anthonytorlucci/rlevo/blob/main/docs/adr/0047-sensor-relocates-emission-model-to-environment.md).

## Rank, components, and shape

| Term | In `rlevo` it means | Habitually mistaken for |
|---|---|---|
| `RANK` / `R` | number of tensor axes | a size, a count, or matrix rank |
| `COMPONENTS` | flattened scalar count | `RANK`, or `shape().iter().product()` |
| `shape()` | per-axis cardinality, an array of length `R` | a component count |
| `DA` / `DO` / `DB` / `DAB` | tensor **ranks**, despite reading as "dims" | dimensionality / element count |
| `dim(A)` in the RL literature | action-space **component count** | `RANK` |

**Rank** (`Action::RANK`, `Observation::RANK`, `= R`) is the number of tensor
axes — "the count of indices needed to address an element" (NumPy `ndim`,
Burn's `Tensor<B, R>`) — never matrix rank, never a size, never a product of
anything. A rank-1 action has exactly one axis; that axis can hold one value
or a thousand.

**`shape()`** returns `[usize; R]`, the per-axis cardinality. Its own doc
comment (`crates/rlevo-core/src/base.rs`, on `Action::shape`) is explicit that
it is *never* a source of component count:

> `shape()` describes **axis cardinality only**. It is *never* a source of
> the flattened scalar component count of a continuous action: neither
> `shape().iter().product()` nor `R` is universally correct, because the
> workspace permits both a rank-1 action with `shape() == [C]` and a
> rank-`C` action with `shape() == [1; C]`.

That last sentence is the whole trap in one line: the workspace has **two
live conventions** for how `shape()` relates to component count, and they
give opposite answers for `RANK == COMPONENTS`. Neither `RANK` nor any
expression built from `shape()` can be trusted to equal the true component
count — it has to come from a source that states it directly.

**`COMPONENTS`** (`ContinuousAction::COMPONENTS`) is that source: the length
of the slice `as_slice()` returns and `from_slice()` consumes, declared
explicitly by every implementor with no default (a default of `= R` would
silently reproduce the bug below). See
`ContinuousAction::COMPONENTS`'s own doc comment
(`crates/rlevo-core/src/action.rs:296-308`) for the full reasoning; this
chapter hoists the summary rather than restating it. `BoundedAction::low()`/
`high()` are keyed on the same `COMPONENTS`, not on `R` — `R` is retained on
`BoundedAction<R>` only to name the `ContinuousAction<R>` supertrait and
appears in no method signature (ADR 0053).

**Historical cost of the conflation** — all four now fixed:

- **#100** — `ContinuousAction::random()` sampled `Self::RANK` values, then
  called `from_slice`, which requires the full `COMPONENTS` — an
  unconditional panic for every rank-1, multi-component action with no
  override (three box2d types). Fixed by ADR 0038.
- **#253** — `BoundedAction::low()/high()` returned `[f32; R]`, so a rank-1
  action with `COMPONENTS > 1` could express only one bound value for all its
  components; the same audit found the DDPG/TD3/SAC `act` paths looping
  `0..A::RANK` instead of `0..A::COMPONENTS`. Fixed by ADR 0053.
- `ContinuousAction::from_slice`'s docs once claimed the slice must have
  exactly `RANK` elements — stale from the moment ADR 0038 made `COMPONENTS`
  the contract. Now reads `COMPONENTS` (`crates/rlevo-core/src/action.rs:356-361`).
- `docs/rules.md:115` (`ContinuousAction<D>` invariant) and `docs/rules.md:205`
  (the `from_slice` panic contract) carried the identical conflation, the
  second contradicting the first once it was corrected. Both now read
  `COMPONENTS`, with an explicit "**not** `D`" callout.

### `DA` / `DO` / `DB` / `DAB` vs `dim(A)`

The const generics on RL agent structs (e.g. `SacAgent<B, Actor, Critic, O, A,
const DO: usize, const DB: usize, const DA: usize, const DAB: usize>`,
`crates/rlevo-reinforcement-learning/src/algorithms/sac/sac_agent.rs:161-164`)
are **ranks** — `DA` is `A::RANK`, not `A::COMPONENTS` — despite the letter
`D` reading as "dimension." This is the same conflation as `RANK` vs
`COMPONENTS`, one layer up, and it collides head-on with a second convention:
in the RL literature, `dim(A)` (or `dim(\mathcal{A})`) means the action-space
**component count**.

SAC's target entropy heuristic is the concrete case: Haarnoja et al. 2018b
(arXiv:1812.05905, Appendix D Table 1) give the entropy target as `−dim(A)`
("e.g., −6 for HalfCheetah-v1"), which is `−COMPONENTS`. Reading `DA` in
`SacAgent`'s generics as that same `dim(A)` and writing `-(A::RANK as f32)`
is exactly the bug that was live until commit `e0ce1cd`; the code now reads
`-(A::COMPONENTS as f32)` (`sac_agent.rs:280`, documented at
`sac_config.rs:43`). When translating a paper's `dim(·)` into this codebase,
it almost always means `COMPONENTS`, not a `D*` generic.

## Action space vs action value (and state vs observation)

The trait names in `rlevo-core` name **values** — a single `Action`,
`Observation`, `State` — but several of the methods on those traits describe
the **space** the value is drawn from, and nothing in the naming signals
which is which:

| Concept | What names it | Kind |
|---|---|---|
| A concrete action | `Action` (an instance) | value |
| The action space's bounds | `BoundedAction::low()` / `high()` | space |
| The action space's cardinality | `DiscreteAction::ACTION_COUNT` | space |
| A concrete observation | `Observation` (an instance) | value |
| A concrete state | `State` (an instance) | value |

This matters because the RL literature talks almost exclusively about
spaces (𝒜, 𝒮, `dim(A)`, `Box(low, high)`), while the code you write mostly
handles values (one `Action` per `step()` call). `BoundedAction::low()`/
`high()` and `DiscreteAction::ACTION_COUNT` are the seam where a *space*
property is exposed through a *value* type's trait — read them as
"properties of the type `A`, describing every possible value of `A`," not as
properties of any one action.

**State vs observation** is a related but separate distinction, sharpened by
[ADR 0047](https://github.com/anthonytorlucci/rlevo/blob/main/docs/adr/0047-sensor-relocates-emission-model-to-environment.md):
`State` no longer produces its own observation. In the POMDP tuple ⟨S, A, T,
R, Ω, O⟩ the emission model `O: S × A → Π(Ω)` is a property of the
**environment**, not of a point `s ∈ S` — so it lives on the env-side
`Sensor<OR, AR, SR>` trait (`crate::environment::Sensor`), with the canonical
signature `observe(&self, action, next_state) -> Observation` where `&self`
is the environment. `State` keeps only what genuinely belongs to a state
value: its rank, `shape()`, `numel()`, and `is_valid()`. If you are asking
"what does the agent perceive," that is a `Sensor`/`Observation` question;
"what is the Markov-complete world state," that is a `State` question — the
two are no longer the same trait, and `Observable<OR>` (ADR 0019) is now a
demoted, optional pure-projection helper rather than the seam observations
flow through.

## Burn's type-level "non-autodiff" vs PyTorch's `no_grad`

Comments across the off-policy agents say things like "runs on the target's
(non-autodiff) backend; no gradients are produced"
(`crates/rlevo-reinforcement-learning/src/algorithms/td3/target_smoothing.rs:32-33`).
A reader arriving from PyTorch — which this codebase courts, it cites CleanRL
by name — will map that straight onto `torch.no_grad()`. That is the wrong
model, and the difference is worth internalizing once rather than
re-explaining at every call site (this chapter is that once; see
[ch07 §"The `AutodiffBackend` bound"](ch07-adding-an-rl-algorithm.md) for
where it plugs into agent code):

- **PyTorch:** `no_grad()` / `inference_mode()` is a *runtime context* you
  enter and exit — a flag flipped on a tensor's autograd machinery for the
  duration of a `with` block. It is easy to forget to enter, and a tensor
  computed inside can still (with effort) end up entangled with one computed
  outside.
- **Burn:** `Autodiff<B>` and `B` are different **types**. A module generic
  over `B: AutodiffBackend` computes gradients; the same module instantiated
  over `B::InnerBackend` (reached via `.valid()` / `.inner()`) computes on
  plain `Tensor<B::InnerBackend, D>` values that have no gradient tape at
  all. Gradients are not suppressed at runtime — they are **unrepresentable**
  at the type level. There is no context to forget to enter, because the
  "did we forget" question has no type that would compile if you did.

See the Burn Book, [Autodiff building block, "Difference with
PyTorch"](https://burn.dev/books/burn/building-blocks/autodiff.html). One
more divergence trips PyTorch readers specifically: Burn's `backward()`
returns gradients in a `Gradients` container rather than populating a
`.grad` field on each tensor in place, so retrieving a gradient is
`grad(&gradients)` / `grad_remove(&mut gradients)` against that container,
not `tensor.grad`.

The practical payoff: reviewing target-network code in PyTorch means
auditing "did we remember to wrap this call in `no_grad()`?" — a runtime
question you must check by reading control flow. In Burn, the function
signature is the proof: if a helper takes `Tensor<B, D>` where `B:
Backend` (not `AutodiffBackend`), no caller can smuggle a gradient-tracked
tensor through it without an explicit `.valid()`/`.inner()` conversion first.
"Non-autodiff backend" in a comment like `target_smoothing.rs:32-33` means
"this function's `B` is not bounded by `AutodiffBackend`," not "gradients
were computed and then discarded."
