---
project: rlevo
status: active
type: decision
date: 2026-08-08
tags: [adr, decision, fitness, sanitization, f32, f64, accumulator, overflow, reduction, neat, rlevo-evolution]
---

# ADR 0069: Sanitized fitness is reduced in `f64` — the `+∞ → f32::MAX` clamp bounds a *value*, not a *reduction*

## Status

**Accepted (2026-08-08).** Resolves the shared mechanism behind issue #132
(`StrategyMetrics::from_host_fitness`, fixed in PR #1061) and issue #1062
(`speciate` / `allocate_offspring` in `crates/rlevo-evolution/src/neuroevolution/
species.rs`), and closes a third, latent instance in `shaping::z_score` found
while writing this ADR.

**Supersedes nothing. Extends ADR
[0034](0034-fitness-hygiene-chokepoint-convention.md)**, in the same relation
0034 itself holds to ADR [0023](0023-objective-sense-and-maximize-convention.md):
0034's `NaN → −∞` / `+∞ → f32::MAX` / `−∞`-pass-through rule is preserved
**verbatim and unchanged**, its four driver chokepoints stand, and 0034 stays
`active`. Nothing in `sanitize_fitness` or `sanitize_fitness_tensor` changes.

**It does correct one sentence of ADR 0034's reasoning.** ADR 0034's
clamp-mapping decision (Decision 1) justifies the `+∞ → f32::MAX` mapping
with the parenthetical

> `+∞ → f32::MAX` (ranks top but **finite**, so it cannot blow a `mean`,
> `variance`, or reward to `+∞`)

That claim is false, and this ADR's own Context section, below, shows it is
not merely false about `f32::MAX` — it is unachievable by *any* finite
sentinel. Per this repository's
immutability rule, ADR 0034 is **not edited**. The correction is carried by this
ADR, by an appended clause on 0034's row in
[`docs/adr/README.md`](README.md), by `docs/rules.md`'s "Optimisation
direction" section, and — most importantly for a reader who will never open
either — by the rustdoc on `sanitize_fitness` itself.

**Chosen shape.** Four parts:

1. The rule: a reduction over sanitized fitness accumulates in `f64` and narrows
   to `f32` **at most once, after the reduction**.
2. Named `f64`-accumulating reduction primitives in
   `rlevo-evolution/src/fitness.rs`, so the correct reduction is the shortest
   thing to write and has a greppable name.
3. The rationale correction propagated to the three editable surfaces that
   currently repeat it.
4. A behavioural **rescale-invariance** property test as the mechanical net —
   explicitly *not* a source-text guard, for the reason in this ADR's own
   Decision 5, below.

## Context

### The claim is not slightly wrong; it is structurally unachievable

`f32::MAX` is finite, so a sanitized `+∞` **joins** a sum rather than being
excluded from it. And `f32::MAX + f32::MAX == f32::INFINITY`. Two sanitized-`+∞`
members in one `f32` accumulator therefore produce exactly the `+∞` mean ADR 0034
says they cannot.

The deeper point, which is what makes this an ADR rather than a comment: the
guarantee "clamping the value protects the reduction" cannot be delivered by any
choice of sentinel. A sentinel `S` protects a sum over `N` members only if
$N \cdot S < \texttt{f32::MAX}$. For `f32::MAX` that fails at $N = 2$; and no
fixed finite $S$ satisfies it for unbounded $N$ (at $N = 10^{6}$ the largest
admissible $S$ is $\approx 3.4 \times 10^{32}$, seven orders of magnitude below
`f32::MAX`). Lowering $S$ would also forfeit the property the clamp exists for —
`+∞` must rank above every legitimate finite fitness, and any
$S < \texttt{f32::MAX}$ is exceeded by some legitimate value.

So the clamp and the accumulator width are answering **two different questions**,
and 0034's parenthetical conflates them:

- `sanitize_fitness` bounds one **value** so it can be *compared*, *stored*, and
  *summed at all*. That is real and it is what ADR 0034 delivers.
- The **accumulator width** is what bounds a *reduction*. Only `f64` delivers
  "cannot blow a mean up", and it delivers it decisively:
  $\texttt{f64::MAX} / \texttt{f32::MAX} \approx 5.3 \times 10^{269}$, so an `f64`
  accumulator absorbs any population of `f32::MAX` terms that could physically
  exist.

`f64` accumulation is therefore not a *safety margin* on the clamp. It is the
mechanism that was mistakenly attributed to the clamp.

### Three instances, and only one of them mentions `sanitize_fitness`

The reduce-over-fitness-magnitude population in `rlevo-evolution` was enumerated
for this ADR (as of 2026-08-08; the method was to grep the *shape* — an
accumulation whose terms are fitness magnitudes — not the name
`sanitize_fitness`, because two of the five sites never mention it). Five sites.
**Four got the width wrong.**

| site | reduction | status |
|---|---|---|
| `StrategyMetrics::from_host_fitness` (`strategy.rs`) | mean of finite members | defect (#132), **fixed** by PR #1061 — widened to `f64` |
| `speciate` (`neuroevolution/species.rs`) | per-species mean adjusted fitness | defect (#1062), being fixed |
| `allocate_offspring` (`neuroevolution/species.rs`) | $\sum$ `adjusted_fitness_sum.max(0.0)` | defect (#1062), being fixed — **overflows independently**, even when every species' mean is individually finite |
| `shaping::z_score` | population mean **and variance** | **latent defect, found for this ADR** — see below |
| `gep::strategy::roulette_select` | $\sum$ fitness-offset weights | the only site that anticipated it — see below |

Two facts about this table decide this ADR's own Decision 5, below.

**First: the taint is transitive through stored fields, so a name-keyed search
cannot find it.** `allocate_offspring` sums `Species::adjusted_fitness_sum`, an
`f32` field. That field holds a *mean of sanitized fitness* written by
`speciate`, but the summation site contains no lexical trace of fitness hygiene
at all. Any grep, lint, or source-text guard keyed on `sanitize_fitness` misses
it — which is precisely how the #132 fix failed to generalize: `species.rs` never
routes through `from_host_fitness`, so it did not inherit the widening.

**Second: `z_score` is the "variance" the parenthetical names, and it fails at
`N = 1`.** `shaping::z_score` computes `mean` on the `f32` fitness tensor, then
`centered.powf_scalar(2.0).sum()` — also `f32`. Squaring means a *single*
`f32::MAX` member overflows its own squared term to `+∞` before any accumulation
happens, so accumulation order is irrelevant. A host simulation of that exact
`f32` arithmetic over nine ordinary members plus one `f32::MAX`:

```
one f32::MAX member -> [-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0]
control (top = 1e18) -> [-0.33, -0.33, -0.33, -0.33, -0.33, -0.33, -0.33, -0.33, -0.33, 3.0]
```

`var = +∞` → `std = +∞` → **every** shaped utility is zero. No `NaN`, no panic,
no infinity in the output. In a gradient-style ES update
($\sum_i u_i \cdot \varepsilon_i$, the exact consumer centered-rank/z-score exist
to feed) that is a silent zero update: the strategy stops learning and reports
finite numbers, which is the same failure class ADR 0068's Context section
names for the $\sigma$ collapse. `z_score` is `pub` and has **no call site in the workspace
today**, so this is latent, not live — but it is
public API, and it is the instance the false parenthetical most directly
licensed.

`roulette_select` (`gep/strategy.rs:217`) is the counter-example worth recording:
it sums in `f32` but guards `if total <= 0.0 || !total.is_finite()` and falls
back to uniform sampling. That is the correct *defensive* shape and it prevents
corruption — but note what it costs. Under saturation GEP silently degrades from
fitness-proportional to uniform selection. The guard converts a wrong answer into
a *different* wrong answer; only the accumulator width preserves the intended
one.

### The failure is severe and it is silent

Verified by execution for #1062: three species with mean fitness 100 / 10 / 1 and
`pop_size = 30` apportion `[27, 3, 0]` correctly, and `[10, 10, 10]` once
saturation collapses the shares — fitness-proportional reproduction becomes
uniform reproduction, **population-wide**, for the rest of the run. NEAT still
runs, still speciates, still reports finite `best_fitness`, and has stopped doing
the one thing the apportionment step exists to do. That is the profile that makes
a convention worth mechanizing: no crash, no `NaN`, no failing assertion.

### The recurrence is not the recurrence ADR 0034 forecast

ADR 0034's Consequences section names one reopen trigger — the **bypass hole** — and one
escalation for it: the `CanonicalFitness<B>` newtype, "the fast-follow if a
bypass regression recurs."

That trigger has **not** fired, and reaching for its remedy here would be a
mis-diagnosis worth stating plainly. All three instances *sanitized correctly*.
`speciate` even carries a comment citing ADR 0034 and sanitizing "before *every*
reduction" — the author read the rule, applied it faithfully, and then wrote
`let sum: f32 = … .sum()`, because the ADR told them the clamp made the sum safe.
`CanonicalFitness<B>` wraps a `Tensor<B, 1>` at the `tell` boundary; every one of
these defects is host-side `f32` scalar arithmetic *downstream* of that boundary.
The newtype would not have caught one of them. 0034's deferral stands on its own
merits, untouched by this ADR.

## Decision

### 1. The rule

> **A reduction over sanitized fitness accumulates in `f64` and narrows to `f32`
> at most once, after the reduction.**

"Reduction" is any fold whose terms are fitness magnitudes: sum, mean, variance,
weighted sum, $\sum$ of a field derived from fitness. This binds **transitively**: a
stored `f32` field holding a sum or mean of sanitized fitness is itself a fitness
magnitude, and reducing over *it* is covered (this is exactly
`allocate_offspring`).

Three operations are explicitly **not** covered, and stay `f32`: ordering
(`total_cmp`, sorts, `fold(−∞, f32::max)`), single-value comparison, and argmax.
They are unaffected because saturation is order-preserving — `rules.md`'s
section 3 sanitize-then-`total_cmp` convention is complete for them and is
not touched here.

`f32` remains the storage and public-API width. This ADR changes accumulator
width only.

This rule is stated for **host iterator reductions**, where the caller owns the
accumulator. A reduction performed by a Burn device op does not own its
accumulator — `Tensor::sum()` accumulates in `B::FloatElem` — and satisfies this
ADR by bounding its terms instead. See this ADR's own Decision 4, below,
where that distinction was forced by the implementation rather than
anticipated by this decision.

### 2. Named `f64`-accumulating reduction primitives

Alongside `sanitize_fitness` in `crates/rlevo-evolution/src/fitness.rs`,
`pub(crate)` like its siblings:

```rust
/// Mean of `values` over an `f64` accumulator, narrowed once at the end.
pub(crate) fn sanitized_mean(values: impl IntoIterator<Item = f32>) -> f32;

/// Sum of `values` over an `f64` accumulator. Returns `f64`: the caller
/// narrows deliberately, or does not narrow at all.
pub(crate) fn sanitized_sum(values: impl IntoIterator<Item = f32>) -> f64;
```

This is the load-bearing half of the decision, and the reasoning is the same
one ADR 0068's Consequences section gives for `config::nondegenerate_bounds`:
**the check acquires a name.** The convention stops being a paragraph a
contributor must have read and becomes the shortest correct thing to write at
the call site. It is also what makes this ADR's own Decision 5's grep viable,
since the *absence* of these names in a fitness reduction is now the signal.

`sanitized_sum` returns `f64` deliberately. A `-> f32` sum would re-introduce the
narrowing this ADR exists to move, and `allocate_offspring`'s `total` is a
divisor — it does not need to be `f32` at all.

**Empty-input contract**, decided during implementation because this decision as
first written omitted it: `sanitized_mean([]) == f32::NEG_INFINITY`, total, never
panicking. `−∞` is the maximise-native worst sentinel (ADR 0023), it is already
what `from_host_fitness` returns for an all-broken population — so adoption is
bit-for-bit behaviour-preserving — and the IEEE answer (`0/0 → NaN`) is the one
value the crate's hygiene rule exists to eliminate. A panicking primitive would
put a runtime-data panic behind `pub` reductions, contrary to `rules.md`'s
section 4. `sanitized_sum([]) == 0.0`, the additive identity, for which a
mean has no analogue. If these are ever promoted past `pub(crate)` (reopen trigger 2), this
contract is part of the public surface.

`from_host_fitness`, `speciate`, and `allocate_offspring` adopt the primitives.

**`from_host_fitness` is not a drop-in, and this decision originally implied it
was.** `sanitized_mean` averages *every* value it is given; `from_host_fitness`
needs the mean over **finite members only**, with a `broken_count` of the rest.
The filter therefore lives at the call site, not in the primitive:

```rust
let mean = crate::fitness::sanitized_mean(
    fitnesses.iter().copied()
        .filter(|&f| crate::fitness::sanitize_fitness(f).is_finite()),
);
```

Same accumulation order, same division, same single narrowing, and the empty case
coincides with the old all-broken `−∞` branch, so `mean_fitness` is unchanged
bit-for-bit for every input — which matters, because it is read in five crates.
The cost is a second pass over a once-per-generation statistic. No
`sanitized_mean_of_finite` variant was added; `best`/`worst`/`broken_count` stay
in their original single pass as order statistics this ADR's own Decision 1
excludes.

**Term consistency is co-equal with accumulator width — swapping in the primitive
for a total alone is a regression.** This decision as first written treats
accumulator width as the whole of the change. At `allocate_offspring` it is not.
Sanitizing `total` while leaving the share *numerator* unsanitized makes a `+∞`
term divide a now-**finite** total, yielding an infinite share; `∞.floor() as
usize` saturates to `usize::MAX`, and the overshoot-reclaim loop then runs
`usize::MAX − pop_size` times. That is a **hang**, not a wrong answer, and it was
hit live during implementation (`test_allocate_offspring_poisoned_species_keeps_
proportionality has been running for over 60 seconds`). The remedy is a single
shared `share_term` closure feeding both the total and every numerator, so
$\sum \text{share} = \texttt{pop\_size}$ remains an identity. Binding rule: **a reduction and the
terms compared against it must be sanitized by the same expression.**

### 3. The rationale correction reaches the surfaces a reader actually hits

ADR 0034's false parenthetical is currently reproduced, near-verbatim, on three
**editable** surfaces. All three are corrected; the ADR itself is not touched.

- **`crates/rlevo-evolution/src/fitness.rs` — the `sanitize_fitness` rustdoc.**
  This is the highest-priority edit in the entire ADR. It currently reads "`+∞ →
  f32::MAX`: … so it cannot blow a population `mean`/`variance`/reward to `+∞`",
  and it is the IDE tooltip at every one of the ~90 `sanitize_fitness` call sites
  in the crate. It is the surface that misled the author of `speciate`. It must
  say instead that the clamp makes the value *summable*, and that the
  reduction's safety comes from this ADR's own Decision 1's accumulator
  width.
- **`docs/rules.md`'s "Optimisation direction" section.** The one-line summary
  of the hygiene rule ("`+∞ → f32::MAX` (ranks top but finite, so it cannot
  blow a `mean`/reward up)") is corrected, and this ADR's own Decision 1's
  corollary is added as its own bullet beside the existing
  sanitize-then-`total_cmp` bullet.
- **`docs/adr/README.md`.** 0034's row repeats the claim in its summary. The row's
  existing text is **kept** — it is a faithful summary of an immutable record and
  rewriting it would launder the history — and a `**Rationale corrected by
  0069:**` clause is appended.

The index is the right carrier for the pointer, and the reasoning is worth
stating because it recurs: `README.md` is a *navigation* artifact, not a decision
record. Its own header says an ADR is not edited and "the superseded record is
annotated rather than deleted" — annotating the index is the mechanism that
sentence presupposes. Amending it is not an edit to 0034.

### 4. `shaping::z_score` bounds its terms — it *cannot* widen its accumulator

In scope for this ADR rather than deferred, per `rules.md`'s section 12
("prefer fixing over filing when the fix is in scope and cheap"): it is a
`pub` function, it is the `variance` case the corrected parenthetical names,
and leaving the workspace's own counter-example standing while the ADR that
names it lands would be the worst of both.

**This ADR's own Decision 1's rule does not transfer verbatim to a device
reduction, and this was discovered while implementing it.** `Tensor::sum()`
accumulates in `B::FloatElem`, which the *backend* fixes; Burn exposes no
"sum in `f64`" knob. Reaching one would mean a device→host round-trip —
precisely what ADR 0034 introduced `sanitize_fitness_tensor` to avoid, and
what `rules.md`'s section 3 tells us to avoid for a whole `Tensor<B, 1>`.
`z_score` compounds this: it has no `&Device` parameter (unlike
`centered_rank`), so rebuilding the tensor needs `fitness.device()` plus a
full host→device upload, and its host read is *fallible* on a non-`f32`
`B::FloatElem` — which its signature cannot report, so the round-trip would
force an `expect` on a read that genuinely can fail, contrary to
`rules.md`'s section 4.

On-device, the equivalent guarantee comes from **bounding the terms rather than
widening the accumulator**: divide by the population's max-abs magnitude before
centering, so every scaled value lies in `[−1, 1]`, the mean in `[−1, 1]`, and
every squared centered term in `[0, 4]`. For a **finite** population neither the
mean nor the variance can then overflow at any size that fits in memory, and
z-scoring is invariant to a positive rescale, so the result is unchanged.

**The bound is conditional, and saying otherwise would repeat this ADR's own
mistake.** `max_abs` is non-finite exactly when the population carries a raw
`±∞` — and a `−∞` member is *legal* input here, being ADR 0034's worst-value
sentinel. The implementation therefore falls back to `scale = 1.0` for a
non-finite or zero max, reproducing the pre-ADR-0069 arithmetic bit-for-bit, so
the `[0, 4]` bound does **not** hold for such a population. That fallback is
deliberate: it keeps this ADR's overflow fix from silently changing the `−∞`
semantics, which is a separate policy question tracked as **#1068** and pinned by
a test marked "Pin, not a fix". An unqualified "cannot overflow for any
population" here would be a second over-strong guarantee of exactly the kind
this ADR's own Context section indicts.

This is **strictly stronger** than `f64` accumulation would have been. It also
survives a narrower `B::FloatElem` — an `f16` backend overflows the current
formula at fitness $\approx 256$ — and it removes the overflow at its actual source. The
squared term overflows at `N = 1`, *before* any accumulation, so accumulator
width was never the binding constraint at this site. A hybrid (host-`f64`
reduction, device-`f32` elementwise application) was also rejected and is worth
recording as rejected: for `[f32::MAX, f32::MAX, −f32::MAX]` the on-device
`centered` term is `−4.54e38`, which overflows `f32` no matter how the mean was
obtained.

Generalised, and binding on future sites:

> **This ADR's own Decision 1's rule governs host iterator reductions. A
> device reduction satisfies this ADR by bounding its terms, because its
> accumulator width belongs to the backend, not to the caller.**

`centered_rank` is untouched — it sanitizes for *ordering* only, which this
ADR's own Decision 1 explicitly excludes.

### 5. The mechanical net is a rescale-invariance property test, not a source-text guard

The invariant: **a fitness reduction is invariant to a positive rescale of the
whole fitness vector, and a saturated population is not a special case.** As
proptests in the `rlevo-evolution` `proptest` style (ADR
[0036](0036-adopt-proptest-for-property-tests.md)):

- `allocate_offspring(speciate(f))` is unchanged when every member of $f$ is
  scaled by $c$, for $0 < c \le \texttt{f32::MAX} / \max(f)$. **The upper bound is required, and this
  bullet originally omitted it.** Once `c` is large enough that members overflow
  and ADR 0034 clamps them, *every* clamped member ties at `f32::MAX` —
  proportionality is destroyed by the clamp itself, by design, and no accumulator
  width restores it, so the equality is simply false of correct code there. The
  intent survives: at that boundary members sit within a factor of two of
  `f32::MAX`, which is precisely the regime where both `f32` accumulators
  overflow. The clamped case is covered instead by the example test
  `test_allocate_offspring_poisoned_species_keeps_proportionality`, whose expected
  answer is the one a *tie* yields.
- `from_host_fitness` reports a finite `mean_fitness` whenever every member is
  finite — including an all-`f32::MAX` population.
- `z_score` output is invariant to a positive affine rescale of its input —
  **bounded**, not unconditional, and by *two* independent bounds. The first:
  strict invariance is false by design once `c` is small enough that the `1e-8`
  standard-deviation floor fires, giving $\texttt{STD\_FLOOR} / \sigma \le c \le \texttt{f32::MAX} / \max|x|$
  where $\sigma$ is the raw population std. The second was found only by the property
  **failing on correct code** during implementation: under $c \cdot x + d$ with the
  offset applied *after* the scale, `xs = [0, 1]`, $d = 1$, $c = 2^{-24}$ rounds both
  members to exactly `1.0` in `f32`, so the population genuinely becomes
  degenerate and `z_score` rightly returns zeros. That is a representability limit
  of the *input*, not a property of `z_score`, and it is resolved structurally
  rather than with a tolerance: write the transform as $c \cdot (x + d)$, which for
  integer `x`, integer `d` and power-of-two `c` is computed with no rounding at
  all. (The second clamp, `.max(f32::MIN_POSITIVE)`, looks like it should add a
  third bound on `c`; it does not — it fires iff $\sigma / M < \texttt{f32::MIN\_POSITIVE}$, which
  depends only on the shape of `x`.)

**The properties must construct the extreme case, not wait to generate it.** This
decision originally implied that keying on behaviour is sufficient on its own. It
is not: for properties 1 and 3 the defect lives only at the very top of the `f32`
range, and for property 2 only at an all-`f32::MAX` population — regions a uniform
generator visits too rarely for mutant detection to be reliable within a CI run.
All three therefore assert the extreme deterministically on *every* case
(`2^k_max`; `vec![f32::MAX; n+1]`) alongside the generated point. That is why each
mutant fails at `successes: 0` rather than after dozens of cases. Behavioural
keying is what makes the net catch a site regardless of how it spells its
arithmetic; deliberate visiting of the saturation regime is what makes it catch
the site at all.

Use a power-of-two `c` so the invariance is **bit-exact** — power-of-two scaling
commutes with round-to-nearest through every intermediate — and assert counts with
`==` rather than a tolerance that could hide drift.

**A source-text guard in the ADR 0068 / `rng_seeding_guards.rs` shape is
rejected, and the reason is the point of this ADR's own Context section.**
Such a guard must key on a name, and the only available name is
`sanitize_fitness`. Two of the four
mis-widened sites — `allocate_offspring` and `z_score` — **do not contain that
name**, and they are the two that a review pass keyed on the name has already
missed twice. A ~900-line guard that provably misses the harder half of its own
known population is guard theatre: it would have gone green through both #132
and #1062. The property test keys on *behaviour*, so it catches a site regardless
of how it spells its arithmetic.

**A clippy lint or a `#[deny]` is not available.** No lint expresses "this `f32`
accumulator's terms came from fitness"; `clippy::cast_precision_loss` and friends
fire on the narrowing, not on the overflow, and the workspace already
`#[allow]`s them at these exact sites. Recorded so it is not re-proposed.

## Alternatives considered

- **`rules.md` entry plus per-site comments, no ADR.** The strongest rejection to
  argue against, because it is the cheap answer and rules.md is where a
  contributor actually looks. Rejected on two grounds. (a) **ADR immutability
  leaves no other mechanism.** ADR 0034's stated rationale is false and it is
  load-bearing prose — a contributor read it and wrote a defect. rules.md can
  state a rule; only an ADR can correct the record of a decision, and this
  repository forbids the alternative of editing 0034. (b) **rules.md was already
  tried and it was not enough.** `rules.md`'s "Optimisation direction" section
  already carried the fitness-hygiene convention when #1062 was written, and
  it carried the *same false clause* — a rules.md-only remedy would have
  propagated the error into the document meant to prevent it. Note the choice
  is not exclusive: this ADR *mandates* the rules.md entry (this ADR's own
  Decision 3), following ADR 0068's shape of ADR + rules.md line + one
  mechanical check.

  On weight: this is the granularity at which this workspace already writes
  ADRs. 0056 (non-finite loss skip), 0060 (config values must be finite), 0065
  and 0067 (non-finite reward / observation dropped at replay ingestion), 0066
  (`clamp` `NaN` behaviour is backend-specific) are all single numeric-hygiene
  rules with their own record. An ADR here is *in keeping*, not an escalation.

- **Supersede ADR 0034.** Rejected as dishonest labelling with concrete costs.
  0034's Decision 1–5 are all still in force: the sanitization rule, the four
  chokepoints, the mean-over-finite metric, and the contract amendments. Marking
  it `Superseded` tells every reader the decision is retired, which is a *worse*
  falsehood than the parenthetical it would be correcting — and it strands ADR
  0023 (which 0034 extends and which stays `active`), `rules.md`'s section
  3's citation of 0034, and the ~7 in-source `ADR 0034` references. It would
  also force this ADR to restate the whole sanitization rule to keep the
  record self-contained, creating two documents that state one rule and can
  drift.

  0034 supplies its own precedent for the correct label: it *extends* ADR 0023
  without superseding it, and says so in its Status section in exactly the
  spelling reused here. The one thing "extends" does not do is warn a reader who
  opens 0034 directly — which is why this ADR's own Decision 3 puts the
  pointer on the index and the corrected text in the rustdoc, and why the
  residual risk is recorded under this ADR's own Consequences section rather
  than papered over.

- **A `SaneFitness(f32)` newtype not implementing `Sum<f32>`,** forcing an
  explicit widening at every reduction. The most attractive rejection, and the
  only option that would make the defect a compile error. Rejected on blast
  radius, and specifically on *where* that radius lands.

  To catch `allocate_offspring` the newtype must be **transitive through stored
  state** — a wrapper only at the `sanitize_fitness` return is exactly the
  name-keyed coverage that missed that site. Transitivity means the 79 stored
  `f32`/`Vec<f32>` fitness fields in `rlevo-evolution` (the `Species` pair,
  `NeatState::fitness`, the per-slot metaheuristic caches from #131, the
  hall-of-fame entries) and, decisively, `StrategyMetrics`' public fields.
  `mean_fitness` alone is read in **five crates**: `rlevo-evolution`,
  `rlevo-benchmarks`, `rlevo`, the `#![no_std]` `rlevo-metrics-registry`, and the
  WASM `rlevo-benchmarks-report-client`, whose wire types mirror the benchmark
  crate's by hand (ADR 0015). That makes it a **one-way door on a cross-crate
  wire surface** — the workspace's own convention of flagging one-way-door
  changes for extra scrutiny (used the same way in, e.g., ADR 0028's "Public
  trait shape is a one-way door. Flagged for scrutiny") — bought for a
  defect class this ADR's own Decision 2 closes at the source for three call
  sites.

  It also would not actually forbid the bug: `.0` is one character, and a
  newtype that must be unwrapped at 90 sites trains its readers to unwrap
  reflexively. Revisit under the reopen triggers below; the arithmetic changes if
  the reduction population grows or leaves the crate.

- **Lower the `+∞` sentinel below `f32::MAX`** (e.g. `f32::MAX / 1024`) so small
  sums survive. Rejected, and worth recording because it is the intuitive repair:
  it does not achieve the property (this ADR's own Context section — no
  fixed finite `S` works for unbounded `N`), it silently caps legitimate
  large finite fitness by making some real values indistinguishable from
  `+∞`, and it changes a *value* convention
  that four chokepoints and ~90 call sites already depend on, to avoid a
  one-word change to an accumulator declaration.

- **Fire ADR 0034's `CanonicalFitness<B>` trigger.** Rejected as a
  mis-diagnosis: 0034's trigger is the *bypass* hole, which has not fired, and
  its remedy guards a `Tensor<B, 1>` at the `tell` boundary while all four
  instances are host-side `f32` scalars downstream of it (this ADR's own
  Context section). It would have caught none of them. 0034's deferral is
  neither invoked nor retired here.

## Consequences

### Positive

- **The mechanism is attributed to the right thing.** "Sanitize, therefore the
  mean is safe" is retired; "sanitize the value, widen the reduction" replaces it,
  in the one place — the `sanitize_fitness` rustdoc — where a contributor about to
  write a fitness reduction will read it.
- **A third instance is closed before it shipped a call site.** `z_score`'s
  zero-gradient collapse is the worst of the four (it fires at `N = 1` and its
  output is finite and plausible) and was found only because this ADR's
  enumeration keyed on the defect *shape* rather than on `sanitize_fitness`.
- **The rule is behaviourally checked**, not conventionally asserted. A future
  site that gets the width wrong fails a proptest that names this ADR, regardless
  of how it spells its arithmetic.
- **`f64` accumulation also removes the ~1 ULP-per-addition drift** an `f32` sum
  accrues over a large population — a real if secondary gain, already noted in
  `from_host_fitness`.
- No public signature change, no serialized-form change, no numerics change for
  any population whose fitness is not saturated.

### Negative / accepted costs — do not soften these

- **A reader who opens ADR 0034 directly still reads the false sentence.** The
  immutability rule and the correction are in genuine tension, and this ADR
  resolves it in immutability's favour while accepting the residual. The
  mitigations are indirect on purpose (index clause, rustdoc, rules.md) and none
  of them intercepts a direct link to
  `0034-fitness-hygiene-chokepoint-convention.md`'s clamp-mapping decision
  (Decision 1). If this shape recurs — a second accepted ADR whose
  *rationale* needs correcting while its *decision* stands — the convention
  itself should be revisited: a permitted, append-only `## Corrections`
  section would resolve it, and that is a change to `CLAUDE.md` and this
  index's header, not something to smuggle in under a numeric-hygiene ADR.
- **Two widths now exist for one quantity.** Fitness is stored and reported in
  `f32` and reduced in `f64`. Every reduction site therefore carries a narrowing,
  and each narrowing is a place a future refactor can "simplify" back to `f32`.
  This ADR's own Decision 2's named primitives are the mitigation — the
  narrowing lives inside `sanitized_mean` and nowhere else — but
  `allocate_offspring`'s `f64` `total`
  divided into `f32` shares is a genuinely mixed-width function and will read as
  untidy.
- **The property tests are weaker than a type.** They cover the three sites
  enumerated today. A *fourth* reduction added tomorrow is covered by nothing
  mechanical until someone writes its property — which is precisely the gap that
  produced #1062, now narrowed rather than closed. This is the cost of rejecting
  the newtype and it should not be described as anything else.
- **The enumeration is a fact as of 2026-08-08, not a law.** This ADR's own
  Context section's "five sites, four wrong" is a grep-and-read of one crate
  on one day, and this ADR's own Decision 5's choice of a per-site property
  test over a type rests on the population being small and closed. It is
  exactly the kind of claim ADR 0068's Assumptions section warns is
  load-bearing.

### Neutral

- No new dependency; `proptest` is already an `rlevo-evolution` dev-dependency
  (ADR 0036) and this is an input-space invariant, which is what this ADR's
  own Decision 5's properties are (unlike ADR 0068's source property).
- `sanitize_fitness` and `sanitize_fitness_tensor` are byte-for-byte unchanged,
  as are all four ADR 0034 chokepoints. `StrategyMetrics`' fields, the
  `rlevo-metrics-registry` table, and the report wire types are untouched.

## Assumptions and the reopen triggers

**Review trigger — any one of these reopens this ADR:**

1. **A fourth mis-widened reduction site**, anywhere. Three is the population
   this ADR's own Decision 5 sizes its per-site properties against; four
   means the properties are not keeping pace with the sites and the
   type-level answer earns its price.
2. **A reduce-over-fitness site outside `rlevo-evolution`** — most plausibly in
   `rlevo-hybrid`'s `RolloutFitness` or in `rlevo-benchmarks`' aggregation. This
   crosses the crate boundary the primitives' `pub(crate)` visibility assumes and
   would force either promoting them or duplicating the rule.
3. **Fitness storage widening to `f64`,** which would dissolve the problem
   entirely and retire most of this ADR.

Trigger (1) or (2) is what would revive the `SaneFitness` newtype, on the
arithmetic this ADR's own Alternatives-considered section gives: its cost is
paid once per *stored fitness field* and per *cross-crate consumer*, while
its benefit scales with the *reduction* population. At 3 reductions against 79 fields and 5 crates it does not earn its
keep; the ratio, not the principle, is what was decided.

## References

- Issue **#132** / PR **#1061** — the first instance
  (`StrategyMetrics::from_host_fitness`), fixed by widening the accumulator to
  `f64`. Its in-file comment already documents this ADR's mechanism, locally.
- Issue **#1062** — the second and third instances (`speciate`,
  `allocate_offspring`), verified by execution to collapse NEAT's
  fitness-proportional apportionment population-wide (`[27, 3, 0] → [10, 10,
  10]`), with `allocate_offspring`'s `total` overflowing independently of the
  per-species means.
- Issue **#131** — the per-slot fitness caches whose stored `f32` fields are part
  of this ADR's own Alternatives-considered section's newtype blast radius.
- ADR [0034](0034-fitness-hygiene-chokepoint-convention.md) — the sanitization
  rule and its four chokepoints, **extended and preserved**; its
  clamp-mapping decision (Decision 1)'s parenthetical, corrected here; its
  Consequences section's `CanonicalFitness<B>` deferral, neither invoked nor
  retired.
- ADR [0023](0023-objective-sense-and-maximize-convention.md) — the
  maximise-native canonical convention; the "extends, does not supersede"
  precedent this ADR's Status reuses.
- ADR [0068](0068-bounds-strictness-enforcement-is-crate-asymmetric.md) — the
  ADR + `rules.md` line + one mechanical check shape adopted here; its
  Context section's "trains and reports finite numbers" failure class; its
  Consequences section's "the check acquires a name" reasoning reused by
  this ADR's own Decision 2; and the source-text guard shape this ADR's own
  Decision 5 deliberately declines.
- ADR [0036](0036-adopt-proptest-for-property-tests.md) — why this ADR's own
  Decision 5's invariants are proptests (input-space) rather than a guard
  test (source property).
- ADR [0015](0015-shared-typed-metric-registry-crate.md) — the hand-mirrored wire types that
  make `StrategyMetrics`' field widths a cross-crate, one-way-door concern.
- ADR [0060](0060-config-values-must-be-finite.md), ADR
  [0065](0065-non-finite-reward-is-dropped-at-replay-ingestion.md), ADR
  [0066](0066-clamp-nan-behavior-is-backend-specific-pin-with-is-nan.md), ADR
  [0067](0067-non-finite-observations-are-dropped-at-replay-ingestion.md) — the
  established precedent that a single non-finite-numerics rule gets its own
  record, which is the granularity argument in this ADR's own
  Alternatives-considered section.
- `docs/rules.md`'s "Optimisation direction" section — gains this ADR's own
  Decision 1's corollary bullet; its existing one-line paraphrase of the `+∞`
  rationale is corrected.
- `docs/adr/README.md` — 0034's row gains the appended correction clause
  (this ADR's own Decision 3).
- Code — the primitive and its rustdoc:
  `crates/rlevo-evolution/src/fitness.rs` (`sanitize_fitness`,
  `sanitize_fitness_tensor`, and this ADR's own Decision 2's
  `sanitized_mean` / `sanitized_sum`).
- Code — the four instances: `crates/rlevo-evolution/src/strategy.rs`
  (`StrategyMetrics::from_host_fitness`),
  `crates/rlevo-evolution/src/neuroevolution/species.rs` (`speciate`,
  `allocate_offspring` — cited by function name because the file is in flight
  under #1062), `crates/rlevo-evolution/src/shaping.rs` (`z_score`; `centered_rank`
  is ordering-only and out of scope).
- Code — the defensive counter-example:
  `crates/rlevo-evolution/src/algorithms/gep/strategy.rs` (`roulette_select`'s
  `!total.is_finite()` fallback — the only site that anticipated the overflow,
  and a degradation rather than a fix).
