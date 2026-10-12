---
project: rlevo
status: active
type: decision
date: 2026-10-12
tags: [adr, decision, architecture, evolution, fitness-hygiene, public-surface]
---

# ADR 0086: The fitness sanitisers are public

## Status

**Accepted (2026-10-12).**

Supersedes one clause of ADR
[0034](0034-fitness-hygiene-chokepoint-convention.md), Decision 1: "Both
stay `pub(crate)`: every caller is in `rlevo-evolution`." The rest of that
ADR stands: the rule, the two functions that apply it, the four driver
chokepoints and the per-site floor of its Decision 3.

ADR [0069](0069-sanitized-fitness-is-reduced-in-f64.md), Decision 2, keeps
`sanitized_mean` and `sanitized_sum` at `pub(crate)`, and names what would
reopen that: a reduction over fitness outside `rlevo-evolution`. There is
none. That decision stands and its trigger has not fired.

## Context

ADR 0034 gave the crate one rule for a fitness value that is not finite,
and two functions that apply it in `rlevo_evolution::fitness`:

- `sanitize_fitness(f32) -> f32`: `NaN` becomes `f32::NEG_INFINITY`,
  $+\infty$ becomes `f32::MAX`, $-\infty$ and every finite value pass through;
- `sanitize_fitness_tensor<B>(Tensor<B, 1>) -> Tensor<B, 1>`: the same rule
  over a whole population, as one device operation.

It kept both private on the count of callers, and the count was right:
every call is inside the crate. It is not a count of who is told to apply
the rule. Two readers outside the crate are, by the crate's own documents.

**Someone who writes a `Strategy`.** `Strategy` is a public trait and the
seam an algorithm author implements. Its `tell` can be reached without
`EvolutionaryLoop`, by a unit test or by a hand-written driver, so ADR 0034
Decision 3 and `docs/rules.md` section 3 require a `tell` that pulls
fitness to host to sanitise it there, once, before any sort, argmax or
best-so-far write. Inside the crate that is `.map(sanitize_fitness)`.
Outside it, `tell`'s documentation said to apply "the same two replacements
by hand".

**Someone who writes the loop.** `docs/ask-tell-contract.md` teaches the
ask/tell protocol by writing the loop out, and names the cases that need a
hand-written one: an evaluation that cannot happen inside one synchronous
call, or one loop driving several strategies. That loop owes `tell` what
`EvolutionaryLoop::step` gives it: a `Minimize` objective negated, then
sanitised. The guide once sent that reader to `sanitize_fitness`, which
they could not call. A review of the guide found it, and the guide was
changed to give the rule and its order in words.

A rule given in words is a rule copied by hand, and this one has three
places to go wrong, each of which the crate has met:

- `f32::NAN` is a positive NaN, so `total_cmp` ranks it as the maximum. A
  copy that forgets the `NaN` case crowns a broken individual.
- $-\infty$ must pass through. A copy that clamps both ends to finite values
  hides a broken member from `StrategyMetrics`, which counts the $-\infty$
  ones.
- On a tensor, the order of the two operations decides the result. A
  `clamp` reached by a `NaN` does not have one behaviour across Burn
  backends (ADR
  [0066](0066-clamp-nan-behavior-is-backend-specific-pin-with-is-nan.md)),
  so the `NaN` mask must come first. A copy that clamps first is correct on
  one backend and not on another.

The maintainer's decision, on review of the change that made
`EvolutionaryLoop` (ADR
[0085](0085-the-generation-loop-is-named-evolutionary-loop.md)): make the
functions public.

## Decision

1. **`rlevo_evolution::fitness::sanitize_fitness` and
   `rlevo_evolution::fitness::sanitize_fitness_tensor` are `pub`.** Their
   signatures and behaviour are unchanged. `sanitize_fitness` gains
   `#[must_use]`, which its sibling had. A consumer of the umbrella writes
   `rlevo::evo::fitness::sanitize_fitness_tensor`. Neither is re-exported
   at the crate root or placed in a prelude: they are for the two readers
   above, and a run driven by `EvolutionaryLoop` calls neither.

2. **The rule becomes part of the public contract.** `NaN` to
   `f32::NEG_INFINITY`; $+\infty$ to `f32::MAX`; $-\infty$ and finite values
   unchanged. The input is canonical: higher is better. A change to any of
   the three is a breaking change from here on.

3. **Each function's documentation says who calls it.**

   | Reader | Calls | Where |
   |---|---|---|
   | the author of a `Strategy` | `sanitize_fitness` | in `tell`, on the `Vec<f32>` pulled to host, before anything orders or stores it |
   | the author of a hand-written ask/tell loop | `sanitize_fitness_tensor` | on the fitness tensor, after negating a `Minimize` objective and before `tell` |
   | anyone running a shipped strategy through `EvolutionaryLoop` | neither | the loop has already done it |

   Both also state the precondition the types do not carry: negate before
   sanitising. A `NaN` cost sanitised first becomes $-\infty$, and the
   negation then makes it $+\infty$, the best member of the population.

4. **`sanitized_mean` and `sanitized_sum` stay `pub(crate)`.** A strategy
   written outside the crate does not reduce over fitness itself: the
   statistics its `tell` returns come from
   `StrategyMetrics::from_host_fitness`, which is public and accumulates in
   `f64`. The public documentation of `sanitize_fitness` says that a
   clamped value does not bound a sum, and points there.

5. **The documents that gave the rule in words name the function.**
   `Strategy::tell`, the recipe in `rlevo_evolution::probe` for a driver of
   one's own, `docs/rules.md` section 3, and the guide. The guide's
   hand-written loop now reads `sanitize_fitness_tensor(cost.neg())`, so
   its body is what one `EvolutionaryLoop::step` does, line for line.

## Consequences

**Positive.**

- One rule and one implementation of it, for code outside the crate as
  well as inside. A strategy or a driver written elsewhere cannot drift
  from the crate's hygiene, because it calls it.
- The guide's claim that `EvolutionaryLoop` is the hand-written loop with
  bookkeeping is now exact, and a test holds it:
  `crates/rlevo-evolution/tests/hand_written_loop.rs` defines a strategy
  outside the crate, drives it once through the loop and once by hand, and
  compares what every `tell` received, bit for bit, on a fitness function
  that returns a `NaN`, a $-\infty$ and a $+\infty$ cost in every
  generation.
- That test is also the workspace's first `Strategy` implemented outside
  `rlevo-evolution`'s own source, so the trait's outside path is compiled
  by something.

**Costs.**

- Two public functions. They pass the test of ADR
  [0081](0081-a-public-item-needs-an-implementor-and-a-consumer.md)
  Decision 2, that a public function is one a user writes the name of, on
  the two readers in the table. They replace two paragraphs of prose.
- The rule is frozen. Moving the $+\infty$ sentinel off `f32::MAX`, or
  mapping `NaN` anywhere else, now breaks callers.
- A public function with a precondition its type cannot check. Nothing
  stops a caller from sanitising a cost and negating it afterwards. The
  documentation of both functions, the guide and the test's own comment say
  not to. A type that carried the direction would stop it, and ADR 0069
  weighed a fitness newtype and declined it on its cost across 79 stored
  fields.

**Unchanged.**

- Every call site inside the crate, and what any run computes.
- `sanitized_mean` and `sanitized_sum`, the four chokepoints of ADR 0034
  and the per-site floor.

**Surface delta**, from `tools/public-surface.sh rlevo-evolution` with all
features on. Two functions enter:
`rlevo_evolution::fitness::sanitize_fitness` and
`rlevo_evolution::fitness::sanitize_fitness_tensor`. The listing grows by
three lines of kind `function`, from 138 to 141 and from 1,222 lines to
1,225. The third is `StrategyMetrics::from_host_fitness`, an inherent
method that was public already and that the tool lists a second time, by
path, once public documentation links to it.

**How it was checked.** Each function has a doctest that calls it by its
public path. The hand-written-loop test above fails under two mutants of
its own loop: without the sanitise call, and with the call made before the
negation, where the `NaN` cost reaches `tell` as $+\infty$. The crate's
documentation builds with `private_intra_doc_links` denied, a lint that
rejects a link from public documentation to a private item. The
documentation of `sanitize_fitness` had three, written while it was
private: `sanitized_mean`, `sanitized_sum` and the local searchers'
evaluator. They are named in prose now.

## Alternatives considered

**Keep them private and keep the rule in words.** This is what the guide
did for one revision. Rejected by the maintainer, and for the three ways a
copy goes wrong listed under Context. The tensor form in particular is not
a line one can safely retype: its correctness depends on the order of two
operations, for a reason that shows on some backends and not others.

**Publish only `sanitize_fitness_tensor`,** since a loop is what the guide
teaches. Rejected. The author of a `Strategy` is the reader the contract
binds: `tell`'s documentation requires the scalar map of every
implementation, and that reader had only the prose.

**Publish `sanitized_mean` and `sanitized_sum` as well.** Not done; see
Decision 4. ADR 0069 names the event that would justify it, and it has not
happened.

**Have `Strategy::tell` take a sanitised type,** so that no implementor
has to remember. Not done here. It is the fitness newtype ADR 0069
declined, and it would change the signature every strategy implements.

**Re-export both at the crate root.** Not done. Most users never call
them, and the crate root is where a newcomer looks first.

## References

- ADR [0023](0023-objective-sense-and-maximize-convention.md), the
  maximise-native contract the rule belongs to.
- ADR [0034](0034-fitness-hygiene-chokepoint-convention.md), whose
  Decision 1 this supersedes in one clause.
- ADR [0066](0066-clamp-nan-behavior-is-backend-specific-pin-with-is-nan.md)
  and ADR [0069](0069-sanitized-fitness-is-reduced-in-f64.md).
- ADR [0081](0081-a-public-item-needs-an-implementor-and-a-consumer.md).
- ADR [0085](0085-the-generation-loop-is-named-evolutionary-loop.md), the
  review of which raised the question.
- `docs/rules.md` section 3; `docs/ask-tell-contract.md`.
- Code: `crates/rlevo-evolution/src/fitness.rs`;
  `crates/rlevo-evolution/src/strategy.rs` (`Strategy::tell`);
  `crates/rlevo-evolution/tests/hand_written_loop.rs`.
