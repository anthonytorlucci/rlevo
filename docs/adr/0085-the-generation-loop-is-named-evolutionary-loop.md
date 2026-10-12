---
project: rlevo
status: active
type: decision
date: 2026-10-12
tags: [adr, decision, architecture, naming, evolution, coevolution, hybrid, breaking-change]
---

# ADR 0085: The generation loop is named `EvolutionaryLoop`

## Status

**Accepted (2026-10-12).**

Renames two types that eleven earlier ADRs name: `EvolutionaryHarness` and
`CoEvolutionaryHarness`. Those ADRs stand as written. Where one says
`EvolutionaryHarness`, read `EvolutionaryLoop`; where one says
`CoEvolutionaryHarness`, read `CoEvolutionaryLoop`; where one says "the
harness" of something that calls `Strategy::ask` and `Strategy::tell`, read
"the loop". Two chokepoints keep their place under the new names.
`EvolutionaryLoop::new` is one of the construction chokepoints of ADR
[0026](0026-shared-config-validation-convention.md), where a configuration
is validated as it is consumed. `EvolutionaryLoop::step` is the "harness
chokepoint" of ADR [0034](0034-fitness-hygiene-chokepoint-convention.md),
where fitness is sanitised before `tell`. ADR 0034 also cites
`src/coevolution/harness.rs`, which is now
`src/coevolution/generation_loop.rs`.

Supersedes one clause of ADR [0077](0077-remove-benchenv.md), which records
that `BenchStep` "was already displaced by `GenerationStep` on the
evolutionary side". `GenerationStep` is deleted here. ADR
[0001](0001-keep-environments-and-benchmarks-separate.md) quotes a call
site, `harness.step(()).done`; that call is now `evo_loop.step()` followed
by `evo_loop.is_done()`.

Closes a cost that ADR
[0084](0084-the-umbrella-carries-the-evaluation-harness.md) recorded: that
"harness" names two things until the evolution crate's generation drivers
are renamed.

## Context

Two things were called a harness.

`rlevo-benchmarks` is the evaluation harness: `Suite`, `Evaluator`, the
metrics and the reporters. Since ADR 0084 a consumer writes it
`rlevo::harness`.

`rlevo_evolution::strategy::EvolutionaryHarness` owned the state, the seed
and the generation budget of one strategy run, and ran ask, evaluate and
tell once per call. `CoEvolutionaryHarness` did the same for two coupled
populations. Neither evaluates an agent on an environment, registers
anything in a suite or reports a result. They are loops. After ADR 0084 a
consumer saw `rlevo::harness` and
`rlevo::evo::strategy::EvolutionaryHarness` in the same crate, and the
evaluator is behind only the first.

The driver's `step` also carried a shape it no longer needed:

```rust
pub fn step(&mut self, _action: ()) -> GenerationStep
pub struct GenerationStep { pub reward: f64, pub done: bool }
```

The unit argument and the `reward` and `done` pair are what an environment
step looks like. The drivers took that shape when they implemented
`BenchEnv`, so that the evaluator could run a generation loop as if it were
an environment. ADRs
[0076](0076-trial-seam-splits-machinery-from-trial-shape.md) and 0077
removed `BenchEnv`: the evaluator drives a loop through
`GenerationProbe::advance`, which returns the generation's typed metrics.
The environment shape stayed behind on the inherent method.

Counted at `42a7595`, the commit before this change, with
`git grep -F 'step(())' 42a7595 -- crates` and each match read:

- `step(())` is written 78 times in 35 files under `crates/`. 68 of them
  read `done` and nothing else. 7 discard the result. 2 read `reward`
  alone and 1 reads both, and all three are unit tests of the two drivers.
- The name `GenerationStep` is written in three files: the two that define
  and return it, and the crate root that re-exports it. No test, bench or
  example names it.
- `GenerationProbe::advance` discards the value: it calls `step(self, ())`
  and returns a clone of `latest_metrics()`.
- `rlevo_hybrid::PolicyNeuroevolution::step` returned
  `self.harness.step(()).done`, a bare `bool`.
- The budget comparison, `generation >= max_generations`, is written four
  times: in each loop's `step` and in each loop's `advance`.
- 30 of the 78 call sites are `while !harness.step(()).done {}`, which runs
  the work and then asks. With a budget of zero it runs one generation.

So "advance one generation" had three shapes: `step(()) ->
GenerationStep`, `advance() -> Option<Metrics>` and `step() -> bool`.

The workspace already has the words. `docs/rules.md` section 2 sorts
method names into kinds: `reset()` and `step()` are mutation methods,
`is_done()` is a predicate, and an episode loop asks `Snapshot::is_done()`.
It does not have one shape for every call that advances something, and
this ADR does not give it one. `Environment::step` returns a `Snapshot`
that carries the status, and `GenerationProbe::advance` returns `None` at
the end; each answers a generic evaluator and returns more than a flag.
`GenerationStep` was neither of those. It was a struct built to carry a
flag and a number that nothing read.

The names are about to be copied. Whoever writes a `Strategy` copies the
call that runs it, and that call names the driver and calls `step`. A name
stamped into an author's crate is one they keep, so the call is settled
before a how-to for strategy authors is written.

## Decision

1. **`EvolutionaryHarness` is renamed `EvolutionaryLoop`, and
   `CoEvolutionaryHarness` is renamed `CoEvolutionaryLoop`.** Their type
   parameters, constructors and other methods are unchanged.
2. **`step` takes no argument and returns nothing, and `is_done` is a
   method of its own**, on both loops.

   ```rust
   pub fn step(&mut self)
   pub const fn is_done(&self) -> bool   // generation >= max_generations
   ```

   A caller that wrote `while !harness.step(()).done {}` writes

   ```rust
   while !evo_loop.is_done() {
       evo_loop.step();
   }
   ```

   Advancing a run and asking whether it is over are separate for three
   reasons.

   - *The loop reads as what it does.* The alternative, `step() -> bool`,
     is written `while !evo_loop.step() {}`, and the reader has to know
     which way the flag points.
   - *It uses the names the workspace already has.* `rules.md` section 2
     lists `step()` among mutation methods and `is_done()` among
     predicates, and `is_done` is the question an episode loop asks of a
     `Snapshot`.
   - *The check comes before the work.* A budget of zero runs no
     generation. `GenerationProbe::advance` becomes `is_done`, then `step`,
     then the metrics, so the comparison written four times is written
     once per loop, and the evaluator's guarded form is visibly built from
     the hand-driven one.

   `step` does not consult the budget. Called when `is_done` is already
   `true`, it runs one more generation, as it did before this change. The
   budget is the point at which `is_done` turns `true`; the loop that calls
   `step` is what stops there. A `step` that checked would have to do
   nothing, and then a call that looks like work is not work, or panic, and
   then a run cannot be carried on a little past its budget when that is
   what the caller means. Nothing in the workspace does that today: with a
   panic placed there, the default test targets of `rlevo-evolution`,
   `rlevo-hybrid` and `rlevo` trip it once, in the test written to pin this
   behaviour. It is kept because both alternatives are worse, not because a
   caller needs it. The evaluator never overruns, because `advance` asks
   first.
3. **`GenerationStep` is deleted, and `reward` with it.** The value has no
   field of its own on the loop, and none is needed:
   - On `EvolutionaryLoop` it was the canonical `best_fitness_ever`.
     `latest_metrics()` gives `best_fitness_ever()` in the objective's
     declared sense.
   - On `CoEvolutionaryLoop` it was `binding_fitness`, which is a field of
     `CoEAMetrics` and stays canonical.
4. **The module `rlevo_evolution::coevolution::harness` is renamed
   `coevolution::generation_loop`.** `CoEAMetrics` and
   `CoEvolutionaryLoop` are re-exported from `coevolution` as before.
5. **No deprecated alias is kept for either type.** An earlier draft of
   this decision kept
   `EvolutionaryHarness` and `CoEvolutionaryHarness` as deprecated type
   aliases for one release. A type alias cannot carry Decision 2: a program
   that advances the loop calls `step(())`, and that call stops compiling
   whether or not the old type name resolves. The alias would save an edit
   only in code that builds a loop and never runs it. It would also add
   two public items that exist so as not to be used, against the test of
   ADR [0081](0081-a-public-item-needs-an-implementor-and-a-consumer.md)
   that a public item has a consumer. ADR
   [0083](0083-episode-guard-is-a-core-contract-helper.md) declined a
   second public name for `EpisodeGuard` for the same reason.
6. **"Harness" names the evaluation harness only.** In `rlevo-evolution`
   and `rlevo-hybrid`, and wherever another crate's documentation describes
   them, the thing that runs generations is "the loop". This covers
   documentation, test names and local variable names, not only the two
   types, so that a search for the word finds one thing. `NeatStrategy` and
   `ArchNasStrategy`, which their documentation called "custom harnesses",
   are described as standalone strategies: they do not implement
   `Strategy`, and no loop type drives them.
7. **`PolicyNeuroevolution` follows.** Its field and documentation name the
   loop. Its `step` returns nothing and it gains `is_done`, so the three
   types a person drives share one shape. Its `run`, which resets and steps
   to the budget, is unchanged. It has no `GenerationProbe` impl, so the
   evaluator cannot run it; nothing hands it to one yet.
8. **`docs/rules.md` follows.** Section 2 states Decision 6 as a naming
   rule, and Decision 2 as the shape a later driver takes: `step()` and
   `is_done()` for a person, a `GenerationProbe` impl for the evaluator,
   the second built from the first. Section 3's optimisation-direction rule
   names the loop. Section 8's seeding rule, which forbids re-seeding in
   `reset()`, gains the exception the loops have always been: a loop's
   `reset` starts a new run and re-seeds on purpose.

## Consequences

**Positive.**

- One word, one meaning. `rlevo::harness` is the only harness a consumer of
  `rlevo` meets.
- "Advance one generation" has two shapes where it had three, and the
  second is built from the first: `step()` beside `is_done()` on the two
  loops and on `PolicyNeuroevolution`, and `advance() -> Option<Metrics>`
  on the probe, which is `is_done`, `step` and the metrics.
- A hand-written driving loop checks the budget before it runs, as the
  evaluator always did. A budget of zero now runs no generation by either
  route; `while !harness.step(()).done {}` ran one.
- An evolutionary run and an episode stop on the same word.
- The documentation says what the loop does, and why. The loops' own
  documentation gives the six things one `step` does and the reason `step`
  reports no status; `rlevo_evolution::probe` sets the two ways of driving
  a loop side by side. Three things it said were wrong and are corrected.
  Several passages described a `reward` handed to the evaluator, which has
  read none since `BenchEnv` was removed. The guide to the ask and tell
  contract listed early stopping and parallel evaluation among the things
  the driver adds, and it does neither. And that guide's hand-written loop
  passed a cost to `tell` without negating it. Run as printed (a genetic
  algorithm at its defaults on the 8-dimensional Sphere, population 64,
  seed 42, 300 generations, the `Flex` backend), it climbed from a best
  value of 121.06 to 209.72, the worst corner of the domain, where the
  corrected loop descends to a cost of 0.11.

**Costs.**

- A breaking change for every program that names either type or calls
  `step`. The edits are mechanical: the two names, `step(())` to `step()`,
  `.done` to a call of `is_done()`, and `coevolution::harness` to
  `coevolution::generation_loop` for a path that named the module.
- Two statements where there was one expression. A site that stepped and
  tested in a single `if` now steps, then asks.
- Three public methods are added: `is_done` on each loop and on
  `PolicyNeuroevolution`. Each replaces what left, the `done` field on the
  two loops and the bare `bool` that `PolicyNeuroevolution::step` returned,
  which is the account ADR 0081 asks of an addition.
- `step` runs past the budget if it is called there. A caller who counts
  iterations and never asks `is_done` overruns without an error. The
  overrun is at least what was asked for, call by call, which a silent
  no-op would not be. Both methods' documentation says so, and unit tests
  pin it on both loops.
- The canonical value of the single-population loop is no longer returned.
  A caller who wants it has the natural value from `latest_metrics()` and
  knows the direction they declared on the fitness function: for a
  `Minimize` objective the old value is the new one negated. Outside the
  loops' own unit tests, nothing in the workspace read it.
- Eleven accepted ADRs name types that no longer exist under those names,
  ADR 0077 names `GenerationStep` and ADR 0001 quotes the old call. They
  are not edited. The Status section above is the mapping.
- The two loops are still one shape written twice. `CoEvolutionaryLoop`'s
  ten fields and seven public methods each have a counterpart in
  `EvolutionaryLoop`. Merging them needs a per-generation step trait on the
  single-population side, which is a design this ADR does not make.

**Unchanged.**

- What a run computes, for any budget above zero. The same generations run
  in the same order from the same seed, and `latest_metrics`, `best`,
  `generation` and `max_generations` return what they returned.
- `GenerationProbe`, its contract and both implementations' behaviour:
  `advance` still checks the budget before it steps, and `step` still does
  not.
- The `tracing` events and their field names, which the live dashboard
  reads.

**Surface delta**, from `tools/public-surface.sh` with all features on.
Four items leave `rlevo-evolution`: the struct `GenerationStep`, its two
public fields and its crate-root re-export. Two enter: `is_done` on each
loop. By kind, structs go from 139 to 138, public fields from 300 to 298,
re-exports from 162 to 161 and inherent methods from 294 to 296.
`rlevo-hybrid` gains one method, `PolicyNeuroevolution::is_done`. The two
renamed structs, the renamed module and their other methods change name and
keep their count. The listings' line totals rise all the same, from 1,215
to 1,222 and from 24 to 27, because the tool lists a method a second time
once documentation links to it, which adds eleven such lines here.

**How it was checked.** The workspace compiles with all targets and all
features, and each library compiles alone. The fast, heavy and GPU test
tiers pass, the determinism tests among them. The three unit tests that
read `reward` now read the value where it lives: the canonical fitness the
test strategy's `tell` received, and `CoEAMetrics::binding_fitness`. Four
unit tests are new, two on each loop, and pin what Decision 2 states:
`is_done` is `true` straight after `reset` when the budget is zero, and a
`step` past the budget runs a generation. The four outside-in probes pass;
none of their templates names a loop. Every Markdown example this change
wrote or corrected was compiled and run once in a throwaway test, since
nothing else compiles them.

## Alternatives considered

**Keep the deprecated aliases for one release.** Rejected in Decision 5.

**Have `step` return `bool`**, `true` once the budget is spent. A first
draft of this change took this shape, since it is `done` with nothing else
and it adds no public item. Rejected. `while !evo_loop.step() {}` does not
say what it tests, the flag's direction has to be learned, and the check
comes after the work. Turning the flag round, so that `true` means "more
to do", fixes the direction and not the order: `while evo_loop.step() {}`
still runs a generation before it asks, so a budget of zero still runs
one, and it inverts what `done` meant at every site.

**Have `step` return the generation's metrics**, `&StrategyMetrics` or
`&CoEAMetrics`. Rejected. 68 of the 78 call sites want to know whether to
stop and nothing else, and `latest_metrics()` already returns the metrics
to a caller who wants them.

**Have `step` refuse to run past the budget**, by doing nothing or by
panicking. Rejected. Doing nothing makes a call that looks like a
generation not be one, and hides a caller's mistake behind a run that
looks complete. Panicking turns a deliberate continuation into an error.
A caller who wants the guarded form has it: ask `is_done` first, or give
the loop to the evaluator, whose `advance` asks for them.

**Add a `run` method that resets and steps to the budget.** Not added. The
three-line loop is clear, and it would be one more public method on each
loop. `PolicyNeuroevolution` keeps the `run` it had.

**Delete the inherent `reset` and `step` and leave `GenerationProbe` as the
one way to drive a loop.** Rejected. Running a genetic algorithm on a
function would then need a trait imported from `rlevo_core::evaluation`,
and `advance` clones the metrics on every generation for a caller who may
not read them. The probe is the evaluator's seam; a loop is usable without
an evaluator.

**Return `std::ops::ControlFlow<()>`.** Rejected. It says the same thing as
a `bool` with nothing to carry, and every call site grows an
`.is_break()`.

**Rename the types and leave the word in tests and local variables.**
Rejected. It leaves `let mut harness = EvolutionaryLoop::new(..)` in 38
files, which is the confusion in miniature, and it makes Decision 6
uncheckable: with only the two types renamed, a search for "harness" in the
files that name a loop returned 635 lines to read by hand.

**Merge the two loops in the same change.** Not done; see Costs.

## References

- ADR [0076](0076-trial-seam-splits-machinery-from-trial-shape.md) and
  ADR [0077](0077-remove-benchenv.md), which removed the environment shape
  from the evaluator and left it on `step`.
- ADR [0081](0081-a-public-item-needs-an-implementor-and-a-consumer.md),
  the test a deprecated alias fails.
- ADR [0084](0084-the-umbrella-carries-the-evaluation-harness.md), which
  named the evaluator `rlevo::harness` and recorded the collision.
- ADR [0026](0026-shared-config-validation-convention.md) and ADR
  [0034](0034-fitness-hygiene-chokepoint-convention.md), the two
  chokepoints that live in `EvolutionaryLoop::new` and
  `EvolutionaryLoop::step`.
- `docs/rules.md` sections 2 and 3.
- Code: `crates/rlevo-evolution/src/strategy.rs` (`EvolutionaryLoop`);
  `crates/rlevo-evolution/src/coevolution/generation_loop.rs`
  (`CoEvolutionaryLoop`, `CoEAMetrics`);
  `crates/rlevo-evolution/src/probe.rs` (the two `GenerationProbe`
  implementations and a table of the two ways to drive a loop);
  `crates/rlevo-hybrid/src/policy_neuroevolution.rs`.
