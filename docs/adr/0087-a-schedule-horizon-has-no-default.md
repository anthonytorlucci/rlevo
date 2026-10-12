---
project: rlevo
status: active
type: decision
date: 2026-10-12
tags: [adr, decision, evolution, metaheuristic, configuration, defaults, breaking-change]
---

# ADR 0087: A schedule horizon has no default

## Status

**Accepted (2026-10-12).**

Supersedes nothing. No earlier ADR mentions the field this one is about.
It applies the reasoning of ADR
[0076](0076-trial-seam-splits-machinery-from-trial-shape.md), which
objected to "a second, unenforced copy of the same number", to a place
where the second copy cannot be removed, only made visible.

## Context

Three strategies in `rlevo_evolution::algorithms::metaheuristic` pace a
coefficient over a number of generations, as the papers that define them
do:

| Strategy | Coefficient | Field |
|---|---|---|
| Grey Wolf Optimizer | $a = 2 (1 - t/T)$, held at 0 after $T$ | `GwoConfig::max_generations` |
| Whale Optimization Algorithm | the same $a$ | `WoaConfig::max_generations` |
| Salp Swarm Algorithm | $c_1 = 2 \exp(-(4t/T)^2)$, held at its final value after $T$ | `SalpConfig::max_generations` |

$t$ is the strategy's own generation count and $T$ is the field. No other
strategy in the crate has a horizon.

A `Strategy` is stateless and is handed its configuration, its state and a
random stream. It is not handed the loop's generation budget, which is the
last argument of `EvolutionaryLoop::new`. So $T$ and the budget are two
numbers, and the configuration's own documentation said so: the value
"need not equal the loop budget".

Each configuration was built by `default_for(pop_size, genome_dim)`, which
set `max_generations` to 500.

**Nothing set it to anything else.** A search of the workspace for a write
to the field finds the three definitions, the three `500`s, and two tests
that set it to 0 to check validation. Every use ran on the default, and
none ran 500 generations:

| Where | Loop budget | Horizon |
|---|---|---|
| `tests/determinism.rs` | 30 | 500 |
| `rlevo/tests/swarm_rastrigin_suite.rs` | 120 | 500 |
| the three `*_converges_on_sphere_d10` unit tests | 600 | 500 |

**What a wrong horizon costs.** Mean best cost over twelve seeds,
population 32, dimension 10, on a Sphere whose optimum is moved to 1.7 in
every dimension. *(Measured on an Apple M2 Pro, 12 cores, 32 GiB, macOS,
`rustc` 1.99.0, `burn` 0.21.0, the `Flex` backend, release profile.)*

| Strategy | Loop budget | Horizon 500 | Horizon equal to the budget |
|---|---|---|---|
| GWO | 120 | 5.5e-1 | 3.0e-4 |
| SSA | 120 | 4.5 | 5.5e-7 |
| WOA | 120 | 3.1e-3 | 2.8e-2 |
| GWO | 2000 | 1.5e-5 | 1.1e-6 |
| WOA | 2000 | 9.8e-4 | 1.2e-4 |

- With a budget shorter than the horizon, GWO and SSA are still exploring
  when the run ends: at 120 of 500 generations $a$ has fallen from 2 to
  1.52 and $c_1$ from 2 to 0.80. The run completes, reports a result and
  has not converged.
- With a budget longer than the horizon, GWO stops. Its results for
  budgets of 600 and of 2000 at a horizon of 500 are the same in every
  statistic. At $a = 0$ every wolf moves to the mean of the three leaders,
  the pack is one point, and the remaining generations evaluate it again
  and again. WOA is nearly as still: the same median at 600 and at 2000.
- Equal is not always best. WOA at a budget of 120 does better on the
  slow schedule. The horizon is the published schedule when it equals the
  budget, and a tuning parameter otherwise.

A default has to be one number, and no one number is right for two loops
of different lengths. A wrong one fails without a sign.

## Decision

1. **The three constructors take the horizon.**

   ```rust
   GwoConfig::default_for(pop_size, genome_dim, max_generations)
   WoaConfig::default_for(pop_size, genome_dim, max_generations)
   SalpConfig::default_for(pop_size, genome_dim, max_generations)
   ```

   There is no 500 and no other value filled in. The caller passes the
   generation budget they give the loop, or another value on purpose.

2. **The field keeps its name, its type and its validation.**
   `pub max_generations: usize`, at least 1. A caller can still set it
   after construction.

3. **The number is written twice, and both are in sight.** Once to the
   configuration and once to the loop, normally from one constant:

   ```rust
   const GENERATIONS: usize = 300;
   let params = GwoConfig::default_for(32, 10, GENERATIONS);
   let evo_loop = EvolutionaryLoop::new(strategy, params, fitness, seed, device, GENERATIONS)?;
   ```

   The loop does not check that they agree. They may differ on purpose,
   and the loop is generic over the strategy and does not know which
   configurations have a horizon.

4. **The documentation says what each mismatch does.** Each field's
   documentation, and a section of each module's, states that the usual
   value is the loop's budget, what a shorter loop gets and what a longer
   one gets, in that strategy's own terms.

5. **The rule, for the next configuration.** `docs/rules.md` section 4: a
   value whose right setting is a property of the run it is used in has no
   default and is an argument of the constructor. A default is for a value
   that is sound whatever the run: a rate, a domain, a shape constant.

6. **The workspace's own uses are corrected.** The determinism test passes
   its generation count, the swarm suite passes `MAX_GENS`, and the three
   convergence tests pass 600.

## Consequences

**Positive.**

- A run of GWO, WOA or SSA through the constructor is the algorithm the
  paper describes, for whatever length the caller names. Before, it was
  that only at 500 generations.
- The second number cannot be forgotten. It can still be set wrongly, but
  not by saying nothing.
- A new test, `crates/rlevo-evolution/tests/swarm_horizon.rs`, holds the
  behaviour: on a shifted Sphere at a budget of 120, GWO and SSA converge
  with the horizon at the budget and do not with it at 500, on each of
  three seeds.

**Costs.**

- A breaking change for every caller of the three constructors. The edit
  is one argument. In the workspace they are called 35 times in five
  files, all tests.
- One number, two places. The caller keeps them in step. Decision 3 says
  why the loop does not.
- Two results the tests recorded change, and one of them looks worse. In
  the swarm suite, 120 generations on a 10-dimensional Rastrigin with its
  optimum at the origin, SSA goes from an average of 42.1 to 11.9, and GWO
  from exactly 0.0 to 2.2. The 0.0 was not search. GWO is drawn to the
  origin, on a long schedule it is still being drawn there when the run
  ends, and this Rastrigin has its optimum at the origin. On the Rastrigin
  moved to 1.7, GWO at 120 generations averages 36.5 on the old default
  and 15.2 with the horizon at the budget.
- The three unit tests named `*_converges_on_sphere_d10` still use a
  Sphere with its optimum at the origin, where GWO and WOA return exactly
  0.0 at almost any setting. They show that the strategies run, not that
  they search. The new test uses a shifted objective for that reason.

**Unchanged.**

- The three algorithms: the update rules, the schedules and their clamps.
- Every other strategy and every wrapper. The `Strategy` trait is not
  touched.
- A run that set the field by hand before gets what it got.

**Surface delta**, from `tools/public-surface.sh rlevo-evolution` with all
features on. No item enters or leaves: three functions change signature.
The listing grows by six lines, from 1,225 to 1,231, because the three
constructors and the three fields are now linked from documentation and
the tool lists a linked item a second time, by path.

**How it was checked.** The workspace compiles with all targets and all
features. A constructor made to ignore its argument and store 500 fails
the new test, the constructor's doctest and the validation test that
passes 0. The fast, heavy and GPU test tiers pass.

## Alternatives considered

**Have the loop supply the horizon.** One number, written once: the field
becomes optional and `EvolutionaryLoop::new` hands its budget to the
strategy through a new method on `Strategy` with a default that does
nothing. Rejected. It is a method on a trait every strategy implements,
for three strategies that the module's own documentation calls legacy
comparators. Every strategy that wraps another, `MemeticWrapper` and
`WeightOnly` today, would have to forward it, and one that did not would
drop the horizon without a sign, which is the failure this ADR removes. A
hand-written ask/tell loop would have to call it too. If more strategies
with a horizon arrive, this is the alternative to reopen.

**Check in the loop that the two agree.** Rejected for the same reason:
the loop would need the same hook to ask a configuration for its horizon,
and a horizon that differs from the budget can be a deliberate setting.

**Keep the default and document it.** Rejected. The library's own five
uses were wrong under a documented field, which shows what documentation
does for a value nobody has to type.

**Make the field `Option<usize>` and fail validation when it is `None`.**
Rejected. It moves the missing value from a compile error to a run-time
one.

**Rename the field,** to `horizon` for instance, since it shares a name
with the loop's `max_generations` and is a different number. Not done.
Unlike the two seeds of ADR
[0085](0085-the-generation-loop-is-named-evolutionary-loop.md), these two
are meant to be equal in the ordinary case, and the shared name says so.
It would also be a second breaking change to three public fields.

**Replace the schedules with ones that need no horizon.** Rejected. The
linear decay of $a$ over the run and the decay of $c_1$ are the
algorithms. A strategy without them is a different one under the same
name.

**Delete the three strategies.** Not decided here. The module documents
them as legacy comparators, shipped because they are widely cited, and the
measurements above (an exact 0.0 at the origin, and nothing below 9 on
average on a shifted Rastrigin) are consistent with the papers it cites.
Whether the library should carry them is a question about the catalogue,
and removing them would also remove the only horizon in the crate.

## References

- ADR [0026](0026-shared-config-validation-convention.md), the
  configuration convention the three configs follow.
- ADR [0076](0076-trial-seam-splits-machinery-from-trial-shape.md), on a
  second copy of a budget.
- ADR [0085](0085-the-generation-loop-is-named-evolutionary-loop.md), the
  loop and its budget.
- `docs/rules.md` section 4.
- Code: `crates/rlevo-evolution/src/algorithms/metaheuristic/gwo.rs`,
  `woa.rs` and `salp.rs`; `crates/rlevo-evolution/tests/swarm_horizon.rs`.
