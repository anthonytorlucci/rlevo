# byoa-1b

An outside-in probe: an optimiser author's walk. BYOA is "bring your own
algorithm". This author has written an optimiser, a `(1+1)` hill-climber,
and wants to run it against a shipped landscape through `rlevo`'s harness
and show that it does better than a strategy that does nothing, with
nothing but what `cargo add rlevo` gives them.

The [`byoe-1-thermostat`](../byoe-1-thermostat/README.md) README covers how
a run works: the staging, the verdict lines, the run directory a red run
keeps, and the rules for a template. This file says what is different here.

```bash
cargo xtask probe byoa-1b
```

## Why an optimiser walk

`Strategy` is the seam an optimiser author implements, and every strategy
in the workspace is written inside it, where every item is in reach. This
walk implements one from outside: its parameters, its state, `init`, `ask`,
`tell` and `best`, against the packaged crates. It then checks the two
things a comparison between submissions rests on: the reported best
fitness tracks the search, and a run repeats from its seed.

## The steps

| # | Step | Holds when | Fatal |
|---|------|------------|-------|
| 1 | `scaffold` | the probe crate exists outside the workspace | yes |
| 2 | `add-deps` | `rlevo` is the only dependency in the manifest. Every other one is a crate the author had to add by hand | no |
| 3 | `implement-strategy` | the crate builds with no features: `src/hill_climber.rs` implements `Strategy`, with `Device` and `Tensor` reached through `rlevo::burn` | yes |
| 4 | `null-strategy` | `src/null.rs` builds (feature `null`): a strategy whose `ask` returns the initial population unchanged | yes |
| 5 | `run-trials` | `src/harness.rs` builds (feature `harness`). Each strategy runs on the shipped `Sphere` in four dimensions, through `EvolutionaryLoop` and `Evaluator::run_trials` with a `GenerationTrial` per trial, for 1 and for 60 generations from one `base_seed`. Each run returns 2 trials, none errored, with no episodes, `generations` equal to the budget and a finite `ea/best_fitness_ever` | yes |
| 6 | `ordering` | per trial, the null strategy's `ea/best_fitness_ever` after 60 generations equals its value after one; the hill-climber's value after one generation equals the null strategy's, since the two share `init`; and after 60 generations the hill-climber's value is strictly lower | yes |
| 7 | `determinism` | both 60-generation runs again, from the same `base_seed`: every scalar but `wall_clock_seconds` is bit-equal, trial by trial | yes |

`xtask` checks steps 1 to 4, which a `cargo` invocation can observe. The
probe binary runs steps 5, 6 and 7 and reports them.

Step 6 is what keeps the comparison honest. If the reported value were the
initial population's best, or anything else that does not follow the
search, the null strategy would score as well as the hill-climber, and the
equality on the null run catches it. A generation trial reports its last
generation's metrics only, so the one-generation runs supply each trial's
starting value. Trial seeds derive from `base_seed`, so trial `i` starts
from the same point in every run.

Step 7 runs the trials on two threads, and matches them by trial index,
since they finish in either order.

Today a run prints this, and exits zero:

```text
BYOA-STEP 1 PASS scaffold :: probe created at <run>/probe (outside the workspace)
BYOA-STEP 2 FAIL add-deps :: rlevo alone is insufficient; 1 additional crate(s) required by hand: rand
BYOA-STEP 3 PASS implement-strategy :: the hill-climber implements Strategy against public API only
BYOA-STEP 4 PASS null-strategy :: the null strategy implements Strategy
BYOA-STEP 5 PASS run-trials :: both strategies, 2 trials each, none errored, no episodes, 1 and 60 generations; ea/best_fitness_ever present and finite
BYOA-STEP 6 PASS ordering :: per trial the null strategy ends where it starts; the hill-climber falls from 26.3606 to 0.3208 and 20.7505 to 0.1546
BYOA-STEP 7 PASS determinism :: a second run of each repeats all 32 scalars bit for bit, the wall clock aside
BYOA-1b passes; discoverability fails at step 2.
```

Step 2's finding is written down as the `FINDING:` line in
`Cargo.toml.template`: `rand`, because `Strategy::init`, `ask` and `tell`
take `&mut dyn rand::Rng`, and `rlevo` does not re-export `rand`.

The walk stops at the determinism check. Recording the run, and showing
that two submissions faced one task, are not walked yet, and the note line
says so on every run.

## The hill-climber

`src/hill_climber.rs` keeps one parent. The first generation scores the
parent itself. Every later generation proposes the parent plus a uniform
step of at most 0.5 in each coordinate, clamped to the landscape's bounds,
and keeps the proposal if it scores no worse. Every draw comes from the RNG
the loop passes in. `tell` sanitises the fitness it reads to host with
`rlevo::evo::fitness::sanitize_fitness`, as the trait's documentation asks
of a strategy written outside the crate, and builds its metrics with
`StrategyMetrics::from_host_fitness`.

`src/null.rs` takes the hill-climber's parameters, state and `init`, and
proposes its starting point every generation.

## Findings in the source

One `FINDING:` comment, in `src/harness.rs`, records friction that
compiles. `rlevo::harness::metrics::ea` names a constant for
`ea/best_fitness` and none for `ea/best_fitness_ever`, the key the loop
reports. The probe spells the key as `docs/ask-tell-contract.md` does.

## Files

| File | What it is |
|------|------------|
| `Cargo.toml.template` | the manifest; step 2 reads its `[dependencies]` |
| `src/hill_climber.rs` | the submission, step 3 |
| `src/null.rs` | the control, step 4, behind the `null` feature |
| `src/harness.rs` | the suite runs and steps 5 to 7, behind the `harness` feature |
| `src/main.rs` | the binary: the `PROBE-STEP` lines |

As for the thermostat, no workspace command compiles, lints or formats these
files. Format them with `rustfmt --edition 2024`.

The walk is `BYOA1B` in `xtask/src/commands/probe/walk.rs`.
