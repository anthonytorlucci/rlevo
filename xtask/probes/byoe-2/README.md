# byoe-2

An outside-in probe: a landscape author's walk. BYOE is "bring your own
environment", and for this author the environment is a cost surface: where
to dig one well for five houses. They want a shipped optimiser to search it
through `rlevo`'s harness, with nothing but what `cargo add rlevo` gives
them.

The [`byoe-1-thermostat`](../byoe-1-thermostat/README.md) README covers how
a run works: the staging, the verdict lines, the run directory a red run
keeps, and the rules for a template. This file says what is different here.

```bash
cargo xtask probe byoe-2
```

## Why a landscape walk

The BYOE-1 probes walk the `Environment` contract and the episodic harness.
A landscape author touches neither. They implement `Landscape`, hand it to
the evolutionary engine through `FromLandscape`, and run a `Strategy` in an
`EvolutionaryLoop`, which the harness drives as a `GenerationTrial` (ADR
0076). A change that broke that path, in the umbrella's re-exports, the
adapter, the loop or the generation trial, would leave every other probe
green.

## The steps

| # | Step | Holds when | Fatal |
|---|------|------------|-------|
| 1 | `scaffold` | the probe crate exists outside the workspace | yes |
| 2 | `add-deps` | `rlevo` is the only dependency in the manifest. Every other one is a crate the author had to add by hand | no |
| 3 | `implement-landscape` | `src/well_siting.rs` names nothing outside `rlevo::core`, and the crate builds with no features | yes |
| 4 | `wrap-landscape` | `src/engine.rs` builds (feature `engine`). `FromLandscape` reports `Minimize`, the default for a landscape that names no direction, and the check reads `well_siting.rs` as compiled to confirm it names none. Three sites score the landscape's own cost through the adapter | yes |
| 5 | `build-loop` | `src/strategy.rs` builds (feature `strategy`): the shipped `(4+12)` evolution strategy in an `EvolutionaryLoop` over the wrapped landscape, with the device named through `rlevo::burn`. `EvolutionaryLoop::new` accepts its parameters | yes |
| 6 | `run-trials` | `src/harness.rs` builds (feature `harness`). `Evaluator::run_trials`, with a `GenerationTrial` per trial, runs twice from one `base_seed`, with budgets of 1 and 30 generations. Each run returns 2 trials, none errored, with no episodes and `generations` equal to the budget. Per trial, `ea/best_fitness_ever` is finite, strictly lower after 30 generations than after one, and not below the surface's minimum | yes |

`xtask` checks steps 1, 2 and 3, which a `cargo` invocation can observe.
The probe binary runs steps 4, 5 and 6 and reports them.

A generation trial reports its last generation's metrics only, so one run
cannot show a fall. The one-generation run supplies each trial's starting
value. Trial seeds derive from `base_seed`, so trial `i` starts from the
same population in both runs.

Today a run prints this, and exits zero:

```text
BYOE-STEP 1 PASS scaffold :: probe created at <run>/probe (outside the workspace)
BYOE-STEP 2 PASS add-deps :: rlevo alone is sufficient
BYOE-STEP 3 PASS implement-landscape :: Landscape implemented against public API only; the file names nothing outside rlevo::core
BYOE-STEP 4 PASS wrap-landscape :: FromLandscape scores 3 sites at the landscape's own cost and minimises; well_siting.rs names no direction
BYOE-STEP 5 PASS build-loop :: EvolutionaryLoop::new accepts a (4+12)-ES on the landscape; the device is named through rlevo::burn
BYOE-STEP 6 PASS run-trials :: 2 trials, none errored, no episodes, 30 generations each; ea/best_fitness_ever fell from 17.8378 to 9.9400 and 25.9830 to 9.9400 (minimum 9.9400)
BYOE-2 passes.
```

The walk stops at the harness run. Recording the population and drawing
the landscape are not walked yet, and the note line says so on every run.

## The landscape

`src/well_siting.rs` places five houses at fixed points. The cost of a well
at `(x, y)` is the mean squared distance from it to the houses. The surface
is two-dimensional and smooth, and its minimum is at the houses' centroid,
where the cost is 9.94. A run that reported a lower value would be
reporting something other than the cost. The file implements `evaluate`
alone, and leaves the direction to the trait's default.

## Findings in the source

One `FINDING:` comment, in `src/harness.rs`, records friction that
compiles. `rlevo::harness::metrics::ea` names a constant for
`ea/best_fitness` and none for `ea/best_fitness_ever`, the key the loop
reports. The probe spells the key as `docs/ask-tell-contract.md` does.

## Files

| File | What it is |
|------|------------|
| `Cargo.toml.template` | the manifest; step 2 reads its `[dependencies]` |
| `src/well_siting.rs` | the landscape, step 3. Every `rlevo` path in it is `rlevo::core::...` in full |
| `src/engine.rs` | the adapter and its check, step 4, behind the `engine` feature |
| `src/strategy.rs` | the loop, step 5, behind the `strategy` feature |
| `src/harness.rs` | the two suite runs and their checks, step 6, behind the `harness` feature |
| `src/main.rs` | the binary: the `PROBE-STEP` lines |

As for the thermostat, no workspace command compiles, lints or formats these
files. Format them with `rustfmt --edition 2024`.

The walk is `BYOE2` in `xtask/src/commands/probe/walk.rs`.
