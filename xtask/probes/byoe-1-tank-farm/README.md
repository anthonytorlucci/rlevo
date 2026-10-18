# byoe-1-tank-farm

An outside-in probe: the BYOE-1 walk of
[`byoe-1-thermostat`](../byoe-1-thermostat/README.md), taken by a second
environment author. This author has two tanks in series and two pumps, and
wants to hold the second tank at half full through `rlevo`'s harness, with
nothing but what `cargo add rlevo` gives them.

The thermostat's README covers how a run works: the staging, the six steps,
the verdict lines, the run directory a red run keeps, and the rules for a
template. This file says only what is different here.

```bash
cargo xtask probe byoe-1-tank-farm
```

## Why a second environment

The thermostat has a fixed horizon, a discrete action and an observation of
the same rank as its state. A path the thermostat never takes can break
without either probe noticing. The tank farm takes these paths:

- **A rank-2 observation over a rank-1 state.** The observation is each
  tank's level over the last four steps, shape `[2, 4]`; the state is the two
  current levels, shape `[2]`.
- **A bounded continuous action that can be invalid.** `PumpCommand` holds
  two rates in `[0, 1]` and implements `ContinuousAction` and
  `BoundedAction`. An out-of-range or `NaN` rate is an error, not a clamp.
- **Two ways an episode ends.** An overflow terminates it. Otherwise the
  environment runs indefinitely, and the shipped `TimeLimit` wrapper
  truncates it, so the harness runs the farm through a wrapper, as a third
  party would.

## The steps

Steps 1, 2, 3 and 5 hold as they do for the thermostat, for
`src/tank_farm.rs` and `src/policy.rs`. Two steps check more:

| # | Step | Holds when |
|---|------|------------|
| 3 | `implement-environment` | as for the thermostat, and the base level's assertion that `TimeLimit<TankFarm>` implements `Environment<2, 1, 1>` compiles. An observation flattened to rank 1 fails here |
| 4 | `post-terminal` | the shapes are `[2, 4]`, `[2]` and `[2]`; the commands `[1.5, 0.0]` and `[NaN, 0.0]` each return `EnvironmentError::InvalidAction`, and the next valid step equals the same step on a fresh farm; with both pumps full on, the bare farm overflows and a further step returns `StepAfterEpisodeEnd` carrying `Terminated` from the farm's guard; with both pumps off, `TimeLimit` truncates at 60 steps and a further step returns it carrying `Truncated` from the wrapper's guard; each environment is usable again after a `reset` |
| 6 | `run-suite` | `Evaluator::run_suite` over `TimeLimit::new(TankFarm::with_seed(seed), 60)`, with the evaluator's cap at 64, returns 2 trials, none errored, each named `tank-farm`, each with 3 episodes of exactly 60 steps and a finite `return/mean`. A shorter episode means the policy overflowed a tank; a longer one, that the cap ended it and not the wrapper |

Today a run prints this, and exits zero:

```text
BYOE-STEP 1 PASS scaffold :: probe created at <run>/probe (outside the workspace)
BYOE-STEP 2 FAIL add-deps :: rlevo alone is insufficient; 1 additional crate(s) required by hand: rand
BYOE-STEP 3 PASS implement-environment :: Environment implemented against public API only; the file names nothing outside rlevo::core
BYOE-STEP 4 PASS post-terminal :: an invalid command returns InvalidAction and changes nothing; a step after the end returns StepAfterEpisodeEnd carrying Terminated after an overflow (step 8) and Truncated after the time limit (step 60); guard from rlevo::core
BYOE-STEP 5 PASS implement-agent :: the policy implements BenchableAgent with act alone
BYOE-STEP 6 PASS run-suite :: 2 trials, none errored, 3 episodes of 60 steps each, ended by TimeLimit; return/mean -1.467 and -2.893
BYOE-1/tank-farm passes; discoverability fails at step 2.
```

Step 2's finding is the thermostat's: `rand`, for `BenchableAgent::act`.

## Findings in the source

Two `FINDING:` comments in `src/tank_farm.rs` record friction that compiles:

- `ContinuousAction::as_slice` returns `&[f32]`, so the command stores its
  rates as an array. Two named fields cannot return a slice without
  `#[repr(C)]` and `unsafe`.
- `ContinuousAction::random` has a default that samples `[-1, 1)`, outside
  the pumps' range, and its documentation says such a type must override it.
  It takes no generator, so an override could only draw from an unseeded one.
  The probe leaves the default and never calls it.

## Files

| File | What it is |
|------|------------|
| `Cargo.toml.template` | the manifest; step 2 reads its `[dependencies]` |
| `src/tank_farm.rs` | the environment, step 3. Every `rlevo` path in it is `rlevo::core::...` in full |
| `src/main.rs` | the binary: the rank assertion, the contract checks of step 4, and the `PROBE-STEP` lines |
| `src/policy.rs` | the controller, step 5, behind the `agent` feature: pump A at a constant 0.3, pump B proportional to tank 2's distance from half full |
| `src/harness.rs` | the suite run and its checks, step 6, behind the `harness` feature |

As for the thermostat, no workspace command compiles, lints or formats these
files. Format them with `rustfmt --edition 2024`.
