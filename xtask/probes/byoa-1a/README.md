# byoa-1a

An outside-in probe: an algorithm author's walk. BYOA is "bring your own
algorithm". This author has written a learner, tabular Q-learning, and wants
to train it on a shipped environment through `rlevo`'s harness and show that
it does better than a random agent, with nothing but what `cargo add rlevo`
gives them.

The [`byoe-1-thermostat`](../byoe-1-thermostat/README.md) README covers how
a run works: the staging, the verdict lines, the run directory a red run
keeps, and the rules for a template. This file says what is different here.

```bash
cargo xtask probe byoa-1a
```

## Why a learning walk

`Agent` is the seam an algorithm author implements. A frozen policy needs
`act` alone, and the BYOE walks implement only that. A learner needs the
rest: the trial must show it every step, with the step's status, tell it
when an episode ends and which phase it is in, and train it before it is
measured. This walk implements a learner from outside, trains it through
the documented entry point, `Evaluator::run_suite`, and checks that the
trial passes it what the environment returned.

## The steps

| # | Step | Holds when | Fatal |
|---|------|------------|-------|
| 1 | `scaffold` | the probe crate exists outside the workspace | yes |
| 2 | `add-deps` | `rlevo` is the only dependency in the manifest. Every other one is a crate the author had to add by hand | no |
| 3 | `implement-algorithm` | the crate builds with no features: `src/q_learner.rs` implements `Agent` for any discrete action, through `rlevo::prelude` | yes |
| 4 | `learning-seam` | `src/learn.rs` builds (feature `learn`), and one `run_suite` call, with `EvaluatorConfig::train_steps` set, trains the learner and evaluates it: one trial, not errored, whose report carries `agent.qlearn_updates` equal to the training budget, one update per transition shown. No `Trial` is written and no loop | no |
| 5 | `first-party-env` | `src/frozen_lake.rs` builds (feature `env`): FrozenLake and `TimeLimit` reached through `rlevo::envs` | yes |
| 6 | `baseline` | `src/harness.rs` builds (feature `harness`), and three checks hold. The trial passes every status through: the order of what the environment returned and what the learner was told is checked event by event, and training holds at least five episodes of each ending. The learner's `return/mean` beats a uniform random agent's under the same configuration. A second run repeats every scalar but the two the wall clock sets, every counter and every evaluation episode, bit for bit | yes |

`xtask` checks steps 1, 2, 3 and 5, which a `cargo` invocation can
observe. The probe binary runs steps 4 and 6 and reports them.

Step 4 is a discoverability step: a learner that trains only through a
hand-written `Trial` works, but the documented path did not carry it. Step
6 is where a driver that loses information is caught, and it is fatal: a
learner told `Terminated` for a time-limit cut is biased, not broken, so it
would still beat the random agent.

The levels are not in step order. The environment (step 5, feature `env`)
is built before the training run (step 4, feature `learn`), which needs it.
A broken environment fails step 5, and step 4 reports that its code was not
built.

Today a run prints this, and exits zero:

```text
BYOA-STEP 1 PASS scaffold :: probe created at <run>/probe (outside the workspace)
BYOA-STEP 2 PASS add-deps :: rlevo alone is sufficient
BYOA-STEP 3 PASS implement-algorithm :: the learner implements Agent against public API only
BYOA-STEP 4 PASS learning-seam :: run_suite trained the learner through its Agent hooks, 20000 updates in 20000 steps, then evaluated it; no Trial written
BYOA-STEP 5 PASS first-party-env :: FrozenLake and TimeLimit reached through rlevo::envs
BYOA-STEP 6 PASS baseline :: every status reached the learner unchanged (20120 transitions; training episodes ended 3090 Terminated, 9 Truncated at the cap of 20); return/mean 1.000 against the random agent's 0.000; a second run repeats 49 values bit for bit, the wall clock aside
BYOA-1a passes.
```

The walk stops at the baseline. Recording the run, the learner's own
metrics in the record, and showing that two submissions faced one task are
not walked yet, and the note line says so on every run.

## The submission

`src/q_learner.rs` is tabular Q-learning, written from scratch; nothing in
`rlevo` implements it. It is generic over any `DiscreteAction<1>` and reads
a state's index through a function it is built with, so the file names
nothing outside the agent seam.

| Parameter | Value |
|---|---|
| table | 64 states × 4 actions: every state FrozenLake's observation can name |
| initial value | 1.0 in every cell. The goal pays 1.0 and every other tile 0, so an untried action looks best and the learner sweeps the map. A zero table ties everywhere, the greedy choice is index 0, and the learner presses into one wall |
| learning rate α | 0.5 |
| discount γ | 0.95 |
| exploration ε | 1.0 falling linearly to 0.05 over 1,500 training episodes; 0 once `set_training(false)` |
| update | the target is `r` when the transition's status is `Terminated`, and `r + γ · max Q(s′, ·)` otherwise, so it bootstraps through a `Truncated` step |

`act` draws from the RNG the trial passes. `emit_metrics` reports
`agent.qlearn_updates` and `agent.qlearn_mean_abs_td`.

## The task and the configuration

FrozenLake's 4×4 preset map, not slippery, from one config function, in
`TimeLimit` at 20 steps. One `EvaluatorConfig` serves every run: 20,000
training steps, 20 evaluation episodes of at most 100 steps, one trial,
`base_seed` 2026, one thread.

Both numbers were measured. With the training budget at 20,000 steps, the
cap gives these training endings:

| Cap | Terminated | Truncated |
|---|---|---|
| 10 | 3,020 | 184 |
| 15 | 3,061 | 44 |
| 20 | 3,090 | 9 |
| 30 | 3,091 | 0 |
| 100 | 3,091 | 0 |

20 is the largest cap that leaves at least five truncations. With the cap
at 20, the learned policy reaches the goal in every evaluation episode from
2,000 training steps on; the ε schedule ends near 10,400 steps, and 20,000
is about twice that.

## The two wrappers

Step 6 reads the order of events from two probe-side wrappers that log
into one list. `StatusLog`, outermost in the environment factory, is an
`Environment` by delegation and logs every reset and every status it
returns. `Watched` is an `Agent` by delegation and logs every status it is
shown, every end it is told of, and every `set_training`. A trial on one
thread makes the list a single sequence. The rules the check reads it by:

- each step the environment returns is shown to the learner next, with the
  same status;
- an episode the environment ended is ended for the learner, with that
  status, before anything else happens;
- an episode that ends while its last step is `Running` has reached the
  trial's `max_steps`, and ends `Truncated`;
- `set_training(true)` comes first, and `set_training(false)` after exactly
  the training budget's transitions.

## Files

| File | What it is |
|------|------------|
| `Cargo.toml.template` | the manifest; step 2 reads its `[dependencies]` |
| `src/q_learner.rs` | the submission, step 3 |
| `src/frozen_lake.rs` | the task, step 5, behind the `env` feature |
| `src/learn.rs` | the training run and step 4, behind the `learn` feature |
| `src/harness.rs` | the wrappers, the random agent and step 6, behind the `harness` feature |
| `src/main.rs` | the binary: the `PROBE-STEP` lines |

As for the thermostat, no workspace command compiles, lints or formats these
files. Format them with `rustfmt --edition 2024`.

The walk is `BYOA1A` in `xtask/src/commands/probe/walk.rs`.
