# BYOA-1 — bring-your-own-algorithm acceptance test

```bash
cargo xtask byoa
```

The mirror of [BYOE-1](../byoe/README.md). BYOE-1 varies the **environment** and
holds the algorithm fixed; BYOA-1 varies the **algorithm** and uses first-party
environments.

## The persona

A submitter who has never seen this repository brings an existing or novel
method and wants it benched against other submissions and a random baseline,
over environments that ship with `rlevo` — including the transpiled Gymnasium
zoo. This is the harness-as-a-product claim, and no other test touches it.

The submitted algorithm is tabular Q-learning on `FrozenLake`. It is
deliberately a **learning** method rather than a fixed policy: "submit your
algorithm" and "submit your trained policy" are different asks, and only the
first is what the product claim promises.

## The nine steps

| # | Label | Asks |
|---|---|---|
| 1 | `scaffold` | a crate exists outside the workspace |
| 2 | `add-deps` | is `rlevo` alone enough? *(discoverability — non-fatal)* |
| 3 | `implement-algorithm` | does a from-scratch learner compile against the packaged public API? |
| 4 | `learning-seam` | can the **documented** entry point express a learning algorithm? *(discoverability — non-fatal)* |
| 5 | `first-party-env` | can it train against a shipped environment reached through the umbrella? |
| 6 | `baseline` | does it beat a random baseline under an identical `EvaluatorConfig`? |
| 7 | `record` | does the run record, under the environment's **own** family? |
| 8 | `custom-metrics` | do the submitter's own metric names reach the report? |
| 9 | `same-task-proof` | can two submissions be shown to have faced the same task? |

Steps 2 and 4 answer *"was this findable?"*; the rest answer *"was this
possible?"*. Conflating the two loses information, so those two are reported,
folded into the verdict, and do not stop the walk. BYOE-1 makes the same split
at its own step 2.

The exit code is the number of the first **fatal** failing step, or 0 if all
pass. `BYOA_KEEP=1` preserves the staging directory, which is the fast way to
iterate: the staged crates are already built, so `cargo run` in
`$WORK/byoa-probe` re-runs the probe without repackaging.

## Why it consumes `cargo package` tarballs

Same reason as BYOE-1, and the logic is literally shared — see
[`xtask/common/stage.sh`](../common/stage.sh). Only files that `cargo package`
would actually ship are present, so a missing `include`, an un-`pub`'d item, or
an uncommitted file surfaces here rather than after a publish.

`PUBLISHABLE` is identical to BYOE-1's, deliberately: a submitter who never
writes an environment still receives the whole umbrella cone, because `rlevo` is
one crate. If the two lists ever diverge, the umbrella has grown a
consumer-visible seam that one persona sees and the other does not.

## Current result

**Fails at step 8 of 9.** Findings, in the order the walk produces them:

- **Step 2** — four crates by hand (`parking_lot`, `rand`, `tracing`,
  `tracing-subscriber`), each an unversioned guess. `parking_lot` and
  `tracing-subscriber` are *forced by public trait signatures*, not by
  convenience.
- **Step 4** — `BenchableAgent` is `act` + `emit_metrics`, and
  `EpisodicTrial::run` never returns reward, next observation, or `done` to the
  agent. It is an **evaluation** seam for a fixed policy. Training required
  implementing `Trial` and calling `run_trials` directly; nothing on the
  documented `run_suite` path names either.
- **Step 7 passes**, and the contrast with BYOE-1 is the point: a first-party
  environment names its own family (`ToyText`) instead of masquerading.
- **Step 8** — the submitter's metrics never arrive.
  `RecordingLayer`'s `CaptureVisitor::record_canonical` drops every `tracing`
  field whose name is not in `CANONICAL_METRICS`, so `qlearn_mean_abs_td` and
  `qlearn_updates` are silently discarded while `episode_return`,
  `episode_length`, and `episode_wall_clock_secs` arrive.
- **Step 9** *(not reached; structural)* — `RunManifest` has no
  environment-configuration fingerprint. `env_family` is the most specific
  task-identity field it carries, and it is one of six values.

## A note on writing probes

The first draft of this probe seeded its training RNG with `rand::rng()` and was
intermittently non-reproducible — step 6 passed or failed depending on whether
exploration happened to find the goal. `EpisodicTrial` seeds from the
harness-supplied trial seed for you; a hand-written `Trial` must remember to,
and nothing enforces it. A flaky acceptance test is worse than no acceptance
test, because its failures get attributed to the thing under test.
