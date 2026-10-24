# byoe-1-thermostat

An outside-in probe: a small crate that plays an environment author from
outside this project. BYOE is "bring your own environment". The author here
has a room, a boiler and a day of weather, and wants to run a controller on
it through `rlevo`'s benchmark harness with nothing but what `cargo add rlevo`
gives them.

This directory is the probe's template. A run copies `Cargo.toml.template`
and `src/` into a scratch directory outside the checkout and builds them
there. This README is not copied.

## Run it

```bash
cargo xtask probe byoe-1-thermostat
```

Run it from anywhere in the checkout. It needs the pinned toolchain, `tar` on
the path, and network access: third-party crates resolve from crates.io as
they would for any consumer. It tests the working tree as it stands,
uncommitted changes included.

Each run packages the workspace and builds the probe from nothing, so it
takes about a minute, not seconds (50 s on an Apple M2 Pro with 12 cores, one
run). Nothing is reused from the workspace's `target/` or from an earlier
run, so a stale artifact cannot make it pass.

| Command | What it does |
|---------|--------------|
| `cargo xtask probe byoe-1-thermostat` | runs this probe |
| `cargo xtask probe --list` | prints every probe's name as a JSON array and runs nothing |
| `RLEVO_PROBE_SCRATCH=<dir> cargo xtask probe byoe-1-thermostat` | puts the run directory under `<dir>`, which must be outside the checkout. The default is the system's temporary directory |

The step lines and the verdict go to stderr. In CI, the
`probe (byoe-1-thermostat)` job of `.github/workflows/crate-tests.yml` runs
the same command on every pull request and writes the lines to the job
summary.

## Why it exists

Every test, bench and example in the workspace resolves its sibling crates by
path, sees `publish = false` crates through dev-dependencies and has its
`required-features` turned on for it. None of that is true for someone who
runs `cargo add rlevo`. A change can pass every workspace test and still
leave that person unable to do the thing the library is for.

This probe is that person, written down as a crate and run on every pull
request. It builds against the crates as `cargo package` would ship them, so
it fails on what an outside author would hit:

- a file that did not make it into a package,
- an item they need that is not public, or not reachable through `rlevo`,
- a crate they must find and add by hand,
- an environment contract that needs more than the contract crate,
- a documented path that no longer compiles.

It is also the only build of the packaged crates against each other.
`cargo package --no-verify` checks that each crate packages, never that the
packages compile together.

The environment is invented on purpose. A copy of a shipped environment
inherits its payload source, its renderer and its family, and each of those
is something a third party does not have.

The template doubles as a measurement. A change to the API it uses changes
the template in the same pull request, so the template's diff shows what an
outside author stops or starts writing.

## What it should do

A run does four things:

1. Packages every publishable crate in the workspace and extracts the
   packages into a fresh run directory outside the checkout.
2. Writes this template beside them as a crate of its own, with a
   `[patch.crates-io]` table that points every `rlevo` crate at its extracted
   copy.
3. Checks that every `rlevo` crate in the probe's resolved graph came from
   those copies. A patch that does not apply is only a warning to cargo, and
   the probe would then test the last release in place of this checkout.
4. Walks six steps and prints one line for each, then a verdict.

| # | Step | Holds when | Fatal |
|---|------|------------|-------|
| 1 | `scaffold` | the probe crate exists outside the workspace | yes |
| 2 | `add-deps` | `rlevo` is the only dependency in the manifest. Every other one is a crate the author had to add by hand | no |
| 3 | `implement-environment` | `src/thermostat.rs` names nothing outside `rlevo::core`, and the crate builds with no features | yes |
| 4 | `post-terminal` | the episode ends `Terminated` after 24 steps; two more steps each return `EnvironmentError::StepAfterEpisodeEnd` carrying `Terminated` and leave the state unchanged; a second `reset` makes the environment usable again | yes |
| 5 | `implement-agent` | `src/policy.rs` builds: a policy that implements `Agent` with `act` alone | yes |
| 6 | `run-suite` | `Evaluator::run_suite` returns 2 trials, none errored, each named `thermostat`, each with 3 episodes of exactly 24 steps and a finite `return/mean` | yes |

`xtask` checks steps 1, 2, 3 and 5, which a `cargo` invocation can observe.
The probe binary runs steps 4 and 6 and reports them. A build failure is
reported at the step whose code broke, with the compiler's first error. A
binary that panics or runs past 300 seconds fails the steps it did not
report.

Today a run prints this, with the run directory in place of `<run>`, and
exits zero:

```text
BYOE-STEP 1 PASS scaffold :: probe created at <run>/probe (outside the workspace)
BYOE-STEP 2 PASS add-deps :: rlevo alone is sufficient
BYOE-STEP 3 PASS implement-environment :: Environment implemented against public API only; the file names nothing outside rlevo::core
BYOE-STEP 4 PASS post-terminal :: a step after the end returns StepAfterEpisodeEnd carrying Terminated and changes nothing; guard from rlevo::core
BYOE-STEP 5 PASS implement-agent :: the policy implements Agent with act alone
BYOE-STEP 6 PASS run-suite :: 2 trials, none errored, 3 episodes of 24 steps each; return/mean -213.564 and -45.564
BYOE-1/thermostat passes.
```

Between step 6 and the verdict it also prints the `rlevo` crates the probe
resolved, the toolchain, and a note on what the walk does not cover.

Step 2 is the one step that is not fatal. A `FAIL` there is a finding on a
path that works: a crate the author had to add by hand, written down as a
`FINDING:` line in `Cargo.toml.template`. It leaves the manifest in the pull
request that makes it unnecessary. The harness left when `rlevo` began to
re-export it as `rlevo::harness`, and `rand`, for the `Rng` the policy's
`act` takes, when `rlevo-core` began to re-export it and the prelude to
carry `Rng`.

The last line is one of four:

| Last line | Exit | Meaning |
|-----------|------|---------|
| `BYOE-1/thermostat passes.` | zero | all six steps held |
| `BYOE-1/thermostat passes; discoverability fails at step 2.` | zero | the path works, and a step that is not fatal recorded a finding |
| `BYOE-1/thermostat fails at step N of 6.` | non-zero | a fatal step failed. The lines stop there |
| `BYOE-1/thermostat aborted: <why>` | non-zero | the run never reached the walk: the workspace did not package, the probe's dependencies did not resolve, or an `rlevo` crate came from somewhere other than this checkout |

The walk stops at the harness run. Recording a run and rendering a report
are not walked yet, and the note line says so on every run.

## When it is red

A green run deletes its run directory. A red run keeps it, without the build
output, and prints `run kept at <run>`:

- `<run>/stage/crates/<crate>-<version>/` holds each crate as packaged. Look
  here for a file that is missing from a package.
- `<run>/probe/` holds the probe as it was built: `Cargo.toml`, `Cargo.lock`,
  `src/`, the patch table in `.cargo/config.toml`, and what the binary
  printed in `probe-stderr.log` and `probe-stdout.log`.

The step line carries the compiler's first error or the binary's own reason.
For the rest, rebuild in place:

```bash
cd <run>/probe
cargo +<channel> run --features harness
```

`<channel>` is the one in `rust-toolchain.toml`. Name it: the pin does not
apply outside the checkout, so a bare `cargo` there uses your default
toolchain.

## Changing it

| File | What it is |
|------|------------|
| `Cargo.toml.template` | the manifest, with `{{rlevo_version}}` filled in at run time. Step 2 reads its `[dependencies]`. It is not named `Cargo.toml`, so cargo never adopts this directory into the workspace |
| `src/thermostat.rs` | the environment, step 3: 24 hourly steps, a boiler that is on or off, heat lost to the outside in proportion to the temperature difference |
| `src/main.rs` | the binary: the post-terminal check of step 4, and the `PROBE-STEP` lines `xtask` reads |
| `src/policy.rs` | the bang-bang controller, step 5, behind the `agent` feature |
| `src/harness.rs` | the suite run and its checks, step 6, behind the `harness` feature |

No workspace command compiles, lints or formats these files. `cargo fmt
--all` does not reach them, so format them with `rustfmt` directly. A probe
run is the only thing that compiles them.

`src/thermostat.rs` has one rule of its own. Every `rlevo` path in it is
written in full as `rlevo::core::...`, with no grouped import, because the
`consumer-core` probe builds a copy of the file against `rlevo-core` alone.
The check is textual, so a path in a comment counts.

The rules for a template are in the module documentation of
`xtask/src/commands/probe/mod.rs`, under "Writing a template". The ones that
matter most here:

- Use public API only, reached the way the documentation says to reach it.
- Keep a workaround and mark it `FINDING:` with what it shows. Smoothing it
  away deletes the evidence.
- Bound every loop over environment steps, so a defect fails the probe and
  does not hang it.
- Seed everything from a constant.
- Keep each step's code behind its level's feature.

The walk is `BYOE1` in `xtask/src/commands/probe/walk.rs`: a step is a row of
its `steps`, and a build level a row of its `levels`. The binary reports a
step by its label, so step numbers live in that one file.
