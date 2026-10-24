# learner-1

An outside-in probe: a learner's walk. The learner has read the README,
runs `cargo new`, pastes the Quick Start's dependency line into the
manifest and its program into `src/main.rs`, and runs it. Nothing compiles
a README, so a Quick Start that has stopped compiling, or that names a
release other than the one the documentation describes, is otherwise found
by that learner.

The [`byoe-1-thermostat`](../byoe-1-thermostat/README.md) README covers how
a run works: the staging, the verdict lines, the run directory a red run
keeps. This file says what is different here.

```bash
cargo xtask probe learner-1
```

## What it copies

This directory holds no program. When the probe is written, `xtask` reads
the `## Quick Start` section of the workspace's `README.md` and copies its
first `toml` block to the end of `Cargo.toml.template` and its first `rust`
block to `src/main.rs`, as written. Neither is edited, so a change to the
README is a change to the probe.

Before anything is packaged, `xtask` also reads every `rlevo` dependency
line in `README.md` and in the crate docs of `crates/rlevo/src/lib.rs`, the
two places a learner copies one from: `rlevo = "0.5"`, or
`rlevo = { version = "0.5", .. }`. Each must name the release this checkout
builds, by its leading components (`0.5` names 0.5.0). A line that does not
aborts the run, and the verdict names the file and the version. The version
bump of a release therefore turns this probe red until both files follow it.

## The steps

| # | Step | Holds when | Fatal |
|---|------|------------|-------|
| 1 | `scaffold` | the probe crate exists outside the workspace | yes |
| 2 | `add-deps` | the Quick Start's dependency block names `rlevo` alone. Every other crate is one the learner adds by hand | no |
| 3 | `build-listing` | the program compiles, as written | yes |
| 4 | `run-listing` | it exits zero and prints a line that starts with `Episode over: ` on stdout. The Quick Start prints it after its episode ends, so a program that exits zero having done nothing fails | yes |

`xtask` checks all four. Unlike the other walks' binaries, the program
prints no step lines: `xtask` reads its exit status and its stdout. A program
that exits non-zero fails step 4 with the last line it printed on stderr,
where a `main` that returns an error prints it.

Today a run prints this, and exits zero:

```text
LEARNER-STEP 1 PASS scaffold :: probe created at <run>/probe (outside the workspace)
LEARNER-STEP 2 PASS add-deps :: rlevo alone is sufficient
LEARNER-STEP 3 PASS build-listing :: the listing compiles as written
LEARNER-STEP 4 PASS run-listing :: it exited zero and printed `Episode over: Terminated`
Learner-1 passes.
```

## What it does not copy yet

The umbrella's examples. A learner who clones the repository runs them with
`cargo run -p rlevo --example <name>`; one who copies an example into a
crate of their own cannot yet, because most include a sibling module or name
a member crate (`rlevo_environments`, `rlevo_core`) that `rlevo` alone does
not give them. Each joins this probe when it builds with `rlevo` alone. The
note line says so on every run.

## Files

| File | What it is |
|------|------------|
| `Cargo.toml.template` | the manifest's package table; the Quick Start's dependency block is appended to it |

The walk is `LEARNER1` in `xtask/src/commands/probe/walk.rs`, and the
listing is `QUICK_START` in `xtask/src/commands/probe/listing.rs`.
