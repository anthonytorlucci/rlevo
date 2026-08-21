# BYOE-1 probe

`run.sh` builds a throwaway crate **outside** this workspace to answer one
question: can a researcher who has never seen this repository bring their own
environment and reach a rendered report, using only published crates and
public API?

Spec: `docs/.private/specs/2026-08-21-byoe-first-class-citizen/`.

| File | Role |
| --- | --- |
| `run.sh` | The driver. Packages the publishable crates, stages the tarballs outside the repo, renders the probe, builds it, runs it, reports the first failing step. Bash, not zsh, because `ubuntu-latest` has no zsh. |
| `probe/Cargo.toml.template` | Manifest rendered by the driver. **Not** named `Cargo.toml` — otherwise cargo discovers this directory as a workspace member and `cargo build` at the repo root compiles the probe against path deps, which is exactly the coupling the test exists to avoid. |
| `probe/src/main.rs` | The probe. A building thermostat, deliberately unlike anything in `rlevo-environments`, plus a bang-bang controller. |

## Rules for editing the probe

1. **Public API only.** Reaching into workspace internals, or adding a path
   dependency on a crate in this repo, silently converts the test into a
   restatement of the in-tree smoke tests and it stops measuring anything.
2. **Do not copy from `rlevo-environments`.** The environment must be written
   the way an outsider would write one. Borrowing a built-in env imports
   assumptions — a payload source, an `AsciiRenderable` impl, a family — that
   a third party does not get for free, and every one of those is a finding.
3. **Friction is the output.** When a step needs an awkward workaround, keep
   the workaround and comment it with the blocker it demonstrates rather than
   smoothing it away. The `FINDING (...)` comments in `src/main.rs` are load
   bearing; deleting one deletes evidence.
4. **Every loop over environment steps is bounded.** A defect in a driver
   should fail the test, not hang it — a hang in CI reads as flake.

## Running it

```bash
cargo xtask byoe
```

Exit code is the first failing step (`90` if the script aborted before testing
anything). `BYOE_KEEP=1` keeps the staging directory for inspection;
`BYOE_TRUNK_DIST=<dir>` supplies interactive-client assets out of band for
step 9. `./xtask/byoe/run.sh` works too — the alias just saves the path.

## Current status

As of the first run (2026-08-21) BYOE-1 **fails at step 8 of 9**: steps 1, 3-7
pass, step 2 fails because `rlevo` alone is insufficient, and step 8 fails
because a third-party environment's frames carry `FamilyPayload::Ascii`. The
CI job is `continue-on-error` for exactly this reason. See the spec's Phase 0
results table.
