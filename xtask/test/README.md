# Test tiers

`cargo xtask test <tier> [selector] [extra cargo-test args...]`

| Tier | What | Cost | Runs in |
| --- | --- | --- | --- |
| `fast` | Every crate's default test set — the pull-request gate | seconds per crate | `crate-tests.yml`, per-crate matrix |
| `heavy` | The `#[ignore]`d suite, release profile | minutes to hours | `weekly-tests.yml`, Sundays 02:00 |
| `gpu` | wgpu/flex backend parity | minutes | **nowhere** — local only |

```bash
cargo xtask test fast                    # everything; the one to run after a change
cargo xtask test fast rlevo-core         # one crate, as CI's matrix does
cargo xtask test heavy                   # every heavy unit
cargo xtask test heavy ppo_integration   # one unit
cargo xtask test heavy qrdqn_integration --skip qrdqn_cartpole_acceptance
cargo xtask test gpu                     # needs a real adapter
```

Exit code is 0 on success, 1 if any unit failed, 90 on a usage error. CI calls
`./xtask/test/run.sh` directly rather than `cargo xtask test`, to avoid
compiling the dispatcher in every matrix job.

## Benchmarks are never run here

`cargo bench`, deliberately, per `docs/rules.md`'s bench placement rule.

**Do not add `--all-targets` to any tier.** It includes `--benches`, and with
`harness = false` criterion targets that *executes* all 19 benchmark binaries
at full measurement — about 14 hours — while still not running the `#[ignore]`d
tests, which need `-- --ignored` to lift. That combination is strictly worse
coverage than `fast` for four orders of magnitude more time, and it is the
reason this runner exists: the tiers used to be folklore spread across a
workflow file, a gitignored zsh script, and `#[ignore]` attribute messages, so
the obvious-looking invention was the catastrophic one.

## The `gpu` tier has no CI home

Three tests verify that the wgpu backend agrees with flex:

- `rlevo-evolution` → `tests/backend_parity.rs`
- `rlevo-reinforcement-learning` → `tests/c51_projection_backend_parity.rs`
- `rlevo-reinforcement-learning` → `clamp_preserving_nan_matches_flex_on_wgpu`
  (in-source; the crate's only `#[ignore]`d lib test, so `--lib -- --ignored`
  selects exactly it)

Every workflow here runs on `ubuntu-latest`, which has no GPU, and
`cubecl-wgpu` aborts on device init rather than failing gracefully — so these
cannot be gated in CI as things stand. Before this tier existed they were run
by *nothing*: `#[ignore]` hid them from the pull-request gate, and the weekly
matrix only reaches `-p rlevo --test <bin>`, which is the wrong crate for all
three. They are the class of defect no other test can catch, so run this tier
by hand after touching a backend, a kernel, or a tensor op.

## Adding tests

Nothing needs editing here in the common case:

- A **new crate** joins `fast` by existing — crates are enumerated from
  `crates/*/`, not from a list. The workflow matrix still names crates
  literally (Actions cannot expand a glob), and `test-matrix-coverage` in
  `crate-tests.yml` is what keeps that honest.
- A **new `#[ignore]`d test** in an existing `crates/rlevo/tests/*.rs` binary
  joins `heavy` automatically, for the same reason.
- A **new test binary outside `crates/rlevo/tests`** with `#[ignore]`d tests
  does need a line in `heavy_units`.
- A **feature override** — a target that compiles to nothing without a feature
  — needs an entry in `features_for` (fast) or `heavy_features_for` (heavy).
  Both existing entries are there because a bare `cargo test -p <crate>`
  silently skips the target and reports success.
