# rlevo task runner — canonical commands so example names never get mistyped.
# Install `just` (https://github.com/casey/just), then run `just` to list recipes.
#
# Every example below uses the correct `-p <package>` for the crate that OWNS it
# (examples are auto-discovered per package — see rules.md §11 / ADR 0012).

# List all recipes (default).
default:
    @just --list

# ── Discover examples ────────────────────────────────────────────────────────

# Print the exact, spellable example target names for every package.
list-examples:
    @echo "rlevo:"          && cargo run -q -p rlevo          --example 2>&1 | grep '^ ' || true
    @echo "rlevo-examples:" && cargo run -q -p rlevo-examples --example 2>&1 | grep '^ ' || true

# ── rlevo umbrella examples (5 library crates only, lightweight) ─────────────

evo-ackley:
    cargo run -p rlevo --example ackley_showcase

evo-rastrigin:
    cargo run -p rlevo --example rastrigin_showcase

evo-sphere:
    cargo run -p rlevo --example sphere_showcase

# Tier 1 — scalable n-D landscapes
evo-rosenbrock:
    cargo run -p rlevo --example rosenbrock_showcase

evo-griewank:
    cargo run -p rlevo --example griewank_showcase

evo-schwefel:
    cargo run -p rlevo --example schwefel_showcase

evo-michalewicz:
    cargo run -p rlevo --example michalewicz_showcase

evo-penalized1:
    cargo run -p rlevo --example penalized1_showcase

# Tier 2 — classical 2-D landscapes
evo-branin:
    cargo run -p rlevo --example branin_showcase

evo-himmelblau:
    cargo run -p rlevo --example himmelblau_showcase

evo-six-hump-camel:
    cargo run -p rlevo --example six_hump_camel_showcase

evo-easom:
    cargo run -p rlevo --example easom_showcase

evo-goldstein-price:
    cargo run -p rlevo --example goldstein_price_showcase

evo-cross-in-tray:
    cargo run -p rlevo --example cross_in_tray_showcase

evo-bukin6:
    cargo run -p rlevo --example bukin6_showcase

# Tier 3 — stress-test landscapes
evo-lunacek:
    cargo run -p rlevo --example lunacek_showcase

evo-deb1:
    cargo run -p rlevo --example deb1_showcase

evo-needle-eye:
    cargo run -p rlevo --example needle_eye_showcase

evo-eggholder:
    cargo run -p rlevo --example eggholder_showcase

evo-alpine1:
    cargo run -p rlevo --example alpine1_showcase

evo-rosenbrock-flat:
    cargo run -p rlevo --example rosenbrock_flat_showcase

evo-trefethen:
    cargo run -p rlevo --example trefethen_showcase

# Memetic: bare DE vs MemeticWrapper<DE, HillClimbing>, Rastrigin-D10 evals-to-target.
evo-memetic:
    cargo run --release -p rlevo --example memetic_showcase

cartpole-timelimit:
    cargo run -p rlevo --example cartpole_timelimit

grid-door-key:
    cargo run -p rlevo --example grid_door_key_scripted

# ── rlevo-examples (heavy: benchmarks + viz/record/report features) ──────────
# Not in default-members; each recipe supplies the required feature flags.

# Phase 3c: competitive co-evolution predator–prey arms race on a separable quadratic.
coevo-competitive:
    cargo run --release -p rlevo-examples --example competitive_predator_prey

# Phase 3c: cooperative CCGA on 6-D Rastrigin split across two 3-D sub-populations.
coevo-cooperative:
    cargo run --release -p rlevo-examples --example cooperative_ccga_rastrigin

harness-ga-rastrigin:
    cargo run -p rlevo-examples --example ga_rastrigin

# EDAs: UMDA vs MIMIC on Rosenbrock (dependency capture) + PBIL vs cGA on OneMax
# (probability-vector convergence). Prints model internals each generation.
eda-showcase:
    cargo run --release -p rlevo-examples --example eda_showcase

harness-tabular-bandit:
    cargo run -p rlevo-examples --example tabular_bandit

tui-ppo-cartpole:
    cargo run -p rlevo-examples --features viz-tui --example tui_ppo_cartpole

# `trunk build` stamps the wire FORMAT_VERSION into dist/wire-version.txt
# (Trunk.toml post_build hook); the report emitter refuses a dist/ that lags
# the source. Every `report-*` recipe depends on this so the bundle is never
# stale — re-running is cheap (~0.2s) when nothing changed. Needs `trunk`
# (cargo install trunk) + the wasm32-unknown-unknown target.
#
# NOTE: the target must be installed for the *pinned* toolchain (rust-toolchain.toml,
# currently 1.94.1), not just the default. A rustup update can leave the pinned
# toolchain without wasm32, surfacing as `error[E0463]: can't find crate for core`.
# The `rustup target add` below is idempotent and self-heals that case.
#
# Rebuild the Leptos/WASM report client (auto-run before every report-* recipe).
client-build:
    rustup target add wasm32-unknown-unknown
    cd crates/rlevo-benchmarks-report-client && trunk build --release

report-ppo-cartpole: client-build
    cargo run -p rlevo-examples --features viz-report --example report_ppo_cartpole_with_client --release

report-sphere: client-build
    cargo run -p rlevo-examples --features viz-report --example report_sphere_landscape_with_client --release

# ── Single-test runner ──────────────────────────────────────────────────────
# Generic slot for any crate-level or umbrella integration test that doesn't
# need a permanent name. Examples:
#   just test-one rlevo-evolution backend_parity
#   just test-one rlevo-hybrid cartpole_smoke
#   just test-one rlevo-benchmarks wire_format_compat --features record
#   just test-one rlevo ddpg_integration
test-one PACKAGE TEST *ARGS='':
    cargo test -p {{PACKAGE}} --test {{TEST}} {{ARGS}}

# ── Heavy / #[ignore]d tests (maintainer-run; not part of any CI gate) ───────
#
# These are excluded from `cargo test`'s default run via #[ignore], so neither
# crate-tests.yml's per-crate matrix nor `test-workspace` ever executes them.
# weekly-tests.yml runs each binary's *entire* ignored suite on a schedule
# (`cargo test --release --test <bin> -- --ignored`) — coarse-grained, whole
# binary, ~45min timeout (350min for the QR-DQN acceptance run). The
# `test-one` invocations below are for iterating on ONE of those tests locally
# without waiting on its siblings in the same binary.
#
#   just test-one rlevo-evolution coevolution_forgetting --release -- observe_dynamics --ignored --nocapture
#       Prints host–parasite trajectory statistics; no assertion, read-only diagnostic.
#
#   just test-one rlevo memetic_rastrigin --release -- calibration_explorer --ignored --nocapture
#       Multi-seed sweep for re-pinning the memetic-vs-bare-DE margin (test-memetic's
#       >=30% threshold). Run after touching DE, HillClimbing, or the wrapper's budget split.
#
#   just test-one rlevo ppo_integration -- ppo_pendulum_improves_over_random --ignored
#       ~30s on Flex. PPO's only continuous-control regression check (CartPole cases run unignored).
#
#   just test-one rlevo ppg_integration -- ppg_without_aux_phase_matches_ppo_baseline --ignored
#   just test-one rlevo ppg_integration -- ppg_aux_phase_actually_runs --ignored
#       Smoke runs: PPG without aux phase must not regress vs. plain PPO; aux phase must
#       actually execute (not silently no-op).
#
#   just test-one rlevo td3_integration -- td3_linear_improves_over_random --ignored
#       ~8,000-step run. TD3's LinearEnv acceptance check; Pendulum stays weekly-only.
#
#   DDPG, TD3 (Pendulum case), SAC: fast plumbing-only tests run unignored via
#   `just test-one rlevo <ddpg|td3|sac>_integration`. Their heavy LinearEnv/Pendulum
#   cases have no individual test-one shortcut here — run the whole ignored
#   suite locally with `cargo test --release -p rlevo --test <bin> -- --ignored`
#   (weekly-tests.yml matrix — .github/workflows/weekly-tests.yml).

# Note: integration_test, recording_episode_count, evaluator_smoke,
# rastrigin_run_suite, swarm_rastrigin_suite, and cartpole_report_smoke (all
# `crates/rlevo/tests/`) are covered by crate-tests.yml's `rlevo` matrix entry
# (`cargo test --package rlevo --features viz-report` — a superset of the
# default features, so every non-ignored rlevo test binary runs there). Use
# `just test-one rlevo <name> [--features viz-report]` to run one in isolation.

# ── viz-examples CI targets ─────────────────────────────────────────────────

build-vis-examples:
    cargo build -p rlevo-examples --examples \
        --features viz-tui,viz-report,box2d,locomotion

clippy-vis-examples:
    cargo clippy -p rlevo-examples --examples \
        --features viz-tui,viz-report,box2d,locomotion \
        -- -D warnings

# ── Common checks ────────────────────────────────────────────────────────────

# Fast workspace tests — all long-running RL training tests carry #[ignore] and are excluded.
test-workspace:
    cargo test --workspace --exclude rlevo-examples

lint:
    cargo clippy --all-targets --all-features
