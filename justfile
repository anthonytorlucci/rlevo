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
    @echo "rlevo:"           && cargo run -q -p rlevo           --example 2>&1 | grep '^ ' || true
    @echo "rlevo-examples:"  && cargo run -q -p rlevo-examples  --example 2>&1 | grep '^ ' || true
    @echo "---"
    @echo "run examples using cargo run -p {WORKSPACE} --example {EXAMPLE_NAME}"

# ── rlevo-examples (heavy: benchmarks + viz/record/report features) ──────────
# Not in default-members; each recipe supplies the required feature flags.

# competitive co-evolution predator–prey arms race on a separable quadratic.
coevo-competitive:
    cargo run --release -p rlevo-examples --example competitive_predator_prey

# cooperative CCGA on 6-D Rastrigin split across two 3-D sub-populations.
coevo-cooperative:
    cargo run --release -p rlevo-examples --example cooperative_ccga_rastrigin

# EDAs: UMDA vs MIMIC on Rosenbrock (dependency capture) + PBIL vs cGA on OneMax
# (probability-vector convergence). Prints model internals each generation.
eda-showcase:
    cargo run --release -p rlevo-examples --example eda_showcase

# TUI example requires feature viz-tui
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
