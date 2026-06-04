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
    @echo "rlevo-core:"     && cargo run -q -p rlevo-core     --example 2>&1 | grep '^ ' || true
    @echo "rlevo:"          && cargo run -q -p rlevo          --example 2>&1 | grep '^ ' || true
    @echo "rlevo-examples:" && cargo run -q -p rlevo-examples --example 2>&1 | grep '^ ' || true

# ── rlevo-core examples (contract/trait demos, lightweight) ──────────────────

core-constraints:
    cargo run -p rlevo-core --example state_constraints

core-grid:
    cargo run -p rlevo-core --example grid_agent

# ── rlevo umbrella examples (5 library crates only, lightweight) ─────────────

evo-ackley:
    cargo run -p rlevo --example ackley_showcase

evo-rastrigin:
    cargo run -p rlevo --example rastrigin_showcase

evo-sphere:
    cargo run -p rlevo --example sphere_showcase

cartpole-random:
    cargo run -p rlevo --example cartpole_random

cartpole-timelimit:
    cargo run -p rlevo --example cartpole_timelimit

grid-door-key:
    cargo run -p rlevo --example grid_door_key_scripted

mountain-car:
    cargo run -p rlevo --example mountain_car_continuous_random

pendulum-random:
    cargo run -p rlevo --example pendulum_random

# ── rlevo-examples (heavy: benchmarks + viz/record/report features) ──────────
# Not in default-members; each recipe supplies the required feature flags.

harness-ga-rastrigin:
    cargo run -p rlevo-examples --example ga_rastrigin

harness-tabular-bandit:
    cargo run -p rlevo-examples --example tabular_bandit

tui-ppo-cartpole:
    cargo run -p rlevo-examples --features viz-tui --example tui_ppo_cartpole

report-ppo-cartpole:
    cargo run -p rlevo-examples --features viz-report --example report_ppo_cartpole_with_client

report-sphere:
    cargo run -p rlevo-examples --features viz-report --example report_sphere_landscape_with_client

report-grids:
    cargo run -p rlevo-examples --features viz-report --example report_grids_with_client

report-toy-text:
    cargo run -p rlevo-examples --features viz-report --example report_toy_text_with_client

report-inverted-pendulum:
    cargo run -p rlevo-examples --features locomotion,viz-report --example report_inverted_pendulum_with_client

report-lunar-lander:
    cargo run -p rlevo-examples --features box2d,viz-report --example report_lunar_lander_with_client

# ── Common checks ────────────────────────────────────────────────────────────

test:
    cargo test

lint:
    cargo clippy --all-targets --all-features
