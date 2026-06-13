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
# Rebuild the Leptos/WASM report client (auto-run before every report-* recipe).
client-build:
    cd crates/rlevo-benchmarks-report-client && trunk build --release

report-ppo-cartpole: client-build
    cargo run -p rlevo-examples --features viz-report --example report_ppo_cartpole_with_client --release

report-sphere: client-build
    cargo run -p rlevo-examples --features viz-report --example report_sphere_landscape_with_client --release

report-grids: client-build
    cargo run -p rlevo-examples --features viz-report --example report_grids_with_client  --release

report-toy-text: client-build
    cargo run -p rlevo-examples --features viz-report --example report_toy_text_with_client --release

report-inverted-pendulum: client-build
    cargo run -p rlevo-examples --features locomotion,viz-report --example report_inverted_pendulum_with_client  --release

report-lunar-lander: client-build
    cargo run -p rlevo-examples --features box2d,viz-report --example report_lunar_lander_with_client  --release

# ── Crate-level integration tests ───────────────────────────────────────────

# Wire-format mirror parity: native EpisodeRecord vs browser-client mirror round-trip.
test-wire-format:
    cargo test -p rlevo-benchmarks --test wire_format_compat --features record

# Grid env solvability canary: scripted optimal rollout through every grids env.
test-grids-solvable:
    cargo test -p rlevo-environments --test grids_solvable

# AsciiRenderable coverage: render contract across classic/grids/toy_text/landscapes/box2d.
test-render-coverage:
    cargo test -p rlevo-environments --test render_coverage

# Cross-backend parity: GA drives Sphere-D10 to a non-trivial optimum on both Flex and wgpu.
test-backend-parity:
    cargo test -p rlevo-evolution --test backend_parity

# Co-evolution forgetting prevention: HoF retains solver coverage in a non-stationary host–parasite game.
test-coevo-forgetting:
    cargo test -p rlevo-evolution --test coevolution_forgetting

# Co-evolution forgetting observer [ignored: prints trajectory statistics, no assertion].
test-coevo-forgetting-observe:
    cargo test -p rlevo-evolution --test coevolution_forgetting -- observe_dynamics --ignored --nocapture

# Bit-exact determinism: every EA strategy produces identical fitness trajectory from same seed.
test-evolution-determinism:
    cargo test -p rlevo-evolution --test determinism

# EDA plumbing smoke: each ProbabilityModel completes ask→evaluate→tell with finite metrics.
test-eda-smoke:
    cargo test -p rlevo-evolution --test eda_smoke

# ── Integration tests (rlevo umbrella) ──────────────────────────────────────

# Cross-crate integration: core + RL replay/metrics.
test-integration:
    cargo test -p rlevo --test integration_test

# Recording/episode-count off-by-one regression.
test-recording:
    cargo test -p rlevo --test recording_episode_count --features="viz-report"

# Evaluator harness smoke tests.
test-evaluator:
    cargo test -p rlevo --test evaluator_smoke

# Rastrigin-D10 end-to-end suite (real-valued EA strategies via harness).
test-rastrigin:
    cargo test -p rlevo --test rastrigin_run_suite

# Every shipping swarm strategy on Rastrigin-D10 and Ackley-D10.
test-swarm:
    cargo test -p rlevo --test swarm_rastrigin_suite

# Memetic headline acceptance: wrapper beats bare DE on evals-to-target (>=30% fewer).
test-memetic:
    cargo test -p rlevo --test memetic_rastrigin

# Memetic calibration explorer [ignored: multi-seed sweep for re-pinning the margin].
test-memetic-calibration:
    cargo test -p rlevo --test memetic_rastrigin --release -- calibration_explorer --ignored --nocapture

# EDA convergence: all four ProbabilityModels reach the Sphere-D10 gate, and
# MIMIC beats UMDA's median on Rosenbrock-D10 across 9 fixed seeds.
test-eda:
    cargo test -p rlevo --test eda_convergence

# EDA determinism: all five ProbabilityModels produce bit-identical trajectories from the same seed.
test-eda-determinism:
    cargo test -p rlevo --test determinism

# Co-evolution harness: CoEvolutionaryHarness driven by rlevo-benchmarks Evaluator end-to-end.
test-coevo-harness:
    cargo test -p rlevo --test coevolution_harness

# Post-run record → report pipeline smoke [requires viz-report feature].
test-report-smoke:
    cargo test -p rlevo --test cartpole_report_smoke --features viz-report

# DQN — normal (non-ignored) tests.
test-dqn:
    cargo test -p rlevo --test dqn_integration

# DQN — CartPole reaches 100 [ignored: smoke run].
test-dqn-cart-pole:
    cargo test -p rlevo --test dqn_integration -- dqn_cart_pole_reaches_100 --ignored

# DQN — short-run finite-rewards check [ignored: smoke run].
test-dqn-short:
    cargo test -p rlevo --test dqn_integration -- dqn_short_run_produces_finite_rewards --ignored

# C51 — normal (non-ignored) tests.
test-c51:
    cargo test -p rlevo --test c51_integration

# C51 — CartPole reaches 100 [ignored: smoke run].
test-c51-cart-pole:
    cargo test -p rlevo --test c51_integration -- c51_cart_pole_reaches_100 --ignored

# QR-DQN — normal (non-ignored) tests.
test-qrdqn:
    cargo test -p rlevo --test qrdqn_integration

# QR-DQN — CartPole reaches 50 [ignored: smoke run].
test-qrdqn-cart-pole:
    cargo test -p rlevo --test qrdqn_integration --release -- qrdqn_cart_pole_reaches_50 --ignored

# QR-DQN — full acceptance target [ignored: ~500k Flex CPU steps].
test-qrdqn-full:
    cargo test -p rlevo --test qrdqn_integration --release -- qrdqn_solves_cart_pole_flex_seed_42 --ignored

# PPO — normal (non-ignored) tests.
test-ppo:
    cargo test -p rlevo --test ppo_integration

# PPO — CartPole reaches 100 [ignored: smoke run].
test-ppo-cart-pole:
    cargo test -p rlevo --test ppo_integration -- ppo_cart_pole_reaches_100 --ignored

# PPO — Pendulum improves over random [ignored: ~30s on Flex].
test-ppo-pendulum:
    cargo test -p rlevo --test ppo_integration -- ppo_pendulum_improves_over_random --ignored

# PPG — normal (non-ignored) tests.
test-ppg:
    cargo test -p rlevo --test ppg_integration

# PPG — CartPole reaches modest threshold [ignored: smoke run].
test-ppg-cart-pole:
    cargo test -p rlevo --test ppg_integration -- ppg_cart_pole_reaches_modest_threshold --ignored

# PPG — without aux phase matches PPO baseline [ignored: smoke run].
test-ppg-no-aux:
    cargo test -p rlevo --test ppg_integration -- ppg_without_aux_phase_matches_ppo_baseline --ignored

# PPG — aux phase actually runs [ignored: smoke run].
test-ppg-aux:
    cargo test -p rlevo --test ppg_integration -- ppg_aux_phase_actually_runs --ignored

# PPG — CartPole reaches 475 [ignored: macro convergence, ~2–5 min on Flex].
test-ppg-full:
    cargo test -p rlevo --test ppg_integration -- ppg_cart_pole_reaches_475 --ignored

# DDPG — normal (non-ignored) tests.
test-ddpg:
    cargo test -p rlevo --test ddpg_integration

# DDPG — Pendulum smoke [ignored: ~50k Pendulum steps].
test-ddpg-pendulum:
    cargo test -p rlevo --test ddpg_integration -- ddpg_pendulum_smoke --ignored

# TD3 — normal (non-ignored) tests.
test-td3:
    cargo test -p rlevo --test td3_integration

# TD3 — Pendulum smoke [ignored: macro run, ~500k steps].
test-td3-pendulum:
    cargo test -p rlevo --test td3_integration -- td3_pendulum_smoke --ignored

# SAC — normal (non-ignored) tests.
test-sac:
    cargo test -p rlevo --test sac_integration

# SAC — Pendulum smoke [ignored: macro run, ~500k steps].
test-sac-pendulum:
    cargo test -p rlevo --test sac_integration -- sac_pendulum_smoke --ignored

# Run all ignored integration tests as independent jobs.
# Use `just --parallel test-all-ignored` in CI to run each job concurrently.
test-all-ignored: \
    test-dqn-cart-pole \
    test-dqn-short \
    test-c51-cart-pole \
    test-qrdqn-cart-pole \
    test-qrdqn-full \
    test-ppo-cart-pole \
    test-ppo-pendulum \
    test-ppg-cart-pole \
    test-ppg-no-aux \
    test-ppg-aux \
    test-ppg-full \
    test-ddpg-pendulum \
    test-td3-pendulum \
    test-sac-pendulum

# ── viz-examples CI targets ─────────────────────────────────────────────────

build-viz-examples:
    cargo build -p rlevo-examples --examples \
        --features viz-tui,viz-report,box2d,locomotion

clippy-viz-examples:
    cargo clippy -p rlevo-examples --examples \
        --features viz-tui,viz-report,box2d,locomotion \
        -- -D warnings

# ── Common checks ────────────────────────────────────────────────────────────

test:
    cargo test --workspace

lint:
    cargo clippy --all-targets --all-features
