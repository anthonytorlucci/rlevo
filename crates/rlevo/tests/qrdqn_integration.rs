//! End-to-end integration tests for the QR-DQN (Quantile Regression DQN)
//! agent.
//!
//! Mirrors the shape of `c51_integration.rs`: modest step budgets on the shared
//! seeded `CartPole` fixture for `cargo test` throughput, with heavier
//! reproducibility checks gated behind `#[ignore]` because Burn's Flex backend
//! shares a global RNG.
//!
//! The `Flex` determinism preamble ([`flex_guard`] / [`seeded_device`]) and the
//! acceptance assertions live in the `rlevo-test-support` dev-crate; only the
//! QR-DQN network and the algorithm-specific tests remain here.

use burn::tensor::Device;

use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo::harness::evaluator::train;
use rlevo_environments::classic::cartpole::{CartPoleAction, CartPoleObservation};
use rlevo_reinforcement_learning::algorithms::dqn::dqn_model::{
    MlpDistributionalQ, MlpDistributionalQConfig,
};
use rlevo_reinforcement_learning::algorithms::qrdqn::qrdqn_agent::QrDqnAgent;
use rlevo_reinforcement_learning::algorithms::qrdqn::qrdqn_config::QrDqnTrainingConfig;
use rlevo_reinforcement_learning::target::TargetUpdate;

use rlevo_test_support::assert::assert_all_finite;
use rlevo_test_support::env::cartpole_seeded;
use rlevo_test_support::flex::{flex_guard, seeded_device};
use rlevo_test_support::{TrainOutcome, rl_learning_test, rl_reproducibility_test};

// ---------------------------------------------------------------------------
// The crate's default distributional network, 4 -> 64 -> 64 -> 2 x num_quantiles.
// ---------------------------------------------------------------------------

const N_ACTIONS: usize = 2;

fn qrdqn_network(num_quantiles: usize, device: &Device) -> MlpDistributionalQ {
    MlpDistributionalQConfig {
        obs_dim: 4,
        hidden: vec![64, 64],
        num_actions: N_ACTIONS,
        num_atoms: num_quantiles,
    }
    .try_init(device)
    .expect("valid network config")
}

type Agent = QrDqnAgent<MlpDistributionalQ, CartPoleObservation, CartPoleAction, 1, 2>;

/// Build a deterministic `QrDqnAgent` for `CartPole` with `num_quantiles = 21`
/// and `kappa = 1.0`.
///
/// Seeds the backend via [`seeded_device`] so two calls with the same `seed`
/// produce identical initial weights. Callers must hold the [`flex_guard`] lock
/// for the duration of the test.
///
/// # Quantile count
///
/// `num_quantiles = 21` rather than the canonical 200. QR-DQN's quantile Huber loss uses a
/// `(batch, N, N)` broadcast — `$O(N^2)$` per learn step — so a smaller N keeps CI wall-clock
/// practical on Flex CPU without changing the algorithmic correctness being tested.
///
/// The target rule is `polyak(0.005, 1)` — a soft update on every gradient step. This is the
/// behaviour this test has always exercised: it used to be spelled
/// `.tau(0.005).target_update_frequency(500)`, in which the `500` was **inert** (the hard path
/// self-gated to a no-op whenever `$\tau > 0$`). Transcribing the `500` as a cadence would be a
/// `$500\times$` slowdown of the soft update, not a faithful translation.
fn fresh_agent(seed: u64) -> Agent {
    let device = seeded_device(seed);
    let num_quantiles = 21; // smaller than canonical 200; see doc comment above
    let config = QrDqnTrainingConfig {
        batch_size: 64,
        gamma: 0.99,
        target_update: TargetUpdate::polyak(0.005, 1),
        learning_rate: 5e-4,
        epsilon_start: 1.0,
        epsilon_end: 0.05,
        epsilon_decay: 0.9995,
        learning_starts: 1_000,
        train_frequency: 4,
        replay_buffer_capacity: 50_000,
        num_quantiles,
        kappa: 1.0,
        ..Default::default()
    };
    let model = qrdqn_network(num_quantiles, &device);
    QrDqnAgent::new(model, config, device).expect("valid config")
}

/// Builds and trains a QR-DQN agent on the shared seeded `CartPole` for `total`
/// steps, returning the standardised outcome consumed by the suite macros.
/// Shared by the convergence, acceptance, and reproducibility tests below.
fn run_cartpole(seed: u64, total: usize) -> TrainOutcome {
    let mut env = cartpole_seeded(seed);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut agent = fresh_agent(seed);
    let episodes = train(&mut env, &mut agent, total, usize::MAX, &mut rng).expect("training");
    TrainOutcome::from_returns(episodes.iter().map(|e| e.return_value))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Smoke test: a short run completes, the buffer populates, and episode
/// rewards are finite. Catches silent regressions (NaN quantiles, loss
/// numerical blow-up, etc.). The `flex_guard` lock serializes execution within
/// this binary so Burn's process-global Flex RNG stays isolated.
#[test]
#[ignore = "2 500-step Flex backend smoke run; checks for NaN rewards and quantile spread — run with `cargo test -- --ignored`"]
fn qrdqn_cartpole_produces_finite_rewards() {
    let _guard = flex_guard();
    let seed: u64 = 7;
    let mut env = cartpole_seeded(seed);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut agent = fresh_agent(seed);
    let episodes = train(&mut env, &mut agent, 2_500, usize::MAX, &mut rng).expect("training");
    assert!(agent.buffer_len() > 0, "buffer should have transitions");
    assert_all_finite(
        "reward",
        &TrainOutcome::from_returns(episodes.iter().map(|e| e.return_value)).rewards,
    );
    // Every loss stayed finite: no update was skipped for a non-finite one.
    assert_eq!(agent.skipped_updates(), 0, "a loss went non-finite");
}

// Reproducibility smoke test: two seeded back-to-back runs produce identical
// reward sequences. (Generated by `rl_reproducibility_test!`.)
rl_reproducibility_test! {
    #[ignore = "6 000-step reproducibility check (two sequential CartPole runs); run with `cargo test -- --ignored`"]
    qrdqn_cartpole_flex_reproducibility,
    seq,
    seed = 123,
    total = 3_000,
    run = run_cartpole,
}

// `CartPole` target: after 20k steps the 100-episode moving average should
// clear 50 — a rough "is it training?" sanity check tuned for the smaller
// quantile count used in CI. The stricter 195-at-500k-steps acceptance test
// follows below. (Generated by `rl_learning_test!`.)
rl_learning_test! {
    #[ignore = "20 000-step QR-DQN CartPole run (several minutes on Flex CPU); confirms avg reward ≥ 50 — run with `cargo test --release -- --ignored`"]
    qrdqn_cartpole_converges,
    reaches(50.0),
    seed = 42,
    total = 20_000,
    run = run_cartpole,
}

// Full acceptance target: `$\geq 195$` on CartPole-v1 in `$\leq 500\text{k}$` steps at seed 42.
// Ignored by default — too expensive for regular CI. Run with:
// `cargo test -p rlevo --test qrdqn_integration --release -- --ignored qrdqn_cartpole_acceptance`.
// (Generated by `rl_learning_test!`.)
rl_learning_test! {
    #[ignore = "500 000-step QR-DQN CartPole acceptance run (~hours on Flex CPU); confirms avg reward ≥ 195 — run with `cargo test --release -- --ignored qrdqn_cartpole_acceptance`"]
    qrdqn_cartpole_acceptance,
    reaches(195.0),
    seed = 42,
    total = 500_000,
    run = run_cartpole,
}
