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

use burn::module::{AutodiffModule, Module};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Tensor, activation};

use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_environments::classic::cartpole::{CartPoleAction, CartPoleObservation};
use rlevo_reinforcement_learning::algorithms::qrdqn::qrdqn_agent::QrDqnAgent;
use rlevo_reinforcement_learning::algorithms::qrdqn::qrdqn_config::QrDqnTrainingConfigBuilder;
use rlevo_reinforcement_learning::algorithms::qrdqn::qrdqn_model::QrDqnModel;
use rlevo_reinforcement_learning::algorithms::qrdqn::train::train;
use rlevo_reinforcement_learning::utils::polyak_update;

use rlevo_test_support::assert::assert_all_finite;
use rlevo_test_support::env::cartpole_seeded;
use rlevo_test_support::flex::{FlexAutodiff as Be, flex_guard, seeded_device};
use rlevo_test_support::{TrainOutcome, rl_learning_test, rl_reproducibility_test};

// ---------------------------------------------------------------------------
// Shared MLP (mirror of the qrdqn_cart_pole example). Implements QR-DQN's model
// trait and so stays test-local.
// ---------------------------------------------------------------------------

const N_ACTIONS: usize = 2;

#[derive(Module, Debug)]
struct QrDqnMlp<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    l3: Linear<B>,
    num_quantiles: usize,
}

impl<B: Backend> QrDqnMlp<B> {
    fn new(
        num_quantiles: usize,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Self {
        Self {
            l1: LinearConfig::new(4, 64).init(device),
            l2: LinearConfig::new(64, 64).init(device),
            l3: LinearConfig::new(64, N_ACTIONS * num_quantiles).init(device),
            num_quantiles,
        }
    }

    fn forward_impl(&self, observations: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch_size, _] = observations.dims();
        let x = activation::relu(self.l1.forward(observations));
        let x = activation::relu(self.l2.forward(x));
        let flat = self.l3.forward(x);
        flat.reshape([batch_size, N_ACTIONS, self.num_quantiles])
    }
}

impl<B: AutodiffBackend> QrDqnModel<B, 2> for QrDqnMlp<B> {
    fn forward(&self, observations: Tensor<B, 2>) -> Tensor<B, 3> {
        self.forward_impl(observations)
    }
    fn forward_inner(
        inner: &Self::InnerModule,
        observations: Tensor<B::InnerBackend, 2>,
    ) -> Tensor<B::InnerBackend, 3> {
        inner.forward_impl(observations)
    }
    #[allow(clippy::cast_possible_truncation)]
    fn soft_update(active: &Self, target: Self::InnerModule, tau: f64) -> Self::InnerModule {
        polyak_update::<B::InnerBackend, QrDqnMlp<B::InnerBackend>>(
            &active.valid(),
            target,
            tau as f32,
        )
    }
}

type Agent = QrDqnAgent<Be, QrDqnMlp<Be>, CartPoleObservation, CartPoleAction, 1, 2>;

/// Build a deterministic `QrDqnAgent` for `CartPole` with `num_quantiles = 21`
/// and `kappa = 1.0`.
///
/// Seeds the backend via [`seeded_device`] so two calls with the same `seed`
/// produce identical initial weights. Callers must hold the [`flex_guard`] lock
/// for the duration of the test.
///
/// # Quantile count
///
/// `num_quantiles = 21` rather than the canonical 200. QR-DQN's quantile
/// Huber loss uses a `(batch, N, N)` broadcast — O(N²) per learn step — so
/// a smaller N keeps CI wall-clock practical on Flex CPU without changing the
/// algorithmic correctness being tested.
fn fresh_agent(seed: u64) -> Agent {
    let device = seeded_device::<Be>(seed);
    let num_quantiles = 21; // smaller than canonical 200; see doc comment above
    let config = QrDqnTrainingConfigBuilder::new()
        .batch_size(64)
        .gamma(0.99)
        .tau(0.005)
        .learning_rate(5e-4)
        .epsilon_start(1.0)
        .epsilon_end(0.05)
        .epsilon_decay(0.9995)
        .learning_starts(1_000)
        .train_frequency(4)
        .target_update_frequency(500)
        .replay_buffer_capacity(50_000)
        .num_quantiles(num_quantiles)
        .kappa(1.0)
        .build()
        .expect("valid config");
    let model: QrDqnMlp<Be> = QrDqnMlp::new(num_quantiles, &device);
    QrDqnAgent::new(model, config, device).expect("valid config")
}

/// Builds and trains a QR-DQN agent on the shared seeded `CartPole` for `total`
/// steps, returning the standardised outcome consumed by the suite macros.
/// Shared by the convergence, acceptance, and reproducibility tests below.
fn run_cartpole(seed: u64, total: usize) -> TrainOutcome {
    let mut env = cartpole_seeded(seed);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut agent = fresh_agent(seed);
    train(&mut agent, &mut env, &mut rng, total, 0).expect("training");
    let stats = agent.stats();
    TrainOutcome {
        avg_score: stats.avg_score().unwrap_or(0.0),
        rewards: stats.recent_history.iter().map(|m| m.reward).collect(),
    }
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
    train(&mut agent, &mut env, &mut rng, 2_500, 0).expect("training");
    assert!(agent.buffer_len() > 0, "buffer should have transitions");
    let history = &agent.stats().recent_history;
    assert_all_finite(
        "reward",
        &history.iter().map(|m| m.reward).collect::<Vec<_>>(),
    );
    assert_all_finite(
        "policy_loss",
        &history.iter().map(|m| m.policy_loss).collect::<Vec<_>>(),
    );
    assert_all_finite(
        "q_mean",
        &history.iter().map(|m| m.q_mean).collect::<Vec<_>>(),
    );
    assert_all_finite(
        "quantile_spread",
        &history
            .iter()
            .map(|m| m.quantile_spread)
            .collect::<Vec<_>>(),
    );
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

// Full acceptance target: ≥ 195 on CartPole-v1 in ≤ 500k steps at seed 42.
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
