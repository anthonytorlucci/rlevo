//! End-to-end integration tests for the DQN agent.
//!
//! These tests wire [`DqnAgent`] against the shared seeded `CartPole` fixture
//! ([`rlevo_test_support::env::cartpole_seeded`]) and assert learning
//! behaviour. They intentionally use modest step budgets so `cargo test`
//! remains tractable; the longer macro-bench targets live in
//! `benches/dqn_bench.rs`.
//!
//! The `Flex` determinism preamble ([`flex_guard`] / [`seeded_device`]) and the
//! acceptance assertions live in the `rlevo-test-support` dev-crate; only the
//! DQN network and the algorithm-specific tests remain here.

use burn::module::{AutodiffModule, Module};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Tensor, activation};

use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_environments::classic::cartpole::{CartPoleAction, CartPoleObservation};
use rlevo_reinforcement_learning::algorithms::dqn::dqn_agent::DqnAgent;
use rlevo_reinforcement_learning::algorithms::dqn::dqn_config::DqnTrainingConfigBuilder;
use rlevo_reinforcement_learning::algorithms::dqn::dqn_model::DqnModel;
use rlevo_reinforcement_learning::algorithms::dqn::train::train;
use rlevo_reinforcement_learning::utils::{PolyakError, polyak_update};

use rlevo_test_support::assert::assert_all_finite;
use rlevo_test_support::env::cartpole_seeded;
use rlevo_test_support::flex::{FlexAutodiff as Be, flex_guard, seeded_device};
use rlevo_test_support::{TrainOutcome, rl_learning_test, rl_reproducibility_test};

// ---------------------------------------------------------------------------
// Shared MLP (mirror of the dqn_cart_pole example). Implements DQN's model
// trait and so stays test-local.
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
struct DqnMlp<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    l3: Linear<B>,
}

impl<B: Backend> DqnMlp<B> {
    fn new(device: &<B as burn::tensor::backend::BackendTypes>::Device) -> Self {
        Self {
            l1: LinearConfig::new(4, 64).init(device),
            l2: LinearConfig::new(64, 64).init(device),
            l3: LinearConfig::new(64, 2).init(device),
        }
    }

    fn forward_impl(&self, observations: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = activation::relu(self.l1.forward(observations));
        let x = activation::relu(self.l2.forward(x));
        self.l3.forward(x)
    }
}

impl<B: AutodiffBackend> DqnModel<B, 2> for DqnMlp<B> {
    fn forward(&self, observations: Tensor<B, 2>) -> Tensor<B, 2> {
        self.forward_impl(observations)
    }

    fn forward_inner(
        inner: &Self::InnerModule,
        observations: Tensor<B::InnerBackend, 2>,
    ) -> Tensor<B::InnerBackend, 2> {
        inner.forward_impl(observations)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn soft_update(
        active: &Self,
        target: Self::InnerModule,
        tau: f64,
    ) -> Result<Self::InnerModule, PolyakError> {
        polyak_update::<B::InnerBackend, DqnMlp<B::InnerBackend>>(
            &active.valid(),
            target,
            tau as f32,
        )
    }
}

type Agent = DqnAgent<Be, DqnMlp<Be>, CartPoleObservation, CartPoleAction, 1, 2>;

/// Constructs a deterministic [`DqnAgent`] for `CartPole`.
///
/// Seeds the backend via [`seeded_device`] so two calls with the same `seed`
/// produce bit-for-bit identical initial weights. Callers must hold the
/// [`flex_guard`] lock for the duration of the test (see that function for the
/// process-global Flex RNG / rayon-pinning rationale).
///
/// The hyperparameters (ε-greedy schedule, τ, γ, learning rate, buffer
/// capacity) are tuned for the 4-observation / 2-action `CartPole` task and
/// intentionally conservative so training converges well within 30 000 steps.
fn fresh_agent(seed: u64) -> Agent {
    let device = seeded_device::<Be>(seed);
    let config = DqnTrainingConfigBuilder::new()
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
        .double_q(false)
        .build()
        .expect("valid config");
    let model: DqnMlp<Be> = DqnMlp::new(&device);
    DqnAgent::new(model, config, device).expect("valid config")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Builds and trains a DQN agent on the shared seeded `CartPole` for `total`
/// steps, returning the standardised outcome consumed by the suite macros.
/// Shared by the convergence and reproducibility tests below.
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

// `CartPole` target: after 30k steps with a 2x64 MLP, the 100-episode moving
// average should comfortably exceed 100. The smoke run during development
// reached ~183 on seed=42; we assert a conservative floor of 100. (Generated
// by `rl_learning_test!`.)
rl_learning_test! {
    #[ignore = "30 000-step CartPole training run (several minutes on CPU); confirms avg reward ≥ 100 — run with `cargo test -- --ignored`"]
    dqn_cartpole_converges,
    reaches(100.0),
    seed = 42,
    total = 30_000,
    run = run_cartpole,
}

// Reproducibility smoke test: two back-to-back runs from the same seed produce
// identical reward sequences. (Generated by `rl_reproducibility_test!`.)
rl_reproducibility_test! {
    #[ignore = "6 000-step reproducibility check (two sequential CartPole runs); run with `cargo test -- --ignored`"]
    dqn_cartpole_flex_reproducibility,
    seq,
    seed = 123,
    total = 3_000,
    run = run_cartpole,
}

/// Smoke test: a short run completes with finite rewards and a populated
/// buffer. Protects against silent regressions that would NaN-out.
/// The `flex_guard` lock serializes execution within this binary so
/// Burn's process-global Flex RNG stays isolated.
#[test]
#[ignore = "2 500-step Flex backend smoke run (~30 s); checks for NaN rewards and non-empty replay buffer — run with `cargo test -- --ignored`"]
fn dqn_cartpole_produces_finite_rewards() {
    let _guard = flex_guard();
    let seed: u64 = 7;
    let mut env = cartpole_seeded(seed);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut agent = fresh_agent(seed);
    train(&mut agent, &mut env, &mut rng, 2_500, 0).expect("training");
    assert!(agent.buffer_len() > 0, "buffer should have transitions");
    let rewards: Vec<f32> = agent
        .stats()
        .recent_history
        .iter()
        .map(|m| m.reward)
        .collect();
    assert_all_finite("reward", &rewards);
}
