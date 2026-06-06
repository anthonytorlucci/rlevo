//! End-to-end integration tests for the DQN agent.
//!
//! These tests wire [`DqnAgent`] against real `rlevo-environments` environments and
//! assert learning behaviour. They intentionally use modest step budgets so
//! `cargo test` remains tractable; the longer macro-bench targets live in
//! `benches/dqn_bench.rs`.

use burn::backend::{Autodiff, Flex};
use burn::module::{AutodiffModule, Module};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Tensor, activation};

use rlevo_reinforcement_learning::utils::polyak_update;

use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_environments::classic::cartpole::{
    CartPole, CartPoleAction, CartPoleConfig, CartPoleObservation,
};
use rlevo_reinforcement_learning::algorithms::dqn::dqn_agent::DqnAgent;
use rlevo_reinforcement_learning::algorithms::dqn::dqn_config::DqnTrainingConfigBuilder;
use rlevo_reinforcement_learning::algorithms::dqn::dqn_model::DqnModel;
use rlevo_reinforcement_learning::algorithms::dqn::train::train;

// ---------------------------------------------------------------------------
// Shared MLP and Polyak update helpers (mirror of the dqn_cart_pole example).
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
    fn soft_update(active: &Self, target: Self::InnerModule, tau: f64) -> Self::InnerModule {
        polyak_update::<B::InnerBackend, DqnMlp<B::InnerBackend>>(
            &active.valid(),
            target,
            tau as f32,
        )
    }
}

type Be = Autodiff<Flex>;
type Agent = DqnAgent<Be, DqnMlp<Be>, CartPoleObservation, CartPoleAction, 1, 2>;

static BACKEND_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Constructs a [`DqnAgent`] with a fully-seeded, deterministic configuration
/// for CartPole experiments.
///
/// The Burn `Flex` backend exposes a **process-global** RNG that governs weight
/// initialisation. This function seeds it via [`Backend::seed`] before
/// constructing the model, so that two calls with identical `seed` values
/// produce bit-for-bit identical initial weights — a hard requirement for the
/// `dqn_reproducibility_flex` test.
///
/// # Caller responsibilities
///
/// - **Hold `BACKEND_LOCK` before calling.** The lock serialises access to
///   the global Flex RNG across test threads; without it a concurrent test
///   could interleave a `seed` call and silently corrupt the weight
///   initialisation sequence.
/// - **Pin rayon to one thread** (`rayon::ThreadPoolBuilder::num_threads(1)`)
///   before the training loop. Flex dispatches matrix operations through rayon;
///   floating-point reduction order is non-deterministic under multi-threading,
///   introducing a second source of run-to-run variance independent of the RNG.
///
/// The environment and `StdRng` are seeded separately by each test body and
/// are not managed here. This function is responsible solely for backend-side
/// weight initialisation determinism.
///
/// The hyperparameters (ε-greedy schedule, τ, γ, learning rate, buffer
/// capacity) are tuned for the 4-observation / 2-action CartPole task and
/// intentionally conservative so training converges well within 30 000 steps.
fn fresh_agent(seed: u64) -> Agent {
    let device = Default::default();
    <Be as Backend>::seed(&device, seed);
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
        .build();
    let model: DqnMlp<Be> = DqnMlp::new(&device);
    DqnAgent::new(model, config, device)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// `CartPole` target: after 30k steps with a 2x64 MLP, the 100-episode moving
/// average should comfortably exceed 100. The smoke run during development
/// reached ~183 on seed=42; we assert a conservative floor of 100.
#[test]
#[ignore = "30 000-step CartPole training run (several minutes on CPU); confirms avg reward ≥ 100 — run with `cargo test -- --ignored`"]
fn dqn_cart_pole_reaches_100() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .ok();
    let _guard = BACKEND_LOCK.lock().expect("backend lock");
    let seed: u64 = 42;
    let mut env = CartPole::with_config(CartPoleConfig {
        seed,
        ..CartPoleConfig::default()
    });
    let mut rng = StdRng::seed_from_u64(seed);
    let mut agent = fresh_agent(seed);
    train(&mut agent, &mut env, &mut rng, 30_000, 0).expect("training loop");
    let avg = agent.stats().avg_score().unwrap_or(0.0);
    // The acceptance target is >= 100 within 200k steps; this test trains
    // for 30k to keep CI fast. The smoke/reproducibility tests are
    // `#[ignore]` so they don't compete for Burn's global Flex RNG.
    assert!(avg >= 100.0, "expected avg reward >= 100, got {avg:.2}");
}

/// Reproducibility smoke test: two back-to-back runs from the same seed
/// produce identical reward sequences when run sequentially in the same
/// process. The `BACKEND_LOCK` serializes execution within this binary so
/// Burn's process-global Flex RNG stays isolated.
#[test]
#[allow(clippy::float_cmp)]
fn dqn_reproducibility_flex() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .ok();
    let _guard = BACKEND_LOCK.lock().expect("backend lock");
    fn run(seed: u64, total: usize) -> Vec<f32> {
        let mut env = CartPole::with_config(CartPoleConfig {
            seed,
            ..CartPoleConfig::default()
        });
        let mut rng = StdRng::seed_from_u64(seed);
        let mut agent = fresh_agent(seed);
        train(&mut agent, &mut env, &mut rng, total, 0).expect("training");
        agent
            .stats()
            .recent_history
            .iter()
            .map(|m| m.reward)
            .collect()
    }

    let a = run(123, 3_000);
    let b = run(123, 3_000);
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(x, y, "divergence at episode {i}: {x} vs {y}");
    }
}

/// Smoke test: a short run completes with finite rewards and a populated
/// buffer. Protects against silent regressions that would NaN-out.
/// The `BACKEND_LOCK` serializes execution within this binary so
/// Burn's process-global Flex RNG stays isolated.
#[test]
#[ignore = "2 500-step Flex backend smoke run (~30 s); checks for NaN rewards and non-empty replay buffer — run with `cargo test -- --ignored`"]
fn dqn_short_run_produces_finite_rewards() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .ok();
    let _guard = BACKEND_LOCK.lock().expect("backend lock");
    let seed: u64 = 7;
    let mut env = CartPole::with_config(CartPoleConfig {
        seed,
        ..CartPoleConfig::default()
    });
    let mut rng = StdRng::seed_from_u64(seed);
    let mut agent = fresh_agent(seed);
    train(&mut agent, &mut env, &mut rng, 2_500, 0).expect("training");
    assert!(agent.buffer_len() > 0, "buffer should have transitions");
    for (i, m) in agent.stats().recent_history.iter().enumerate() {
        assert!(m.reward.is_finite(), "non-finite reward at episode {i}");
    }
}
