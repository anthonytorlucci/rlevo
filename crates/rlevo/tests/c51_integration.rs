//! End-to-end integration tests for the C51 (Categorical DQN) agent.
//!
//! Follows the shape of `dqn_integration.rs`: modest step budgets on the shared
//! seeded `CartPole` fixture for `cargo test` throughput, with a couple of
//! heavier reproducibility checks gated behind `#[ignore]` because Burn's Flex
//! backend shares a global RNG.
//!
//! The `Flex` determinism preamble ([`flex_guard`] / [`seeded_device`]) and the
//! acceptance assertions live in the `rlevo-test-support` dev-crate; only the
//! C51 network and the algorithm-specific tests remain here.

use burn::module::{AutodiffModule, Module};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Tensor, activation};

use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_environments::classic::cartpole::{CartPoleAction, CartPoleObservation};
use rlevo_reinforcement_learning::algorithms::c51::c51_agent::C51Agent;
use rlevo_reinforcement_learning::algorithms::c51::c51_config::{
    C51TrainingConfig, C51TrainingConfigBuilder,
};
use rlevo_reinforcement_learning::algorithms::c51::c51_model::C51Model;
use rlevo_reinforcement_learning::algorithms::c51::train::train;
use rlevo_reinforcement_learning::target::TargetUpdate;
use rlevo_reinforcement_learning::utils::{PolyakError, polyak_update};

use rlevo_test_support::assert::assert_all_finite;
use rlevo_test_support::env::cartpole_seeded;
use rlevo_test_support::flex::{FlexAutodiff as Be, flex_guard, seeded_device};
use rlevo_test_support::{TrainOutcome, rl_learning_test, rl_reproducibility_test};

// ---------------------------------------------------------------------------
// Shared MLP (mirror of the c51_cart_pole example). Implements C51's model
// trait and so stays test-local.
// ---------------------------------------------------------------------------

const N_ACTIONS: usize = 2;

#[derive(Module, Debug)]
struct C51Mlp<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    l3: Linear<B>,
    num_atoms: usize,
}

impl<B: Backend> C51Mlp<B> {
    fn new(num_atoms: usize, device: &<B as burn::tensor::backend::BackendTypes>::Device) -> Self {
        Self {
            l1: LinearConfig::new(4, 64).init(device),
            l2: LinearConfig::new(64, 64).init(device),
            l3: LinearConfig::new(64, N_ACTIONS * num_atoms).init(device),
            num_atoms,
        }
    }

    fn forward_impl(&self, observations: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch_size, _] = observations.dims();
        let x = activation::relu(self.l1.forward(observations));
        let x = activation::relu(self.l2.forward(x));
        let flat = self.l3.forward(x);
        flat.reshape([batch_size, N_ACTIONS, self.num_atoms])
    }
}

impl<B: AutodiffBackend> C51Model<B, 2> for C51Mlp<B> {
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
    fn soft_update(
        active: &Self,
        target: Self::InnerModule,
        tau: f64,
    ) -> Result<Self::InnerModule, PolyakError> {
        polyak_update::<B::InnerBackend, C51Mlp<B::InnerBackend>>(
            &active.valid(),
            target,
            tau as f32,
        )
    }
}

type Agent = C51Agent<Be, C51Mlp<Be>, CartPoleObservation, CartPoleAction, 1, 2>;

/// Constructs a deterministic [`C51Agent`] for `CartPole`.
///
/// Seeds the backend via [`seeded_device`] so two calls with the same `seed`
/// produce identical initial weights. Callers must hold the [`flex_guard`] lock
/// for the duration of the test.
///
/// C51-specific settings: `num_atoms = 51` discretises the return distribution;
/// `v_min = 0.0` / `v_max = 500.0` spans `CartPole`'s full 500-step
/// cumulative-return range. The distributional loss is a categorical projection
/// (cross-entropy over the projected Bellman target), in contrast to DQN's
/// scalar Bellman MSE. The ε-greedy schedule and replay configuration are
/// otherwise equivalent to the DQN integration test.
///
/// The target rule is `polyak(0.005, 1)` — a soft update on every gradient
/// step. This is the behaviour this test has always exercised: it used to be
/// spelled `.tau(0.005).target_update_frequency(500)`, in which the `500` was
/// **inert** (the hard path self-gated to a no-op whenever τ > 0). Transcribing
/// the `500` as a cadence would be a 500× slowdown of the soft update, not a
/// faithful translation.
fn fresh_agent(seed: u64) -> Agent {
    let device = seeded_device::<Be>(seed);
    let num_atoms = 51;
    let config = C51TrainingConfigBuilder::new()
        .batch_size(64)
        .gamma(0.99)
        .target_update(TargetUpdate::polyak(0.005, 1))
        .learning_rate(5e-4)
        .epsilon_start(1.0)
        .epsilon_end(0.05)
        .epsilon_decay(0.9995)
        .learning_starts(1_000)
        .train_frequency(4)
        .replay_buffer_capacity(50_000)
        .num_atoms(num_atoms)
        .v_min(0.0)
        .v_max(500.0)
        .build()
        .expect("valid config");
    let model: C51Mlp<Be> = C51Mlp::new(num_atoms, &device);
    C51Agent::new(model, config, device).expect("valid config")
}

/// Builds and trains a C51 agent on the shared seeded `CartPole` for `total`
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Smoke test: a short run completes, the buffer populates, and episode
/// rewards are finite. Catches silent regressions (NaN logits, projection
/// numerical blow-up, etc.). The `flex_guard` lock serializes execution within
/// this binary so Burn's process-global Flex RNG stays isolated.
#[test]
#[ignore = "2 500-step Flex backend smoke run; checks for NaN rewards — run with `cargo test -- --ignored`"]
fn c51_cartpole_produces_finite_rewards() {
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
}

/// `C51Agent::new` rejects a degenerate support (`v_min == v_max`) at
/// construction time. The config is assembled directly (bypassing the builder's
/// own validation) so the check being exercised is the agent-level one.
#[test]
fn c51_agent_new_rejects_degenerate_support() {
    let _guard = flex_guard();
    let device = seeded_device::<Be>(1);
    let config = C51TrainingConfig {
        v_min: 5.0,
        v_max: 5.0,
        ..C51TrainingConfig::default()
    };
    let model: C51Mlp<Be> = C51Mlp::new(config.num_atoms, &device);
    let result =
        C51Agent::<Be, _, CartPoleObservation, CartPoleAction, 1, 2>::new(model, config, device);
    let err = result.unwrap_err();
    assert_eq!(err.field, "v_max");
}

// Reproducibility smoke test: two seeded back-to-back runs produce identical
// reward sequences. (Generated by `rl_reproducibility_test!`.)
rl_reproducibility_test! {
    #[ignore = "6 000-step reproducibility check (two sequential CartPole runs); run with `cargo test -- --ignored`"]
    c51_cartpole_flex_reproducibility,
    seq,
    seed = 123,
    total = 3_000,
    run = run_cartpole,
}

// `CartPole` target: after 30k steps the 100-episode moving average should
// comfortably clear 100. This is the distributional counterpart to the DQN
// integration test; the modest budget keeps CI fast while still catching
// wholesale training failures. (Generated by `rl_learning_test!`.)
rl_learning_test! {
    #[ignore = "30 000-step C51 CartPole run (several minutes on CPU); confirms avg reward ≥ 100 — run with `cargo test -- --ignored`"]
    c51_cartpole_converges,
    reaches(100.0),
    seed = 42,
    total = 30_000,
    run = run_cartpole,
}
