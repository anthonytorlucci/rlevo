//! End-to-end integration tests for the DDPG agent.
//!
//! Default-run tests exercise DDPG on the shared synthetic 1-D continuous env
//! ([`rlevo_test_support::env::LinearEnv`]) so that `cargo test` stays tractable
//! (no physics simulator, small networks, modest budgets). The Pendulum
//! macro-smoke is gated behind `#[ignore]` per the project's integration-test
//! budget convention.
//!
//! The shared fixture, the `Flex` determinism preamble ([`flex_guard`] /
//! [`seeded_device`]), and the acceptance assertions live in the
//! `rlevo-test-support` dev-crate; only the DDPG networks and the
//! algorithm-specific tests remain here.

use burn::module::{AutodiffModule, Module};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::Tensor;
use burn::tensor::activation::{relu, tanh};
use burn::tensor::backend::{AutodiffBackend, Backend};

use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_core::action::ContinuousAction;
use rlevo_environments::classic::pendulum::{
    Pendulum, PendulumAction, PendulumConfig, PendulumObservation,
};
use rlevo_environments::wrappers::TimeLimit;
use rlevo_reinforcement_learning::algorithms::ddpg::ddpg_agent::DdpgAgent;
use rlevo_reinforcement_learning::algorithms::ddpg::ddpg_config::DdpgTrainingConfigBuilder;
use rlevo_reinforcement_learning::algorithms::ddpg::ddpg_model::{ContinuousQ, DeterministicPolicy};
use rlevo_reinforcement_learning::algorithms::ddpg::train::train;
use rlevo_reinforcement_learning::utils::polyak_update;

use rlevo_test_support::assert::assert_improves_over_random;
use rlevo_test_support::baseline::{random_return, uniform_bounded};
use rlevo_test_support::env::{LinearAction, LinearEnv, LinearObservation};
use rlevo_test_support::flex::{FlexAutodiff as Be, flex_guard, seeded_device};
use rlevo_test_support::{TrainOutcome, rl_learning_test, rl_reproducibility_test};

// ---------------------------------------------------------------------------
// Actor & critic: tiny MLPs (shared between LinearEnv test and Pendulum
// smoke test). These implement DDPG's model traits and so stay test-local.
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
struct Actor<B: Backend> {
    fc1: Linear<B>,
    head: Linear<B>,
    action_scale: f32,
}

impl<B: Backend> Actor<B> {
    fn new(
        obs_dim: usize,
        hidden: usize,
        action_dim: usize,
        scale: f32,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Self {
        Self {
            fc1: LinearConfig::new(obs_dim, hidden).init(device),
            head: LinearConfig::new(hidden, action_dim).init(device),
            action_scale: scale,
        }
    }

    fn forward_impl(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = relu(self.fc1.forward(obs));
        tanh(self.head.forward(h)).mul_scalar(self.action_scale)
    }
}

impl<B: AutodiffBackend> DeterministicPolicy<B, 2, 2> for Actor<B> {
    fn forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        self.forward_impl(obs)
    }
    fn forward_inner(
        inner: &Self::InnerModule,
        obs: Tensor<B::InnerBackend, 2>,
    ) -> Tensor<B::InnerBackend, 2> {
        inner.forward_impl(obs)
    }
    #[allow(clippy::cast_possible_truncation)]
    fn soft_update(active: &Self, target: Self::InnerModule, tau: f64) -> Self::InnerModule {
        polyak_update::<B::InnerBackend, Actor<B::InnerBackend>>(
            &active.valid(),
            target,
            tau as f32,
        )
    }
}

#[derive(Module, Debug)]
struct Critic<B: Backend> {
    fc1: Linear<B>,
    head: Linear<B>,
}

impl<B: Backend> Critic<B> {
    fn new(
        obs_dim: usize,
        action_dim: usize,
        hidden: usize,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Self {
        Self {
            fc1: LinearConfig::new(obs_dim + action_dim, hidden).init(device),
            head: LinearConfig::new(hidden, 1).init(device),
        }
    }
    fn forward_impl(&self, obs: Tensor<B, 2>, act: Tensor<B, 2>) -> Tensor<B, 1> {
        let x = Tensor::cat(vec![obs, act], 1);
        let h = relu(self.fc1.forward(x));
        self.head.forward(h).squeeze_dim::<1>(1)
    }
}

impl<B: AutodiffBackend> ContinuousQ<B, 2, 2> for Critic<B> {
    fn forward(&self, obs: Tensor<B, 2>, act: Tensor<B, 2>) -> Tensor<B, 1> {
        self.forward_impl(obs, act)
    }
    fn forward_inner(
        inner: &Self::InnerModule,
        obs: Tensor<B::InnerBackend, 2>,
        act: Tensor<B::InnerBackend, 2>,
    ) -> Tensor<B::InnerBackend, 1> {
        inner.forward_impl(obs, act)
    }
    #[allow(clippy::cast_possible_truncation)]
    fn soft_update(active: &Self, target: Self::InnerModule, tau: f64) -> Self::InnerModule {
        polyak_update::<B::InnerBackend, Critic<B::InnerBackend>>(
            &active.valid(),
            target,
            tau as f32,
        )
    }
}

// Concrete DDPG agent over the shared 1-D continuous fixture.
type LinearAgent = DdpgAgent<Be, Actor<Be>, Critic<Be>, LinearObservation, LinearAction, 1, 2, 1, 2>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Builds and trains a DDPG agent on the shared `LinearEnv` for `total` steps,
/// returning the standardised outcome consumed by the suite macros. Shared by
/// the convergence and reproducibility tests below.
fn run_linear(seed: u64, total: usize) -> TrainOutcome {
    let device = seeded_device::<Be>(seed);
    let mut env = LinearEnv::with_seed(seed, 20);
    let mut rng = StdRng::seed_from_u64(seed);

    let actor: Actor<Be> = Actor::new(1, 32, 1, 1.0, &device);
    let critic: Critic<Be> = Critic::new(1, 1, 32, &device);
    let config = DdpgTrainingConfigBuilder::new()
        .buffer_capacity(20_000)
        .batch_size(32)
        .learning_starts(500)
        .actor_lr(1e-3)
        .critic_lr(1e-3)
        .gamma(0.99)
        .tau(0.02)
        .exploration_noise(0.2)
        .policy_frequency(2)
        .build();
    let mut agent: LinearAgent = DdpgAgent::new(actor, critic, config, device);

    train::<Be, _, _, _, _, LinearAction, _, 1, 1, 2, 1, 2>(
        &mut agent, &mut env, &mut rng, total, 0,
    )
    .expect("training");

    let stats = agent.stats();
    TrainOutcome {
        avg_score: stats.avg_score().unwrap_or(0.0),
        rewards: stats.recent_history.iter().map(|m| m.reward).collect(),
    }
}

/// Mean episode return of a uniform-random `U(-1, 1)` policy on the shared
/// `LinearEnv`, measured over 200 episodes from `seed`. The learning test below
/// asserts the trained agent beats this measured baseline by a margin.
fn random_linear(seed: u64) -> f32 {
    let mut env = LinearEnv::with_seed(seed, 20);
    let mut rng = StdRng::seed_from_u64(seed);
    random_return(&mut env, 200, 20, &mut rng, uniform_bounded::<1, LinearAction>)
}

/// Mean episode return of a uniform-random torque policy on the `TimeLimit`ed
/// Pendulum, measured over 100 episodes from `seed`. Baseline for the Pendulum
/// learning check.
fn random_pendulum(seed: u64) -> f32 {
    let base = Pendulum::with_config(PendulumConfig {
        seed,
        ..PendulumConfig::default()
    });
    let mut env = TimeLimit::new(base, 200);
    let mut rng = StdRng::seed_from_u64(seed);
    random_return(&mut env, 100, 200, &mut rng, uniform_bounded::<1, PendulumAction>)
}

// Default-run convergence check: DDPG should clear the random baseline within
// 8k env steps. A uniform `U(-1, 1)` policy scores `random_linear`; an optimal
// policy approaches `0`. (Generated by `rl_learning_test!`.)
rl_learning_test! {
    #[ignore = "8 000-step LinearEnv convergence check; run with `cargo test -- --ignored`"]
    ddpg_linear_improves_over_random,
    improves_over_random(margin = 2.0),
    seed = 42,
    total = 8_000,
    run = run_linear,
    random = random_linear,
}

/// `act_with` (inner-backend greedy inference) must produce the same action as
/// `act(obs, false, _)` (the autodiff deterministic mean) for identical
/// weights — the inner snapshot is just a non-autodiff copy of the actor, so
/// the bench's faster greedy path stays faithful to the eval policy.
#[test]
fn ddpg_act_with_matches_deterministic_act() {
    let _guard = flex_guard();
    let seed: u64 = 7;
    let device = seeded_device::<Be>(seed);

    let actor: Actor<Be> = Actor::new(1, 32, 1, 1.0, &device);
    let critic: Critic<Be> = Critic::new(1, 1, 32, &device);
    let config = DdpgTrainingConfigBuilder::new()
        .buffer_capacity(20_000)
        .batch_size(32)
        .learning_starts(500)
        .actor_lr(1e-3)
        .critic_lr(1e-3)
        .gamma(0.99)
        .tau(0.02)
        .exploration_noise(0.2)
        .policy_frequency(2)
        .build();
    let agent: LinearAgent = DdpgAgent::new(actor, critic, config, device);

    let net = agent.inference_net();
    let mut rng = StdRng::seed_from_u64(seed);
    for &x in &[-0.9_f32, -0.3, 0.0, 0.25, 0.8] {
        let obs = LinearObservation { x };
        // training=false ⇒ deterministic mean + bound clip; rng is unused.
        let det = agent.act(&obs, false, &mut rng);
        let greedy = agent.act_with(&net, &obs);
        let a = det.as_slice()[0];
        let b = greedy.as_slice()[0];
        assert!((a - b).abs() < 1e-5, "x={x}: act(false)={a}, act_with={b}");
    }
}

// Seeded reproducibility: two same-seed runs must produce bit-equal metrics.
// (Generated by `rl_reproducibility_test!`.)
rl_reproducibility_test! {
    ddpg_linear_flex_reproducibility,
    bits,
    seed = 42,
    total = 1_000,
    run = run_linear,
}

/// Pendulum learning check: 50k steps with small networks; verifies training
/// runs without crashing and beats a measured uniform-random torque policy.
#[test]
#[ignore = "50 000-step continuous DDPG Pendulum run (~several minutes on CPU); confirms avg reward beats a measured uniform-random baseline — run with `cargo test -- --ignored`"]
fn ddpg_pendulum_improves_over_random() {
    let _guard = flex_guard();
    let seed: u64 = 42;
    let device = seeded_device::<Be>(seed);
    let base = Pendulum::with_config(PendulumConfig {
        seed,
        ..PendulumConfig::default()
    });
    let mut env = TimeLimit::new(base, 200);
    let mut rng = StdRng::seed_from_u64(seed);

    let actor: Actor<Be> = Actor::new(3, 64, 1, 2.0, &device);
    let critic: Critic<Be> = Critic::new(3, 1, 64, &device);
    let config = DdpgTrainingConfigBuilder::new()
        .buffer_capacity(20_000)
        .batch_size(64)
        .learning_starts(1_000)
        .actor_lr(1e-4)
        .critic_lr(1e-3)
        .gamma(0.99)
        .tau(0.005)
        .exploration_noise(0.1)
        .policy_frequency(2)
        .build();
    let mut agent: DdpgAgent<
        Be,
        Actor<Be>,
        Critic<Be>,
        PendulumObservation,
        PendulumAction,
        1,
        2,
        1,
        2,
    > = DdpgAgent::new(actor, critic, config, device);

    train::<Be, _, _, _, _, PendulumAction, _, 1, 1, 2, 1, 2>(
        &mut agent, &mut env, &mut rng, 50_000, 0,
    )
    .expect("training");

    let avg = agent.stats().avg_score().expect("non-empty history");
    // Compare against an actual uniform-random torque policy rather than a
    // hard-coded number; DDPG should clear it with margin after 50k steps.
    let baseline = random_pendulum(seed);
    assert_improves_over_random(avg, baseline, 50.0);
}
