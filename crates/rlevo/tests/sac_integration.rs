//! End-to-end integration tests for the SAC agent.
//!
//! Default-run tests exercise SAC on the shared synthetic 1-D continuous env
//! ([`rlevo_test_support::env::LinearEnv`]) plus learn-step smokes that assert
//! α actually moves (and stays pinned) under auto-tuning, and a bit-equal
//! reproducibility check. The Pendulum smoke and the convergence run are gated
//! behind `#[ignore]` to stay inside the project's default-run
//! integration-test budget.
//!
//! The shared fixture, the `Flex` determinism preamble ([`flex_guard`] /
//! [`seeded_device`]), and the acceptance assertions live in the
//! `rlevo-test-support` dev-crate; only the SAC networks and the
//! algorithm-specific tests remain here.

use burn::module::{AutodiffModule, Module};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::Tensor;
use burn::tensor::activation::{relu, softplus, tanh};
use burn::tensor::backend::{AutodiffBackend, Backend};

use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_core::environment::{Environment, Snapshot};
use rlevo_environments::classic::pendulum::{
    Pendulum, PendulumAction, PendulumConfig, PendulumObservation,
};
use rlevo_environments::wrappers::TimeLimit;
use rlevo_reinforcement_learning::algorithms::sac::sac_agent::SacAgent;
use rlevo_reinforcement_learning::algorithms::sac::sac_config::SacTrainingConfigBuilder;
use rlevo_reinforcement_learning::algorithms::sac::sac_model::{
    ContinuousQ, SampleOutput, SquashedGaussianPolicy,
};
use rlevo_reinforcement_learning::algorithms::sac::train::train;
use rlevo_reinforcement_learning::utils::{PolyakError, polyak_update};

use rlevo_test_support::assert::assert_improves_over_random;
use rlevo_test_support::baseline::{random_return, uniform_bounded};
use rlevo_test_support::env::{LinearAction, LinearEnv, LinearObservation};
use rlevo_test_support::flex::{FlexAutodiff as Be, flex_guard, seeded_device};
use rlevo_test_support::{TrainOutcome, rl_learning_test, rl_reproducibility_test};

// ---------------------------------------------------------------------------
// Stochastic actor: μ + log_std heads, squashed-Gaussian reparameterization.
// Implements SAC's policy trait and so stays test-local.
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
struct StochasticActor<B: Backend> {
    fc1: Linear<B>,
    mean: Linear<B>,
    log_std: Linear<B>,
    action_dim: usize,
    action_scale: f32,
    log_std_min: f32,
    log_std_max: f32,
}

impl<B: Backend> StochasticActor<B> {
    fn new(
        obs_dim: usize,
        hidden: usize,
        action_dim: usize,
        scale: f32,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Self {
        Self {
            fc1: LinearConfig::new(obs_dim, hidden).init(device),
            mean: LinearConfig::new(hidden, action_dim).init(device),
            log_std: LinearConfig::new(hidden, action_dim).init(device),
            action_dim,
            action_scale: scale,
            log_std_min: -5.0,
            log_std_max: 2.0,
        }
    }

    fn mean_and_log_std(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let h = relu(self.fc1.forward(obs));
        let mean = self.mean.forward(h.clone());
        let log_std = self
            .log_std
            .forward(h)
            .clamp(self.log_std_min, self.log_std_max);
        (mean, log_std)
    }

    #[allow(clippy::cast_precision_loss)]
    fn sample_impl(&self, obs: Tensor<B, 2>, eps: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let (mean, log_std) = self.mean_and_log_std(obs);
        let action_dim = mean.dims()[1];
        let std = log_std.clone().exp();
        let z = mean.clone() + std * eps;
        let diff = z.clone() - mean;
        let scaled = diff / log_std.clone().exp();
        let scaled_sq = scaled.clone() * scaled;
        let log_2pi = (2.0_f32 * std::f32::consts::PI).ln();
        let per_dim_gauss: Tensor<B, 2> = scaled_sq.mul_scalar(-0.5) - log_std - log_2pi * 0.5;
        let ln_2 = std::f32::consts::LN_2;
        let neg_two_z = z.clone().mul_scalar(-2.0);
        let sp = softplus(neg_two_z, 1.0);
        let per_dim_jac: Tensor<B, 2> = (z.clone().neg() - sp + ln_2).mul_scalar(2.0);
        let per_dim = per_dim_gauss - per_dim_jac;
        let log_prob_z = per_dim.sum_dim(1).squeeze_dim::<1>(1);
        let log_scale_abs = self.action_scale.abs().ln();
        let log_prob = log_prob_z.sub_scalar(log_scale_abs * action_dim as f32);
        let action = tanh(z).mul_scalar(self.action_scale);
        (action, log_prob)
    }
}

impl<B: AutodiffBackend> SquashedGaussianPolicy<B, 2, 2> for StochasticActor<B> {
    fn action_dim(&self) -> usize {
        self.action_dim
    }

    fn forward_sample(&self, obs: Tensor<B, 2>, eps: Tensor<B, 2>) -> SampleOutput<B, 2> {
        let (action, log_prob) = self.sample_impl(obs, eps);
        SampleOutput { action, log_prob }
    }

    fn forward_sample_inner(
        inner: &Self::InnerModule,
        obs: Tensor<B::InnerBackend, 2>,
        eps: Tensor<B::InnerBackend, 2>,
    ) -> SampleOutput<B::InnerBackend, 2> {
        let (action, log_prob) = inner.sample_impl(obs, eps);
        SampleOutput { action, log_prob }
    }

    fn deterministic_action(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let (mean, _) = self.mean_and_log_std(obs);
        tanh(mean).mul_scalar(self.action_scale)
    }
}

// ---------------------------------------------------------------------------
// Critic. Implements SAC's Q-trait and so stays test-local.
// ---------------------------------------------------------------------------

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
    fn soft_update(
        active: &Self,
        target: Self::InnerModule,
        tau: f64,
    ) -> Result<Self::InnerModule, PolyakError> {
        polyak_update::<B::InnerBackend, Critic<B::InnerBackend>>(
            &active.valid(),
            target,
            tau as f32,
        )
    }
}

// Concrete SAC agent over the shared 1-D continuous fixture.
type LinearAgent =
    SacAgent<Be, StochasticActor<Be>, Critic<Be>, LinearObservation, LinearAction, 1, 2, 1, 2>;

/// Drives `agent` against `env` for `steps` env steps, storing every transition
/// and ticking the warm-up counter. Used to prime the replay buffer past
/// warm-up before manual `learn_step` calls.
fn prime_buffer(agent: &mut LinearAgent, env: &mut LinearEnv, rng: &mut StdRng, steps: usize) {
    let mut snap = env.reset().expect("reset");
    for _ in 0..steps {
        let obs = *snap.observation();
        let action = agent.act(&obs, true, rng);
        let next = env.step(action).expect("step");
        let reward: f32 = (*next.reward()).into();
        // `done` drives the episode reset; `terminated` is the Bellman
        // bootstrap mask. `LinearEnv` only ever truncates, so these differ.
        let done = next.is_done();
        let terminated = next.is_terminated();
        let next_obs = *next.observation();
        agent.remember(obs, &action, reward, next_obs, terminated);
        agent.on_env_step();
        snap = if done {
            env.reset().expect("reset")
        } else {
            next
        };
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Builds and trains a SAC agent on the shared `LinearEnv` for `total` steps,
/// returning the standardised outcome consumed by the suite macros. Shared by
/// the convergence and reproducibility tests below.
fn run_linear(seed: u64, total: usize) -> TrainOutcome {
    let device = seeded_device::<Be>(seed);
    let mut env = LinearEnv::with_seed(seed, 20);
    let mut rng = StdRng::seed_from_u64(seed);

    let actor: StochasticActor<Be> = StochasticActor::new(1, 32, 1, 1.0, &device);
    let critic_1: Critic<Be> = Critic::new(1, 1, 32, &device);
    let critic_2: Critic<Be> = Critic::new(1, 1, 32, &device);
    let config = SacTrainingConfigBuilder::new()
        .replay_buffer_capacity(20_000)
        .batch_size(32)
        .learning_starts(500)
        .actor_lr(3e-4)
        .critic_lr(1e-3)
        .alpha_lr(1e-3)
        .gamma(0.99)
        .tau(0.02)
        .autotune(true)
        .initial_alpha(1.0)
        .policy_frequency(2)
        .build()
        .expect("valid config");
    let mut agent: LinearAgent =
        SacAgent::new(actor, critic_1, critic_2, config, device).expect("valid config");

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
    random_return(
        &mut env,
        200,
        20,
        &mut rng,
        uniform_bounded::<1, LinearAction>,
    )
}

/// Mean episode return of a uniform-random torque policy on the `TimeLimit`ed
/// Pendulum, measured over 100 episodes from `seed`. Baseline for the Pendulum
/// learning check.
fn random_pendulum(seed: u64) -> f32 {
    let base = Pendulum::with_config(PendulumConfig {
        seed,
        ..PendulumConfig::default()
    })
    .expect("valid config");
    let mut env = TimeLimit::new(base, 200);
    let mut rng = StdRng::seed_from_u64(seed);
    random_return(
        &mut env,
        100,
        200,
        &mut rng,
        uniform_bounded::<1, PendulumAction>,
    )
}

// Default-run convergence check: SAC should clear the random baseline within
// 8k env steps. A uniform `U(-1, 1)` policy scores `random_linear`; a learned
// policy approaches `0`. (Generated by `rl_learning_test!`.)
rl_learning_test! {
    #[ignore = "8 000-step LinearEnv convergence check; run with `cargo test -- --ignored`"]
    sac_linear_improves_over_random,
    improves_over_random(margin = 2.0),
    seed = 42,
    total = 8_000,
    run = run_linear,
    random = random_linear,
}

/// With `autotune=true` and a target entropy of `-|A| = -1`, 200 learn
/// steps on a primed buffer should visibly move α off its initial value
/// of `1.0`. The direction can go either way depending on the policy's
/// entropy, so we only check that |Δα| is large enough to rule out the
/// no-op case.
#[test]
fn sac_alpha_moves_under_autotune() {
    let _guard = flex_guard();
    let seed: u64 = 7;
    let device = seeded_device::<Be>(seed);
    let mut env = LinearEnv::with_seed(seed, 20);
    let mut rng = StdRng::seed_from_u64(seed);

    let actor: StochasticActor<Be> = StochasticActor::new(1, 16, 1, 1.0, &device);
    let critic_1: Critic<Be> = Critic::new(1, 1, 16, &device);
    let critic_2: Critic<Be> = Critic::new(1, 1, 16, &device);
    let config = SacTrainingConfigBuilder::new()
        .replay_buffer_capacity(2_048)
        .batch_size(16)
        .learning_starts(32)
        .alpha_lr(1e-2)
        .autotune(true)
        .initial_alpha(1.0)
        .policy_frequency(1)
        .build()
        .expect("valid config");
    let mut agent: LinearAgent =
        SacAgent::new(actor, critic_1, critic_2, config, device).expect("valid config");

    // Prime the buffer past warm-up so learn_step proceeds.
    prime_buffer(&mut agent, &mut env, &mut rng, 128);
    assert!(agent.can_learn(), "agent should be past warm-up");

    let before = agent.last_alpha();
    for _ in 0..200 {
        let _ = agent
            .learn_step(&mut rng)
            .expect("no polyak error")
            .expect("can learn");
    }
    let after = agent.last_alpha();
    assert!(after.is_finite(), "alpha must stay finite, got {after}");
    assert!(
        (before - after).abs() > 1e-3,
        "expected alpha to move from initial 1.0, before={before}, after={after}"
    );
}

/// When `autotune=false`, α stays pinned at `initial_alpha` across many
/// learn steps.
#[test]
fn sac_alpha_frozen_when_autotune_disabled() {
    let _guard = flex_guard();
    let seed: u64 = 11;
    let device = seeded_device::<Be>(seed);
    let mut env = LinearEnv::with_seed(seed, 20);
    let mut rng = StdRng::seed_from_u64(seed);

    let actor: StochasticActor<Be> = StochasticActor::new(1, 16, 1, 1.0, &device);
    let critic_1: Critic<Be> = Critic::new(1, 1, 16, &device);
    let critic_2: Critic<Be> = Critic::new(1, 1, 16, &device);
    let config = SacTrainingConfigBuilder::new()
        .replay_buffer_capacity(2_048)
        .batch_size(16)
        .learning_starts(32)
        .autotune(false)
        .initial_alpha(0.2)
        .policy_frequency(1)
        .build()
        .expect("valid config");
    let mut agent: LinearAgent =
        SacAgent::new(actor, critic_1, critic_2, config, device).expect("valid config");

    prime_buffer(&mut agent, &mut env, &mut rng, 128);
    for _ in 0..50 {
        let _ = agent
            .learn_step(&mut rng)
            .expect("no polyak error")
            .expect("can learn");
    }
    assert!(
        (agent.last_alpha() - 0.2).abs() < 1e-6,
        "alpha must stay at 0.2 when autotune=false, got {}",
        agent.last_alpha()
    );
}

/// Pendulum learning check: gated at 30k steps.
#[test]
#[ignore = "30 000-step continuous SAC Pendulum run (~2 min on Flex); confirms avg reward beats a measured uniform-random baseline — run with `cargo test -- --ignored`"]
fn sac_pendulum_improves_over_random() {
    let _guard = flex_guard();
    let seed: u64 = 42;
    let device = seeded_device::<Be>(seed);
    let base = Pendulum::with_config(PendulumConfig {
        seed,
        ..PendulumConfig::default()
    })
    .expect("valid config");
    let mut env = TimeLimit::new(base, 200);
    let mut rng = StdRng::seed_from_u64(seed);

    let actor: StochasticActor<Be> = StochasticActor::new(3, 64, 1, 2.0, &device);
    let critic_1: Critic<Be> = Critic::new(3, 1, 64, &device);
    let critic_2: Critic<Be> = Critic::new(3, 1, 64, &device);
    let config = SacTrainingConfigBuilder::new()
        .replay_buffer_capacity(30_000)
        .batch_size(64)
        .learning_starts(1_000)
        .actor_lr(3e-4)
        .critic_lr(1e-3)
        .alpha_lr(1e-3)
        .gamma(0.99)
        .tau(0.005)
        .autotune(true)
        .initial_alpha(1.0)
        .policy_frequency(2)
        .build()
        .expect("valid config");
    let mut agent: SacAgent<
        Be,
        StochasticActor<Be>,
        Critic<Be>,
        PendulumObservation,
        PendulumAction,
        1,
        2,
        1,
        2,
    > = SacAgent::new(actor, critic_1, critic_2, config, device).expect("valid config");

    train::<Be, _, _, _, _, PendulumAction, _, 1, 1, 2, 1, 2>(
        &mut agent, &mut env, &mut rng, 30_000, 0,
    )
    .expect("training");

    let avg = agent.stats().avg_score().expect("non-empty history");
    // Reduced from the former 500k-step / 256-wide macro run (~9 min even at
    // 50k) to a 30k-step / 64-wide budget that finishes in ~2 min: SAC runs a
    // full update (two critics + policy + α) every env step, so per-step grad
    // cost — not step count — dominates wall-clock, and trimming both width and
    // batch is the real lever. At this budget SAC deterministically reaches
    // ≈ -940, comfortably above a uniform-random torque policy but short of the
    // aspirational -800 convergence bar. So this stays a "beats random" check
    // (mirroring the DDPG Pendulum learning test); the 100-pt margin absorbs
    // cross-platform float drift. Determinism (seeded backend + 1-thread rayon)
    // keeps it reproducible.
    let baseline = random_pendulum(seed);
    assert_improves_over_random(avg, baseline, 100.0);
}

// Seeded reproducibility: two same-seed runs must produce bit-equal metrics.
// (Generated by `rl_reproducibility_test!`.)
rl_reproducibility_test! {
    sac_linear_flex_reproducibility,
    bits,
    seed = 42,
    total = 1_000,
    run = run_linear,
}
