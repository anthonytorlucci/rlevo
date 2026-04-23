//! End-to-end integration tests for the SAC agent.
//!
//! Default-run tests exercise SAC on a minimal synthetic 1-D continuous env
//! (shared with TD3 / DDPG) plus a learn-step smoke that asserts α actually
//! moves under auto-tuning. The Pendulum macro-smoke and the bit-equal
//! reproducibility check are gated behind `#[ignore]` to stay inside the
//! project's default-run integration-test budget.

use std::collections::HashMap;

use burn::backend::{Autodiff, NdArray};
use burn::module::{AutodiffModule, Module, ModuleMapper, ModuleVisitor, Param, ParamId};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{relu, softplus, tanh};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Tensor, TensorData};
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use rlevo_core::action::{BoundedAction, ContinuousAction};
use rlevo_core::base::{Action, Observation, State, TensorConversionError, TensorConvertible};
use rlevo_core::environment::{
    Environment, EnvironmentError, EpisodeStatus, Snapshot, SnapshotBase,
};
use rlevo_core::reward::ScalarReward;
use rlevo_envs::classic::pendulum::{
    Pendulum, PendulumAction, PendulumConfig, PendulumObservation,
};
use rlevo_envs::wrappers::TimeLimit;
use rlevo_rl::algorithms::sac::sac_agent::SacAgent;
use rlevo_rl::algorithms::sac::sac_config::SacTrainingConfigBuilder;
use rlevo_rl::algorithms::sac::sac_model::{ContinuousQ, SampleOutput, SquashedGaussianPolicy};
use rlevo_rl::algorithms::sac::train::train;

// ---------------------------------------------------------------------------
// Synthetic 1-D continuous environment (same as `td3_integration.rs`).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
struct LinearState {
    x: f32,
    steps: usize,
}

impl State<1> for LinearState {
    type Observation = LinearObservation;
    fn shape() -> [usize; 1] {
        [1]
    }
    fn numel(&self) -> usize {
        1
    }
    fn is_valid(&self) -> bool {
        self.x.is_finite()
    }
    fn observe(&self) -> LinearObservation {
        LinearObservation { x: self.x }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct LinearObservation {
    x: f32,
}

impl Observation<1> for LinearObservation {
    fn shape() -> [usize; 1] {
        [1]
    }
}

impl<B: Backend> TensorConvertible<1, B> for LinearObservation {
    fn to_tensor(&self, device: &B::Device) -> Tensor<B, 1> {
        Tensor::from_data(TensorData::new(vec![self.x], vec![1]), device)
    }
    fn from_tensor(tensor: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        let v = tensor.into_data().convert::<f32>();
        Ok(Self {
            x: v.as_slice::<f32>().unwrap()[0],
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct LinearAction(f32);

impl Action<1> for LinearAction {
    fn shape() -> [usize; 1] {
        [1]
    }
    fn is_valid(&self) -> bool {
        self.0.is_finite() && self.0.abs() <= 1.0
    }
}

impl ContinuousAction<1> for LinearAction {
    fn as_slice(&self) -> &[f32] {
        std::slice::from_ref(&self.0)
    }
    fn clip(&self, min: f32, max: f32) -> Self {
        Self(self.0.clamp(min, max))
    }
    fn from_slice(values: &[f32]) -> Self {
        assert_eq!(values.len(), 1);
        Self(values[0])
    }
}

impl BoundedAction<1> for LinearAction {
    fn low() -> [f32; 1] {
        [-1.0]
    }
    fn high() -> [f32; 1] {
        [1.0]
    }
}

struct LinearEnv {
    state: LinearState,
    rng: StdRng,
    episode_len: usize,
}

impl LinearEnv {
    fn with_seed(seed: u64, episode_len: usize) -> Self {
        Self {
            state: LinearState { x: 0.0, steps: 0 },
            rng: StdRng::seed_from_u64(seed),
            episode_len,
        }
    }

    fn sample_x(rng: &mut StdRng) -> f32 {
        use rand::RngExt;
        rng.random_range(-1.0_f32..=1.0_f32)
    }
}

impl Environment<1, 1, 1> for LinearEnv {
    type StateType = LinearState;
    type ObservationType = LinearObservation;
    type ActionType = LinearAction;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<1, LinearObservation, ScalarReward>;

    fn new(_render: bool) -> Self {
        Self::with_seed(0, 20)
    }

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.state = LinearState {
            x: Self::sample_x(&mut self.rng),
            steps: 0,
        };
        Ok(SnapshotBase {
            observation: self.state.observe(),
            reward: ScalarReward::new(0.0),
            status: EpisodeStatus::Running,
        })
    }

    fn step(&mut self, action: Self::ActionType) -> Result<Self::SnapshotType, EnvironmentError> {
        let a = action.0.clamp(-1.0, 1.0);
        let err = a - self.state.x;
        let reward = -(err * err);
        let next_x = Self::sample_x(&mut self.rng);
        self.state = LinearState {
            x: next_x,
            steps: self.state.steps + 1,
        };
        let status = if self.state.steps >= self.episode_len {
            EpisodeStatus::Truncated
        } else {
            EpisodeStatus::Running
        };
        Ok(SnapshotBase {
            observation: self.state.observe(),
            reward: ScalarReward::new(reward),
            status,
        })
    }
}

// ---------------------------------------------------------------------------
// Stochastic actor: μ + log_std heads, squashed-Gaussian reparameterization.
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
        device: &B::Device,
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
// Critic + Polyak (shared with td3_integration).
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
struct Critic<B: Backend> {
    fc1: Linear<B>,
    head: Linear<B>,
}

impl<B: Backend> Critic<B> {
    fn new(obs_dim: usize, action_dim: usize, hidden: usize, device: &B::Device) -> Self {
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
    fn soft_update(active: &Self, target: Self::InnerModule, tau: f64) -> Self::InnerModule {
        polyak_update::<B::InnerBackend, Critic<B::InnerBackend>>(
            active.valid(),
            target,
            tau as f32,
        )
    }
}

struct ParamCollector<B: Backend> {
    tensors: HashMap<ParamId, TensorData>,
    _marker: std::marker::PhantomData<B>,
}
impl<B: Backend> ModuleVisitor<B> for ParamCollector<B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        self.tensors.insert(param.id, param.val().to_data());
    }
}
struct PolyakMapper<B: Backend> {
    active: HashMap<ParamId, TensorData>,
    tau: f32,
    _marker: std::marker::PhantomData<B>,
}
impl<B: Backend> ModuleMapper<B> for PolyakMapper<B> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let id = param.id;
        let active = self.active.remove(&id).expect("paired active param");
        let tau = self.tau;
        param.map(move |t| {
            let device = t.device();
            let a = Tensor::<B, D>::from_data(active, &device);
            t.mul_scalar(1.0 - tau) + a.mul_scalar(tau)
        })
    }
}
fn polyak_update<B: Backend, M: Module<B>>(active: M, target: M, tau: f32) -> M {
    let mut c = ParamCollector::<B> {
        tensors: HashMap::new(),
        _marker: std::marker::PhantomData,
    };
    active.visit(&mut c);
    let mut m = PolyakMapper::<B> {
        active: c.tensors,
        tau,
        _marker: std::marker::PhantomData,
    };
    target.map(&mut m)
}

type Be = Autodiff<NdArray>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Default-run test: SAC should solve the 1-D tracking task well enough that
/// the 100-episode moving average clears a lax `-1.0` threshold within 8k
/// env steps. Random actions average ≈ −6.67 per episode; a learned policy
/// approaches `0`.
#[test]
fn sac_solves_linear_1d_continuous() {
    let seed: u64 = 42;
    let device = Default::default();
    <Be as Backend>::seed(&device, seed);
    let mut env = LinearEnv::with_seed(seed, 20);
    let mut rng = StdRng::seed_from_u64(seed);

    let actor: StochasticActor<Be> = StochasticActor::new(1, 32, 1, 1.0, &device);
    let critic_1: Critic<Be> = Critic::new(1, 1, 32, &device);
    let critic_2: Critic<Be> = Critic::new(1, 1, 32, &device);
    let config = SacTrainingConfigBuilder::new()
        .buffer_capacity(20_000)
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
        .build();
    let mut agent: SacAgent<
        Be,
        StochasticActor<Be>,
        Critic<Be>,
        LinearObservation,
        LinearAction,
        1,
        2,
        1,
        2,
    > = SacAgent::new(actor, critic_1, critic_2, config, device);

    train::<Be, _, _, _, _, LinearAction, _, 1, 1, 2, 1, 2>(
        &mut agent, &mut env, &mut rng, 8_000, 0,
    )
    .expect("training");

    let avg = agent.stats().avg_score().expect("non-empty history");
    assert!(avg.is_finite(), "avg reward must be finite, got {avg}");
    assert!(
        avg > -1.0,
        "expected avg reward > -1.0, got {avg:.3} (random baseline ≈ -6.67)"
    );
}

/// With `autotune=true` and a target entropy of `-|A| = -1`, 200 learn
/// steps on a primed buffer should visibly move α off its initial value
/// of `1.0`. The direction can go either way depending on the policy's
/// entropy, so we only check that |Δα| is large enough to rule out the
/// no-op case.
#[test]
fn sac_alpha_moves_under_autotune() {
    let seed: u64 = 7;
    let device = Default::default();
    <Be as Backend>::seed(&device, seed);
    let mut env = LinearEnv::with_seed(seed, 20);
    let mut rng = StdRng::seed_from_u64(seed);

    let actor: StochasticActor<Be> = StochasticActor::new(1, 16, 1, 1.0, &device);
    let critic_1: Critic<Be> = Critic::new(1, 1, 16, &device);
    let critic_2: Critic<Be> = Critic::new(1, 1, 16, &device);
    let config = SacTrainingConfigBuilder::new()
        .buffer_capacity(2_048)
        .batch_size(16)
        .learning_starts(32)
        .alpha_lr(1e-2)
        .autotune(true)
        .initial_alpha(1.0)
        .policy_frequency(1)
        .build();
    let mut agent: SacAgent<
        Be,
        StochasticActor<Be>,
        Critic<Be>,
        LinearObservation,
        LinearAction,
        1,
        2,
        1,
        2,
    > = SacAgent::new(actor, critic_1, critic_2, config, device);

    // Prime the buffer past warm-up so learn_step proceeds.
    let mut snap = env.reset().expect("reset");
    for _ in 0..128 {
        let obs = *snap.observation();
        let action = agent.act(&obs, true, &mut rng);
        let next = env.step(action).expect("step");
        let reward: f32 = (*next.reward()).into();
        let done = next.is_done();
        let next_obs = *next.observation();
        agent.remember(obs, &action, reward, next_obs, done);
        agent.on_env_step();
        snap = if done {
            env.reset().expect("reset")
        } else {
            next
        };
    }
    assert!(agent.can_learn(), "agent should be past warm-up");

    let before = agent.last_alpha();
    for _ in 0..200 {
        let _ = agent.learn_step(&mut rng).expect("can learn");
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
    let seed: u64 = 11;
    let device = Default::default();
    <Be as Backend>::seed(&device, seed);
    let mut env = LinearEnv::with_seed(seed, 20);
    let mut rng = StdRng::seed_from_u64(seed);

    let actor: StochasticActor<Be> = StochasticActor::new(1, 16, 1, 1.0, &device);
    let critic_1: Critic<Be> = Critic::new(1, 1, 16, &device);
    let critic_2: Critic<Be> = Critic::new(1, 1, 16, &device);
    let config = SacTrainingConfigBuilder::new()
        .buffer_capacity(2_048)
        .batch_size(16)
        .learning_starts(32)
        .autotune(false)
        .initial_alpha(0.2)
        .policy_frequency(1)
        .build();
    let mut agent: SacAgent<
        Be,
        StochasticActor<Be>,
        Critic<Be>,
        LinearObservation,
        LinearAction,
        1,
        2,
        1,
        2,
    > = SacAgent::new(actor, critic_1, critic_2, config, device);

    let mut snap = env.reset().expect("reset");
    for _ in 0..128 {
        let obs = *snap.observation();
        let action = agent.act(&obs, true, &mut rng);
        let next = env.step(action).expect("step");
        let reward: f32 = (*next.reward()).into();
        let done = next.is_done();
        let next_obs = *next.observation();
        agent.remember(obs, &action, reward, next_obs, done);
        agent.on_env_step();
        snap = if done {
            env.reset().expect("reset")
        } else {
            next
        };
    }
    for _ in 0..50 {
        let _ = agent.learn_step(&mut rng).expect("can learn");
    }
    assert!(
        (agent.last_alpha() - 0.2).abs() < 1e-6,
        "alpha must stay at 0.2 when autotune=false, got {}",
        agent.last_alpha()
    );
}

/// Pendulum macro-smoke: gated at 500k steps, `--test-threads=1` so the
/// ndarray backend's global RNG stays isolated.
#[test]
#[ignore = "macro run (~500k Pendulum steps); --test-threads=1 for isolated Burn RNG"]
fn sac_pendulum_smoke() {
    let seed: u64 = 42;
    let device = Default::default();
    <Be as Backend>::seed(&device, seed);
    let base = Pendulum::with_config(PendulumConfig {
        seed,
        ..PendulumConfig::default()
    });
    let mut env = TimeLimit::new(base, 200);
    let mut rng = StdRng::seed_from_u64(seed);

    let actor: StochasticActor<Be> = StochasticActor::new(3, 256, 1, 2.0, &device);
    let critic_1: Critic<Be> = Critic::new(3, 1, 256, &device);
    let critic_2: Critic<Be> = Critic::new(3, 1, 256, &device);
    let config = SacTrainingConfigBuilder::new()
        .buffer_capacity(200_000)
        .batch_size(256)
        .learning_starts(10_000)
        .actor_lr(3e-4)
        .critic_lr(1e-3)
        .alpha_lr(1e-3)
        .gamma(0.99)
        .tau(0.005)
        .autotune(true)
        .initial_alpha(1.0)
        .policy_frequency(2)
        .build();
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
    > = SacAgent::new(actor, critic_1, critic_2, config, device);

    train::<Be, _, _, _, _, PendulumAction, _, 1, 1, 2, 1, 2>(
        &mut agent, &mut env, &mut rng, 500_000, 0,
    )
    .expect("training");

    let avg = agent.stats().avg_score().expect("non-empty history");
    assert!(avg.is_finite(), "avg reward must be finite, got {avg}");
    // Zero-torque Pendulum scores ≈ -1200; SAC should comfortably beat
    // that after 500k steps. The tighter -150 acceptance target stays
    // aspirational, mirroring the DDPG/TD3 macro-smoke thresholds.
    assert!(avg > -800.0, "expected avg reward > -800, got {avg:.2}");
}

/// Seeded reproducibility: two identical runs on the same seed must produce
/// bit-equal metrics. Gated — shares the ndarray backend's global RNG with
/// other tests.
#[test]
#[ignore = "reproducibility run; --test-threads=1 for isolated Burn RNG"]
fn sac_reproducibility_ndarray() {
    fn run_once() -> f32 {
        let seed: u64 = 42;
        let device = Default::default();
        <Be as Backend>::seed(&device, seed);
        let mut env = LinearEnv::with_seed(seed, 20);
        let mut rng = StdRng::seed_from_u64(seed);
        let actor: StochasticActor<Be> = StochasticActor::new(1, 16, 1, 1.0, &device);
        let c1: Critic<Be> = Critic::new(1, 1, 16, &device);
        let c2: Critic<Be> = Critic::new(1, 1, 16, &device);
        let config = SacTrainingConfigBuilder::new()
            .buffer_capacity(2_048)
            .batch_size(16)
            .learning_starts(64)
            .policy_frequency(1)
            .build();
        let mut agent: SacAgent<
            Be,
            StochasticActor<Be>,
            Critic<Be>,
            LinearObservation,
            LinearAction,
            1,
            2,
            1,
            2,
        > = SacAgent::new(actor, c1, c2, config, device);
        train::<Be, _, _, _, _, LinearAction, _, 1, 1, 2, 1, 2>(
            &mut agent, &mut env, &mut rng, 1_000, 0,
        )
        .expect("training");
        agent.stats().avg_score().unwrap_or(0.0)
    }

    let a = run_once();
    let b = run_once();
    assert_eq!(
        a.to_bits(),
        b.to_bits(),
        "SAC run not reproducible: {a} vs {b}"
    );
}
